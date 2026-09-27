from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from ..util import loads, sha1_text
from .store import Store

CURRENT = "current"
BASE = "base"
HEAD = "head"


def change_state(row) -> str:
    if row["version"] == HEAD:
        return "after_change"
    if row["superseded"]:
        return "before_change"
    return "unchanged"


@dataclass
class ViewFile:
    path: str
    blob_sha: str
    version: str
    superseded: bool
    indexed: bool
    reason: str | None


class View:
    def __init__(self, store: Store, view_key: str):
        row = store.query_one("SELECT * FROM views WHERE view_key = ?", (view_key,))
        if row is None:
            raise KeyError(f"unknown view {view_key}")
        self.store = store
        self.view_key = view_key
        self.root_key = row["root_key"] or view_key
        self.base_sha = row["base_sha"]
        self.head_sha = row["head_sha"]
        self.label = row["label"]
        self.is_overlay = self.root_key != view_key
        self.active_table = f"active_{sha1_text(view_key)[:12]}"
        self.overridden_table = f"overridden_{sha1_text(view_key)[:12]}"
        self._files: dict[tuple[str, str], ViewFile] | None = None
        self.materialize()

    def load_files(self) -> dict[tuple[str, str], ViewFile]:
        if self._files is not None:
            return self._files
        files: dict[tuple[str, str], ViewFile] = {}
        root_rows = self.store.query("SELECT * FROM view_files WHERE view_key = ?", (self.root_key,))
        overlay_rows = []
        if self.is_overlay:
            overlay_rows = self.store.query("SELECT * FROM view_files WHERE view_key = ?", (self.view_key,))
        overridden = {row["path"] for row in overlay_rows}
        for row in root_rows:
            if row["path"] in overridden:
                continue
            files[(row["path"], row["version"])] = ViewFile(
                row["path"], row["blob_sha"], row["version"], bool(row["superseded"]), bool(row["indexed"]), row["reason"]
            )
        for row in overlay_rows:
            files[(row["path"], row["version"])] = ViewFile(
                row["path"], row["blob_sha"], row["version"], bool(row["superseded"]), bool(row["indexed"]), row["reason"]
            )
        self._files = files
        return files

    def materialize(self) -> None:
        files = self.load_files()
        with self.store.lock:
            connection = self.store.connection
            connection.execute(f"DROP TABLE IF EXISTS temp.{self.active_table}")
            connection.execute(
                f"CREATE TEMP TABLE {self.active_table} (path TEXT, blob_sha TEXT, version TEXT, superseded INTEGER, "
                f"PRIMARY KEY (path, blob_sha, version))"
            )
            rows = []
            for view_file in files.values():
                if view_file.indexed:
                    rows.append((view_file.path, view_file.blob_sha, view_file.version, int(view_file.superseded)))
            connection.executemany(f"INSERT OR IGNORE INTO {self.active_table} VALUES (?, ?, ?, ?)", rows)
            connection.execute(f"DROP TABLE IF EXISTS temp.{self.overridden_table}")
            connection.execute(f"CREATE TEMP TABLE {self.overridden_table} (path TEXT PRIMARY KEY)")
            connection.executemany(
                f"INSERT OR IGNORE INTO {self.overridden_table} VALUES (?)",
                [(path,) for path in self.changed_paths()],
            )

    def changed_paths(self) -> set[str]:
        if not self.is_overlay:
            return set()
        rows = self.store.query("SELECT DISTINCT path FROM view_files WHERE view_key = ?", (self.view_key,))
        return {row["path"] for row in rows}

    def file(self, path: str, version: str = CURRENT) -> ViewFile | None:
        files = self.load_files()
        if version == BASE:
            candidate = files.get((path, BASE))
            return candidate
        head_file = files.get((path, HEAD))
        if head_file is not None:
            return head_file
        if version == HEAD:
            return None
        base_file = files.get((path, BASE))
        if base_file is not None and not base_file.superseded:
            return base_file
        return None

    def all_paths(self, version: str = CURRENT) -> list[str]:
        paths = set()
        for view_file in self.load_files().values():
            if version == CURRENT and view_file.superseded:
                continue
            if version == BASE and view_file.version != BASE:
                continue
            if version == HEAD and view_file.version != HEAD:
                continue
            paths.add(view_file.path)
        return sorted(paths)

    def version_clause(self, version: str, alias: str = "a") -> str:
        if version == BASE:
            return f"{alias}.version = 'base'"
        if version == HEAD:
            return f"{alias}.version = 'head'"
        return f"{alias}.superseded = 0"

    def chunk_rows(self, where: str = "1=1", parameters=(), version: str = CURRENT, limit: int | None = None, order: str = "") -> list[sqlite3.Row]:
        sql = (
            f"SELECT c.*, a.version AS version, a.superseded AS superseded FROM chunks c "
            f"JOIN {self.active_table} a ON c.path = a.path AND c.blob_sha = a.blob_sha "
            f"WHERE {self.version_clause(version)} AND ({where})"
        )
        if order:
            sql += f" ORDER BY {order}"
        if limit:
            sql += f" LIMIT {int(limit)}"
        return self.store.query(sql, parameters)

    def chunk_by_row_id(self, row_id: int) -> sqlite3.Row | None:
        rows = self.store.query(
            f"SELECT c.*, a.version AS version, a.superseded AS superseded FROM chunks c "
            f"JOIN {self.active_table} a ON c.path = a.path AND c.blob_sha = a.blob_sha WHERE c.id = ?",
            (row_id,),
        )
        return rows[0] if rows else None

    def chunks_for_ids(self, chunk_ids: list[str], version: str = CURRENT) -> list[sqlite3.Row]:
        if not chunk_ids:
            return []
        placeholders = ",".join("?" for _ in chunk_ids)
        return self.chunk_rows(f"c.chunk_id IN ({placeholders})", tuple(chunk_ids), version=version)

    def chunks_for_row_ids(self, row_ids: list[int]) -> list[sqlite3.Row]:
        if not row_ids:
            return []
        placeholders = ",".join("?" for _ in row_ids)
        return self.store.query(
            f"SELECT c.*, a.version AS version, a.superseded AS superseded FROM chunks c "
            f"JOIN {self.active_table} a ON c.path = a.path AND c.blob_sha = a.blob_sha WHERE c.id IN ({placeholders})",
            tuple(row_ids),
        )

    def edge_rows(self, where: str, parameters=(), version: str = CURRENT) -> list[sqlite3.Row]:
        edge_view = self.root_key if version == BASE else self.view_key
        return self.store.query(f"SELECT * FROM edges WHERE view_key = ? AND ({where})", (edge_view, *parameters))

    def edges_to(self, chunk_ids: list[str], kinds: list[str] | None = None, version: str = CURRENT) -> list[sqlite3.Row]:
        if not chunk_ids:
            return []
        placeholders = ",".join("?" for _ in chunk_ids)
        where = f"dst_chunk_id IN ({placeholders})"
        parameters: list = list(chunk_ids)
        if kinds:
            where += f" AND kind IN ({','.join('?' for _ in kinds)})"
            parameters.extend(kinds)
        return self.edge_rows(where, tuple(parameters), version)

    def edges_from(self, chunk_ids: list[str], kinds: list[str] | None = None, version: str = CURRENT) -> list[sqlite3.Row]:
        if not chunk_ids:
            return []
        placeholders = ",".join("?" for _ in chunk_ids)
        where = f"src_chunk_id IN ({placeholders})"
        parameters: list = list(chunk_ids)
        if kinds:
            where += f" AND kind IN ({','.join('?' for _ in kinds)})"
            parameters.extend(kinds)
        return self.edge_rows(where, tuple(parameters), version)

    def tags_key(self) -> str:
        return self.view_key

    def derived_tags(self, chunk_ids: list[str]) -> dict[str, set[str]]:
        result: dict[str, set[str]] = {}
        if not chunk_ids:
            return result
        placeholders = ",".join("?" for _ in chunk_ids)
        rows = self.store.query(
            f"SELECT chunk_id, tag FROM chunk_tags WHERE view_key = ? AND chunk_id IN ({placeholders})",
            (self.tags_key(), *chunk_ids),
        )
        for row in rows:
            result.setdefault(row["chunk_id"], set()).add(row["tag"])
        return result

    def all_tags(self, row: sqlite3.Row, derived: dict[str, set[str]] | None = None) -> list[str]:
        tags = set(loads(row["tags"], []))
        if derived is not None:
            tags.update(derived.get(row["chunk_id"], set()))
        return sorted(tags)

    def url_routes_for(self, view_chunk_ids: list[str]) -> list[sqlite3.Row]:
        if not view_chunk_ids:
            return []
        placeholders = ",".join("?" for _ in view_chunk_ids)
        return self.store.query(
            f"SELECT * FROM url_routes WHERE view_key = ? AND view_chunk_id IN ({placeholders})",
            (self.view_key, *view_chunk_ids),
        )

    def all_url_routes(self) -> list[sqlite3.Row]:
        return self.store.query("SELECT * FROM url_routes WHERE view_key = ?", (self.view_key,))
