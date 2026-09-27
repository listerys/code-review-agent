from __future__ import annotations

import re
import sqlite3
import struct
import threading
from datetime import datetime, UTC
from pathlib import Path

import sqlite_vec

from ..util import dumps, expand_identifiers
from .chunker import FileIndex

SCHEMA_VERSION = 5
PARSER_VERSION = 8

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);

CREATE TABLE IF NOT EXISTS file_versions (
    path TEXT NOT NULL,
    blob_sha TEXT NOT NULL,
    language TEXT NOT NULL,
    handler TEXT NOT NULL,
    module TEXT,
    status TEXT NOT NULL,
    reason TEXT,
    skeleton TEXT,
    parser_version INTEGER NOT NULL,
    syntax_errors INTEGER NOT NULL DEFAULT 0,
    migration TEXT,
    PRIMARY KEY (path, blob_sha)
);

CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY,
    chunk_id TEXT NOT NULL,
    path TEXT NOT NULL,
    blob_sha TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    kind TEXT NOT NULL,
    symbol TEXT NOT NULL,
    name TEXT NOT NULL,
    parent_id TEXT,
    start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL,
    line_numbers TEXT,
    signature TEXT,
    decorators TEXT NOT NULL,
    tags TEXT NOT NULL,
    template_vars TEXT NOT NULL,
    extra TEXT NOT NULL,
    text TEXT NOT NULL,
    token_count INTEGER NOT NULL,
    part INTEGER NOT NULL DEFAULT 0,
    UNIQUE (path, blob_sha, chunk_id)
);
CREATE INDEX IF NOT EXISTS chunks_file ON chunks(path, blob_sha);
CREATE INDEX IF NOT EXISTS chunks_symbol ON chunks(symbol);
CREATE INDEX IF NOT EXISTS chunks_name ON chunks(name);
CREATE INDEX IF NOT EXISTS chunks_chunk_id ON chunks(chunk_id);
CREATE INDEX IF NOT EXISTS chunks_hash ON chunks(content_hash);

CREATE TABLE IF NOT EXISTS refs (
    path TEXT NOT NULL,
    blob_sha TEXT NOT NULL,
    src_chunk_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    target TEXT NOT NULL,
    line INTEGER NOT NULL,
    extra TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS refs_file ON refs(path, blob_sha);

CREATE TABLE IF NOT EXISTS imports (
    path TEXT NOT NULL,
    blob_sha TEXT NOT NULL,
    alias TEXT NOT NULL,
    target TEXT NOT NULL,
    is_star INTEGER NOT NULL,
    line INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS imports_file ON imports(path, blob_sha);

CREATE TABLE IF NOT EXISTS views (
    view_key TEXT PRIMARY KEY,
    root_key TEXT,
    base_sha TEXT NOT NULL,
    head_sha TEXT,
    label TEXT,
    created_at TEXT NOT NULL,
    resolved INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS view_files (
    view_key TEXT NOT NULL,
    path TEXT NOT NULL,
    blob_sha TEXT NOT NULL,
    version TEXT NOT NULL,
    superseded INTEGER NOT NULL,
    indexed INTEGER NOT NULL,
    reason TEXT,
    PRIMARY KEY (view_key, path, version)
);

CREATE TABLE IF NOT EXISTS edges (
    view_key TEXT NOT NULL,
    src_chunk_id TEXT NOT NULL,
    src_path TEXT NOT NULL,
    dst_chunk_id TEXT,
    dst_name TEXT NOT NULL,
    kind TEXT NOT NULL,
    confidence TEXT NOT NULL,
    line INTEGER NOT NULL,
    extra TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS edges_src ON edges(view_key, src_chunk_id);
CREATE INDEX IF NOT EXISTS edges_dst ON edges(view_key, dst_chunk_id);
CREATE INDEX IF NOT EXISTS edges_name ON edges(view_key, dst_name);
CREATE INDEX IF NOT EXISTS edges_path ON edges(view_key, src_path);

CREATE TABLE IF NOT EXISTS url_routes (
    view_key TEXT NOT NULL,
    url_chunk_id TEXT NOT NULL,
    view_chunk_id TEXT,
    route TEXT NOT NULL,
    name TEXT,
    view_ref TEXT
);
CREATE INDEX IF NOT EXISTS url_routes_view ON url_routes(view_key, view_chunk_id);

CREATE TABLE IF NOT EXISTS chunk_tags (
    view_key TEXT NOT NULL,
    chunk_id TEXT NOT NULL,
    tag TEXT NOT NULL,
    PRIMARY KEY (view_key, chunk_id, tag)
);

CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(body, content = '', contentless_delete = 1, tokenize = 'unicode61 remove_diacritics 2');

CREATE TABLE IF NOT EXISTS summaries (
    content_hash TEXT NOT NULL,
    model TEXT NOT NULL,
    summary TEXT NOT NULL,
    PRIMARY KEY (content_hash, model)
);

CREATE TABLE IF NOT EXISTS vector_tables (
    model TEXT NOT NULL,
    dims INTEGER NOT NULL,
    table_name TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (model, dims)
);

CREATE TABLE IF NOT EXISTS vector_rows (
    table_name TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    facet TEXT NOT NULL,
    PRIMARY KEY (table_name, content_hash, facet)
);

CREATE TABLE IF NOT EXISTS memory (
    id INTEGER PRIMARY KEY,
    source TEXT NOT NULL,
    sha TEXT,
    date TEXT,
    subject TEXT NOT NULL,
    body TEXT,
    path TEXT,
    function_context TEXT,
    hunk TEXT NOT NULL,
    finding TEXT,
    content_hash TEXT NOT NULL UNIQUE
);
CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(body, content = '', contentless_delete = 1, tokenize = 'unicode61 remove_diacritics 2');

CREATE TABLE IF NOT EXISTS batches (
    batch_id TEXT PRIMARY KEY,
    purpose TEXT NOT NULL,
    model TEXT NOT NULL,
    dims INTEGER,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS usage (
    ts TEXT NOT NULL,
    purpose TEXT NOT NULL,
    model TEXT NOT NULL,
    input_tokens INTEGER NOT NULL,
    cached_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    cost_usd REAL NOT NULL
);
"""


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def vector_table_name(model: str, dims: int) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", model.lower()).strip("_")
    return f"vec_{slug}_{dims}"


def pack_vector(vector: list[float]) -> bytes:
    return struct.pack(f"{len(vector)}f", *vector)


def fts_body(symbol: str, name: str, path: str, tags: list[str], text: str, summary: str | None) -> str:
    parts = [symbol.replace(".", " "), name, path.replace("/", " "), " ".join(tags), text]
    if summary:
        parts.append(summary)
    return expand_identifiers("\n".join(parts))


class Store:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.connection = sqlite3.connect(str(path), check_same_thread=False, timeout=60)
        self.connection.row_factory = sqlite3.Row
        self.connection.enable_load_extension(True)
        sqlite_vec.load(self.connection)
        self.connection.enable_load_extension(False)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self.connection.execute("PRAGMA foreign_keys=OFF")
        self.connection.executescript(SCHEMA)
        self.set_meta("schema_version", str(SCHEMA_VERSION))

    def close(self) -> None:
        self.connection.close()

    def execute(self, sql: str, parameters=()) -> sqlite3.Cursor:
        with self.lock:
            return self.connection.execute(sql, parameters)

    def query(self, sql: str, parameters=()) -> list[sqlite3.Row]:
        with self.lock:
            return self.connection.execute(sql, parameters).fetchall()

    def query_one(self, sql: str, parameters=()) -> sqlite3.Row | None:
        with self.lock:
            return self.connection.execute(sql, parameters).fetchone()

    def commit(self) -> None:
        with self.lock:
            self.connection.commit()

    def set_meta(self, key: str, value: str) -> None:
        self.execute("INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)", (key, value))
        self.commit()

    def get_meta(self, key: str) -> str | None:
        row = self.query_one("SELECT value FROM meta WHERE key = ?", (key,))
        return row["value"] if row else None

    def known_file_versions(self, keys: list[tuple[str, str]]) -> dict[tuple[str, str], sqlite3.Row]:
        found: dict[tuple[str, str], sqlite3.Row] = {}
        with self.lock:
            self.connection.execute("CREATE TEMP TABLE IF NOT EXISTS lookup_keys (path TEXT, blob_sha TEXT)")
            self.connection.execute("DELETE FROM lookup_keys")
            self.connection.executemany("INSERT INTO lookup_keys VALUES (?, ?)", keys)
            rows = self.connection.execute(
                "SELECT f.* FROM file_versions f JOIN lookup_keys k ON f.path = k.path AND f.blob_sha = k.blob_sha"
            ).fetchall()
        for row in rows:
            found[(row["path"], row["blob_sha"])] = row
        return found

    def record_file_version(
        self,
        path: str,
        blob_sha: str,
        language: str,
        handler: str,
        module: str | None,
        status: str,
        reason: str | None,
        skeleton: str | None,
        parser_version: int,
    ) -> None:
        self.execute(
            """INSERT OR REPLACE INTO file_versions
               (path, blob_sha, language, handler, module, status, reason, skeleton, parser_version, syntax_errors, migration)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, NULL)""",
            (path, blob_sha, language, handler, module, status, reason, skeleton, parser_version),
        )

    def remove_file_index(self, path: str, blob_sha: str) -> None:
        with self.lock:
            rows = self.connection.execute("SELECT id FROM chunks WHERE path = ? AND blob_sha = ?", (path, blob_sha)).fetchall()
            for row in rows:
                self.connection.execute("DELETE FROM chunk_fts WHERE rowid = ?", (row["id"],))
            self.connection.execute("DELETE FROM chunks WHERE path = ? AND blob_sha = ?", (path, blob_sha))
            self.connection.execute("DELETE FROM refs WHERE path = ? AND blob_sha = ?", (path, blob_sha))
            self.connection.execute("DELETE FROM imports WHERE path = ? AND blob_sha = ?", (path, blob_sha))

    def save_file_index(self, file_index: FileIndex) -> None:
        with self.lock:
            self.remove_file_index(file_index.path, file_index.blob_sha)
            cursor = self.connection.cursor()
            for chunk in file_index.chunks:
                cursor.execute(
                    """INSERT INTO chunks (chunk_id, path, blob_sha, content_hash, kind, symbol, name, parent_id,
                       start_line, end_line, line_numbers, signature, decorators, tags, template_vars, extra, text,
                       token_count, part)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        chunk.chunk_id,
                        chunk.path,
                        chunk.blob_sha,
                        chunk.content_hash,
                        chunk.kind,
                        chunk.symbol,
                        chunk.name,
                        chunk.parent_id,
                        chunk.start_line,
                        chunk.end_line,
                        dumps(chunk.line_numbers) if chunk.line_numbers else None,
                        chunk.signature,
                        dumps(chunk.decorators),
                        dumps(chunk.tags),
                        dumps(chunk.template_vars),
                        dumps(chunk.extra),
                        chunk.text,
                        chunk.token_count,
                        chunk.part,
                    ),
                )
                row_id = cursor.lastrowid
                body = fts_body(chunk.symbol, chunk.name, chunk.path, chunk.tags, chunk.text, None)
                cursor.execute("INSERT INTO chunk_fts(rowid, body) VALUES (?, ?)", (row_id, body))
            cursor.executemany(
                "INSERT INTO refs (path, blob_sha, src_chunk_id, kind, target, line, extra) VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    (file_index.path, file_index.blob_sha, ref.src_chunk_id, ref.kind, ref.target, ref.line, dumps(ref.extra))
                    for ref in file_index.refs
                ],
            )
            cursor.executemany(
                "INSERT INTO imports (path, blob_sha, alias, target, is_star, line) VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (file_index.path, file_index.blob_sha, entry.alias, entry.target, int(entry.is_star), entry.line)
                    for entry in file_index.imports
                ],
            )
            cursor.execute(
                "UPDATE file_versions SET status = 'parsed', syntax_errors = ?, migration = ? WHERE path = ? AND blob_sha = ?",
                (file_index.syntax_errors, dumps(file_index.migration) if file_index.migration else None, file_index.path, file_index.blob_sha),
            )

    def refresh_fts_summary(self, row_id: int, body: str) -> None:
        with self.lock:
            self.connection.execute("DELETE FROM chunk_fts WHERE rowid = ?", (row_id,))
            self.connection.execute("INSERT INTO chunk_fts(rowid, body) VALUES (?, ?)", (row_id, body))

    def ensure_vector_table(self, model: str, dims: int) -> str:
        table = vector_table_name(model, dims)
        with self.lock:
            self.connection.execute(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS {table} USING vec0(embedding float[{dims}] distance_metric=cosine)"
            )
            self.connection.execute(
                "INSERT OR IGNORE INTO vector_tables(model, dims, table_name, created_at) VALUES (?, ?, ?, ?)",
                (model, dims, table, now_iso()),
            )
            self.connection.commit()
        return table

    def vector_table(self, model: str, dims: int) -> str | None:
        row = self.query_one("SELECT table_name FROM vector_tables WHERE model = ? AND dims = ?", (model, dims))
        return row["table_name"] if row else None

    def insert_vectors(self, table: str, items: list[tuple[str, str, list[float]]]) -> None:
        with self.lock:
            for content_hash_value, facet, vector in items:
                existing = self.connection.execute(
                    "SELECT rowid FROM vector_rows WHERE table_name = ? AND content_hash = ? AND facet = ?",
                    (table, content_hash_value, facet),
                ).fetchone()
                if existing is not None:
                    continue
                cursor = self.connection.execute(
                    "INSERT INTO vector_rows(table_name, content_hash, facet) VALUES (?, ?, ?)",
                    (table, content_hash_value, facet),
                )
                self.connection.execute(f"INSERT INTO {table}(rowid, embedding) VALUES (?, ?)", (cursor.lastrowid, pack_vector(vector)))
            self.connection.commit()

    def record_usage(self, purpose: str, model: str, input_tokens: int, cached_tokens: int, output_tokens: int, cost: float) -> None:
        self.execute(
            "INSERT INTO usage(ts, purpose, model, input_tokens, cached_tokens, output_tokens, cost_usd) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (now_iso(), purpose, model, input_tokens, cached_tokens, output_tokens, cost),
        )
        self.commit()
