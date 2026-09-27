from __future__ import annotations

import os
import re
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from ..config import Settings, load_settings
from ..git.repo import GitRepo, TreeEntry
from ..util import sha1_text
from .chunker import ChunkLimits, FileIndex, index_file
from .files import (
    FileDecision,
    boilerplate_exclusions,
    classify_path,
    content_exclusion,
    module_name_for,
    source_roots,
    skeleton_hash,
    template_name_for,
)
from .resolve import write_resolution
from .store import PARSER_VERSION, SCHEMA_VERSION, Store, now_iso
from .view import View


@dataclass
class BuildReport:
    view_key: str
    files_total: int = 0
    indexed: int = 0
    parsed_now: int = 0
    chunks_in_view: int = 0
    excluded: Counter = field(default_factory=Counter)
    resolution: dict = field(default_factory=dict)
    seconds: float = 0.0


@dataclass
class PreparedFile:
    path: str
    blob_sha: str
    decision: FileDecision
    indexed: bool
    reason: str | None
    skeleton: str | None
    module: str | None


def repo_cache_dir(repo: GitRepo, settings: Settings) -> Path:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "-", repo.root.name).strip("-") or "repo"
    return settings.cache_root / f"{name}-{repo.root_commit_id()[:12]}"


def open_store(repo: GitRepo, settings: Settings) -> Store:
    return Store(repo_cache_dir(repo, settings) / f"index-v{SCHEMA_VERSION}.db")


def settings_for_repo(repo: GitRepo, ref: str = "HEAD", use_llm: bool = True) -> Settings:
    config_text = None
    if repo.ref_exists(ref):
        config_text = repo.file_at(repo.rev_parse(ref), "review.toml")
    return load_settings(config_text, use_llm=use_llm)


def parse_job(arguments: tuple) -> FileIndex:
    return index_file(*arguments)


class IndexBuilder:
    def __init__(self, repo: GitRepo, settings: Settings, store: Store | None = None, progress=None):
        self.repo = repo
        self.settings = settings
        self.store = store or open_store(repo, settings)
        self.progress = progress
        self.limits = ChunkLimits(
            max_tokens=settings.index.max_chunk_tokens,
            target_tokens=settings.index.target_chunk_tokens,
            small_template_tokens=settings.index.small_template_tokens,
            money_pattern=settings.money.identifier_pattern,
        )

    def say(self, message: str) -> None:
        if self.progress is not None:
            self.progress(message)

    def view_exists(self, view_key: str) -> bool:
        row = self.store.query_one("SELECT resolved FROM views WHERE view_key = ?", (view_key,))
        return row is not None and bool(row["resolved"])

    def prepare(self, entries: dict[str, TreeEntry], roots: list[str], apply_families: bool) -> list[PreparedFile]:
        prepared: list[PreparedFile] = []
        keys = [(path, entry.blob_sha) for path, entry in entries.items()]
        known = self.store.known_file_versions(keys)
        decisions: dict[str, FileDecision] = {}
        needs_content: list[str] = []
        for path, entry in entries.items():
            decision = classify_path(path, entry.size, self.settings.index.exclude, self.settings.index.max_file_bytes)
            decisions[path] = decision
            if decision.excluded_reason is None and (path, entry.blob_sha) not in known:
                needs_content.append(path)

        content: dict[str, bytes] = {}
        if needs_content:
            self.say(f"reading {len(needs_content)} new file versions")
            blobs = self.repo.read_blobs([entries[path].blob_sha for path in needs_content])
            for path in needs_content:
                content[path] = blobs.get(entries[path].blob_sha, b"")

        with self.store.lock:
            for path in needs_content:
                entry = entries[path]
                decision = decisions[path]
                data = content[path]
                reason = content_exclusion(decision, data)
                status = "excluded" if reason else "pending"
                module = module_name_for(path, roots) if decision.language == "python" else None
                self.store.record_file_version(
                    path, entry.blob_sha, decision.language, decision.handler, module, status, reason, skeleton_hash(data), PARSER_VERSION
                )
            self.store.commit()
        known = self.store.known_file_versions(keys)

        family_reasons: dict[str, str] = {}
        if apply_families:
            candidates = []
            for path, entry in entries.items():
                row = known.get((path, entry.blob_sha))
                if row is not None and row["status"] != "excluded" and row["skeleton"]:
                    candidates.append((path, row["skeleton"]))
            family_reasons = boilerplate_exclusions(candidates, self.settings.index.boilerplate_family_min)

        for path, entry in entries.items():
            decision = decisions[path]
            row = known.get((path, entry.blob_sha))
            reason = decision.excluded_reason
            if reason is None and row is not None and row["status"] == "excluded":
                reason = row["reason"]
            if reason is None and path in family_reasons:
                reason = family_reasons[path]
            module = row["module"] if row is not None else None
            if decision.language == "python" and module is None:
                module = module_name_for(path, roots)
            prepared.append(
                PreparedFile(
                    path=path,
                    blob_sha=entry.blob_sha,
                    decision=decision,
                    indexed=reason is None,
                    reason=reason,
                    skeleton=row["skeleton"] if row is not None else None,
                    module=module,
                )
            )
        return prepared

    def parse_pending(self, prepared: list[PreparedFile]) -> int:
        keys = [(item.path, item.blob_sha) for item in prepared if item.indexed]
        known = self.store.known_file_versions(keys)
        pending = []
        for item in prepared:
            if not item.indexed:
                continue
            row = known.get((item.path, item.blob_sha))
            if row is None:
                continue
            if row["status"] == "parsed" and row["parser_version"] == PARSER_VERSION:
                continue
            pending.append(item)
        if not pending:
            return 0
        self.say(f"parsing {len(pending)} files")
        blobs = self.repo.read_blobs([item.blob_sha for item in pending])
        jobs = []
        for item in pending:
            template_name = template_name_for(item.path) if item.decision.handler == "template" else None
            jobs.append(
                (
                    item.path,
                    item.blob_sha,
                    blobs.get(item.blob_sha, b""),
                    item.decision.handler,
                    item.decision.language,
                    item.module,
                    template_name,
                    self.limits,
                )
            )
        results = self.run_jobs(jobs)
        with self.store.lock:
            for result in results:
                self.store.save_file_index(result)
                self.store.execute(
                    "UPDATE file_versions SET parser_version = ? WHERE path = ? AND blob_sha = ?",
                    (PARSER_VERSION, result.path, result.blob_sha),
                )
            self.store.commit()
        return len(results)

    def run_jobs(self, jobs: list[tuple]) -> list[FileIndex]:
        worker_count = self.settings.index.parse_workers or min(os.cpu_count() or 2, 4)
        if len(jobs) < 40 or worker_count <= 1:
            return [parse_job(job) for job in jobs]
        results = []
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            for result in executor.map(parse_job, jobs, chunksize=8):
                results.append(result)
        return results

    def write_view(self, view_key: str, root_key: str | None, base_sha: str, head_sha: str | None, label: str, rows: list[tuple]) -> None:
        with self.store.lock:
            self.store.connection.execute("DELETE FROM views WHERE view_key = ?", (view_key,))
            self.store.connection.execute("DELETE FROM view_files WHERE view_key = ?", (view_key,))
            self.store.connection.execute(
                "INSERT INTO views(view_key, root_key, base_sha, head_sha, label, created_at, resolved) VALUES (?, ?, ?, ?, ?, ?, 0)",
                (view_key, root_key, base_sha, head_sha, label, now_iso()),
            )
            self.store.connection.executemany(
                "INSERT OR REPLACE INTO view_files(view_key, path, blob_sha, version, superseded, indexed, reason) VALUES (?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
            self.store.connection.commit()

    def build_root(self, sha: str, force: bool = False) -> tuple[View, BuildReport]:
        started = time.monotonic()
        view_key = f"root:{sha}"
        report = BuildReport(view_key=view_key)
        if self.view_exists(view_key) and not force:
            view = View(self.store, view_key)
            report.seconds = time.monotonic() - started
            return view, report
        self.say(f"indexing snapshot {sha[:12]}")
        entries = self.repo.list_tree(sha)
        roots = source_roots(list(entries))
        prepared = self.prepare(entries, roots, apply_families=True)
        report.parsed_now = self.parse_pending(prepared)
        rows = []
        for item in prepared:
            rows.append((view_key, item.path, item.blob_sha, "base", 0, int(item.indexed), item.reason))
            if not item.indexed:
                report.excluded[(item.reason or "").split(":")[0]] += 1
        report.files_total = len(prepared)
        report.indexed = sum(1 for item in prepared if item.indexed)
        self.write_view(view_key, None, sha, None, f"snapshot {sha[:12]}", rows)
        view = View(self.store, view_key)
        self.say("resolving graph")
        report.resolution = write_resolution(self.store, view, self.settings)
        report.chunks_in_view = len(view.chunk_rows("1=1"))
        report.seconds = time.monotonic() - started
        return view, report

    def build_overlay(self, base_sha: str, head_sha: str, force: bool = False) -> tuple[View, BuildReport]:
        if base_sha == head_sha:
            return self.build_root(base_sha, force)
        overrides: dict[str, str | None] = {}
        changes = self.repo.name_status(base_sha, head_sha)
        for change in changes:
            if change.old_path and change.old_path != change.new_path:
                overrides[change.old_path] = None
        head_tree = self.repo.list_tree(head_sha)
        base_tree = self.repo.list_tree_cached(base_sha)
        for change in changes:
            if change.new_path:
                entry = head_tree.get(change.new_path)
                base_entry = base_tree.get(change.new_path)
                same_content = entry is not None and base_entry is not None and entry.blob_sha == base_entry.blob_sha
                if same_content and change.old_path == change.new_path:
                    continue
                overrides[change.new_path] = entry.blob_sha if entry else None
        view_key = f"overlay:{base_sha}..{head_sha}"
        return self.build_view_with_overrides(base_sha, overrides, view_key, head_sha, f"{base_sha[:12]}..{head_sha[:12]}", head_tree, force)

    def build_synthetic(self, root_sha: str, overrides: dict[str, str | None], label: str, force: bool = False) -> tuple[View, BuildReport]:
        digest_source = "\n".join(f"{path}={blob}" for path, blob in sorted(overrides.items()))
        view_key = f"synthetic:{root_sha}:{sha1_text(digest_source)[:16]}"
        return self.build_view_with_overrides(root_sha, overrides, view_key, None, label, None, force)

    def build_view_with_overrides(
        self,
        root_sha: str,
        overrides: dict[str, str | None],
        view_key: str,
        head_sha: str | None,
        label: str,
        head_tree: dict[str, TreeEntry] | None,
        force: bool,
    ) -> tuple[View, BuildReport]:
        started = time.monotonic()
        root_view, root_report = self.build_root(root_sha)
        report = BuildReport(view_key=view_key)
        if self.view_exists(view_key) and not force:
            view = View(self.store, view_key)
            report.seconds = time.monotonic() - started
            return view, report

        base_files = root_view.load_files()
        base_tree = self.repo.list_tree_cached(root_sha)
        all_paths = set(base_tree)
        for path, blob in overrides.items():
            if blob is None:
                all_paths.discard(path)
            else:
                all_paths.add(path)
        roots = source_roots(sorted(all_paths))

        head_entries: dict[str, TreeEntry] = {}
        for path, blob in overrides.items():
            if blob is None:
                continue
            if head_tree is not None and path in head_tree:
                head_entries[path] = head_tree[path]
            else:
                size = len(self.repo.read_blob(blob))
                head_entries[path] = TreeEntry(path=path, blob_sha=blob, mode="100644", size=size)
        prepared = self.prepare(head_entries, roots, apply_families=False)
        report.parsed_now = self.parse_pending(prepared)

        rows = []
        for path in overrides:
            base_file = base_files.get((path, "base"))
            if base_file is not None:
                rows.append((view_key, path, base_file.blob_sha, "base", 1, int(base_file.indexed or path in head_entries), base_file.reason))
        for item in prepared:
            rows.append((view_key, item.path, item.blob_sha, "head", 0, int(item.indexed), item.reason))
            if not item.indexed:
                report.excluded[(item.reason or "").split(":")[0]] += 1

        base_needing_parse = []
        for path in overrides:
            base_file = base_files.get((path, "base"))
            if base_file is not None and not base_file.indexed and path in head_entries:
                entry = base_tree.get(path)
                if entry is not None:
                    base_needing_parse.append(entry)
        if base_needing_parse:
            base_prepared = self.prepare({entry.path: entry for entry in base_needing_parse}, roots, apply_families=False)
            self.parse_pending(base_prepared)

        report.files_total = len(prepared)
        report.indexed = sum(1 for item in prepared if item.indexed)
        self.write_view(view_key, f"root:{root_sha}", root_sha, head_sha, label, rows)
        view = View(self.store, view_key)
        report.resolution = write_resolution(self.store, view, self.settings)
        report.chunks_in_view = len(view.chunk_rows("1=1"))
        report.seconds = time.monotonic() - started + root_report.seconds
        return view, report
