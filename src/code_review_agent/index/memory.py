from __future__ import annotations

from ..git.diff import parse_diff
from ..git.repo import GitRepo
from ..util import expand_identifiers, sha1_text
from .files import classify_path, is_test_path
from .store import Store

FIX_COMMIT_PATTERN = r"(^|[^a-z])(fix|fixes|fixed|fixing|bug|bugfix|hotfix|regression|revert|broken|crash|incorrect|wrong)([^a-z]|$)"
MAX_FILES_PER_COMMIT = 12
MAX_CHANGED_LINES_PER_COMMIT = 600
MAX_HUNK_LINES = 80


def insert_memory(store: Store, source: str, sha: str | None, date: str | None, subject: str, body: str, path: str | None, function_context: str | None, hunk: str, finding: str | None) -> bool:
    content_hash = sha1_text(f"{source}\x00{sha}\x00{path}\x00{hunk}\x00{finding}")
    existing = store.query_one("SELECT id FROM memory WHERE content_hash = ?", (content_hash,))
    if existing is not None:
        return False
    cursor = store.execute(
        """INSERT INTO memory(source, sha, date, subject, body, path, function_context, hunk, finding, content_hash)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (source, sha, date, subject, body, path, function_context, hunk, finding, content_hash),
    )
    fts_text = expand_identifiers("\n".join(part for part in (subject, body, path or "", function_context or "", hunk, finding or "") if part))
    store.execute("INSERT INTO memory_fts(rowid, body) VALUES (?, ?)", (cursor.lastrowid, fts_text))
    return True


def build_memory(repo: GitRepo, store: Store, limit: int, rev: str = "HEAD", progress=None) -> int:
    commits = repo.commits_matching(FIX_COMMIT_PATTERN, limit, rev)
    inserted = 0
    for commit in commits:
        if len(commit.parents) != 1:
            continue
        diff = parse_diff(repo.show_with_function_context(commit.sha), commit.parents[0], commit.sha)
        if len(diff.files) > MAX_FILES_PER_COMMIT:
            continue
        changed_lines = sum(file_diff.changed_line_count() for file_diff in diff.files)
        if changed_lines > MAX_CHANGED_LINES_PER_COMMIT:
            continue
        for file_diff in diff.files:
            decision = classify_path(file_diff.path, 0, [], 10**9)
            if decision.excluded_reason or decision.handler not in ("python", "template", "migration"):
                continue
            for hunk in file_diff.hunks:
                hunk_lines = hunk.lines[:MAX_HUNK_LINES]
                hunk_text = f"@@ -{hunk.old_start},{hunk.old_count} +{hunk.new_start},{hunk.new_count} @@\n" + "\n".join(hunk_lines)
                source = "fix_commit_test" if is_test_path(file_diff.path) else "fix_commit"
                if insert_memory(store, source, commit.sha, commit.date, commit.subject, commit.body[:1000], file_diff.path, hunk.header or None, hunk_text, None):
                    inserted += 1
    store.commit()
    if progress is not None:
        progress(f"memory: {inserted} hunks from {len(commits)} fix commits")
    return inserted


def add_confirmed_finding(store: Store, finding_id: str, path: str, claim: str, snippet: str, verdict: str) -> bool:
    inserted = insert_memory(store, "finding", finding_id, None, f"[{verdict}] {claim}", "", path, None, snippet, claim)
    store.commit()
    return inserted
