from __future__ import annotations

import json
import re
from pathlib import Path, PurePosixPath

from pydantic import BaseModel, Field

from ..git.diff import FileDiff, parse_diff
from ..git.repo import GitRepo
from ..index.chunker import ChunkLimits, index_file
from ..index.files import classify_path, is_test_path, module_name_for, source_roots, template_name_for
from ..index.memory import FIX_COMMIT_PATTERN

MAX_SOURCE_FILES = 6
MAX_CHANGED_LINES = 150
RELEVANT_KINDS = {"function", "method", "test", "class_skeleton", "model", "template_block", "url"}
SECURITY_WORDS = re.compile(r"(?i)(tenant|permission|auth|csrf|xss|inject|leak|idor|access|scope|privilege|sanitiz|escape)")
STATE_WORDS = re.compile(r"(?i)(cache|transaction|atomic|celery|task|signal|race|lock|retry|commit|concurren|stale|duplicate|idempot)")


class RelevantChunk(BaseModel):
    chunk_id: str
    path: str
    symbol: str
    kind: str


class ExpectedFinding(BaseModel):
    path: str
    start_line: int
    end_line: int
    category: str
    note: str = ""


class GoldenCase(BaseModel):
    id: str
    fix_sha: str
    parent_sha: str
    date: str
    subject: str
    primary_path: str
    source_paths: list[str]
    test_paths: list[str]
    view_overrides: dict[str, str | None]
    diff_overrides: dict[str, str | None]
    relevant: list[RelevantChunk] = Field(default_factory=list)
    extra_relevant_symbols: list[str] = Field(default_factory=list)
    expected_findings: list[ExpectedFinding] = Field(default_factory=list)
    labels_confirmed: bool = False


def golden_dir(cache_dir: Path) -> Path:
    path = cache_dir / "golden" / "cases"
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_case(cache_dir: Path, case: GoldenCase) -> Path:
    path = golden_dir(cache_dir) / f"{case.id}.json"
    path.write_text(case.model_dump_json(indent=2))
    return path


def load_cases(cache_dir: Path, ids: list[str] | None = None) -> list[GoldenCase]:
    cases = []
    for path in sorted(golden_dir(cache_dir).glob("*.json")):
        case = GoldenCase.model_validate(json.loads(path.read_text()))
        if ids and case.id not in ids:
            continue
        cases.append(case)
    return cases


def proposed_category(subject: str, body: str) -> str:
    text = f"{subject}\n{body}"
    if SECURITY_WORDS.search(text):
        return "security"
    if STATE_WORDS.search(text):
        return "state"
    return "bug"


def fix_side_lines(file_diff: FileDiff) -> set[int]:
    lines = set(file_diff.added_line_numbers())
    for hunk in file_diff.hunks:
        if not hunk.added_line_numbers():
            start, end = hunk.new_range()
            lines.update(range(start, end + 1))
    return lines


def buggy_side_ranges(file_diff: FileDiff) -> list[tuple[int, int]]:
    ranges = []
    for hunk in file_diff.hunks:
        removed = hunk.removed_line_numbers()
        if removed:
            ranges.append((min(removed), max(removed)))
        else:
            start, end = hunk.old_range()
            ranges.append((start, end))
    return ranges


def buggy_side_touches(file_diff: FileDiff) -> tuple[set[int], set[int]]:
    removed = set(file_diff.removed_line_numbers())
    insertion_points: set[int] = set()
    for hunk in file_diff.hunks:
        old_line = hunk.old_start - 1
        previous_was_addition = False
        for line in hunk.lines:
            if line.startswith("+"):
                if not previous_was_addition:
                    insertion_points.add(old_line)
                previous_was_addition = True
                continue
            previous_was_addition = False
            if line.startswith("\\"):
                continue
            old_line += 1
    return removed, insertion_points


def relevant_chunks_for(repo: GitRepo, commit_sha: str, file_diff: FileDiff, limits: ChunkLimits, side: str = "fix") -> list[RelevantChunk]:
    path = file_diff.new_path if side == "fix" else file_diff.old_path
    if path is None:
        return []
    tree = repo.list_tree_cached(commit_sha)
    entry = tree.get(path)
    if entry is None:
        return []
    decision = classify_path(path, entry.size, [], 10**9)
    if decision.excluded_reason:
        return []
    module = module_name_for(path, source_roots(list(tree))) if decision.language == "python" else None
    template_name = template_name_for(path) if decision.handler == "template" else None
    file_index = index_file(path, entry.blob_sha, repo.read_blob(entry.blob_sha), decision.handler, decision.language, module, template_name, limits)
    if side == "fix":
        lines = fix_side_lines(file_diff)
        insertion_points: set[int] = set()
    else:
        lines, insertion_points = buggy_side_touches(file_diff)
    relevant = []
    for chunk in file_index.chunks:
        if chunk.kind not in RELEVANT_KINDS or chunk.part > 0:
            continue
        chunk_lines = set(chunk.line_numbers or range(chunk.start_line, chunk.end_line + 1))
        inserted_inside = any(chunk.start_line <= point < chunk.end_line for point in insertion_points)
        if chunk_lines & lines or inserted_inside:
            relevant.append(RelevantChunk(chunk_id=chunk.chunk_id, path=path, symbol=chunk.symbol, kind=chunk.kind))
    return relevant


def mine_cases(repo: GitRepo, cache_dir: Path, limit: int, rev: str = "HEAD", scan: int = 400, progress=None) -> list[GoldenCase]:
    limits = ChunkLimits()
    existing = {case.id for case in load_cases(cache_dir)}
    created: list[GoldenCase] = []
    for commit in repo.commits_matching(FIX_COMMIT_PATTERN, scan, rev):
        if len(created) >= limit:
            break
        if len(commit.parents) != 1 or commit.subject.lower().startswith("revert") or commit.sha[:12] in existing:
            continue
        diff = parse_diff(repo.show_with_function_context(commit.sha), commit.parents[0], commit.sha)
        source_diffs = []
        test_diffs = []
        for file_diff in diff.files:
            decision = classify_path(file_diff.path, 0, [], 10**9)
            if decision.excluded_reason or decision.handler not in ("python", "template"):
                continue
            if is_test_path(file_diff.path):
                test_diffs.append(file_diff)
            elif file_diff.status == "M" and file_diff.hunks:
                source_diffs.append(file_diff)
        if not source_diffs or len(source_diffs) > MAX_SOURCE_FILES:
            continue
        total_changed = sum(file_diff.changed_line_count() for file_diff in source_diffs)
        if total_changed > MAX_CHANGED_LINES or total_changed == 0:
            continue
        primary = max(source_diffs, key=lambda item: (item.changed_line_count(), PurePosixPath(item.path).suffix == ".py"))
        parent_tree = repo.list_tree_cached(commit.parents[0])
        parent_entry = parent_tree.get(primary.path)
        if parent_entry is None:
            continue
        view_overrides: dict[str, str | None] = {primary.path: parent_entry.blob_sha}
        for test_diff in test_diffs:
            if test_diff.new_path and test_diff.new_path != test_diff.old_path:
                view_overrides[test_diff.new_path] = None
            if test_diff.old_path:
                old_entry = parent_tree.get(test_diff.old_path)
                view_overrides[test_diff.old_path] = old_entry.blob_sha if old_entry is not None else None
        relevant: list[RelevantChunk] = []
        for file_diff in source_diffs:
            if file_diff.path != primary.path:
                relevant.extend(relevant_chunks_for(repo, commit.sha, file_diff, limits))
        for test_diff in test_diffs:
            if test_diff.status in ("M", "R") and test_diff.old_path:
                for chunk in relevant_chunks_for(repo, commit.parents[0], test_diff, limits, side="buggy"):
                    if chunk.kind == "test":
                        relevant.append(chunk)
        category = proposed_category(commit.subject, commit.body)
        expected = []
        for start, end in buggy_side_ranges(primary):
            expected.append(ExpectedFinding(path=primary.path, start_line=start, end_line=end, category=category, note=commit.subject[:160]))
        case = GoldenCase(
            id=commit.sha[:12],
            fix_sha=commit.sha,
            parent_sha=commit.parents[0],
            date=commit.date,
            subject=commit.subject[:200],
            primary_path=primary.path,
            source_paths=[item.path for item in source_diffs],
            test_paths=[item.path for item in test_diffs],
            view_overrides=view_overrides,
            diff_overrides={primary.path: parent_entry.blob_sha},
            relevant=relevant,
            expected_findings=expected,
        )
        save_case(cache_dir, case)
        created.append(case)
        if progress is not None:
            progress(f"case {case.id}: {case.subject[:70]} ({len(relevant)} relevant chunks)")
    return created
