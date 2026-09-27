from __future__ import annotations

from ..tools.executor import ToolError, normalize_repo_path
from .findings import SEVERITY_ORDER, Finding, ReviewedFinding, Verdict

MERGE_LINE_SLACK = 2


class Locator:
    def __init__(self, session):
        self.session = session
        self.line_counts: dict[str, int | None] = {}

    def line_count(self, path: str) -> int | None:
        if path not in self.line_counts:
            counts = []
            for version in ("current", "base"):
                view_file = self.session.view.file(path, version)
                if view_file is not None:
                    counts.append(len(self.session.repo.read_text(view_file.blob_sha).splitlines()))
            self.line_counts[path] = max(counts) if counts else None
        return self.line_counts[path]

    def check(self, finding: Finding) -> Finding | str:
        try:
            path = normalize_repo_path(finding.path)
        except ToolError as error:
            return str(error)
        count = self.line_count(path)
        if count is None:
            return f"{path} is not a file in this revision"
        if not 1 <= finding.start_line <= count:
            return f"line {finding.start_line} is outside {path} ({count} lines)"
        end_line = min(max(finding.end_line, finding.start_line), count)
        return finding.model_copy(update={"path": path, "end_line": end_line})


def same_problem(first: Finding, second: Finding) -> bool:
    return (
        first.path == second.path
        and first.category.strip().lower() == second.category.strip().lower()
        and first.start_line <= second.end_line + MERGE_LINE_SLACK
        and second.start_line <= first.end_line + MERGE_LINE_SLACK
    )


def merge_candidates(reported: list[tuple[str, Finding]]) -> list[ReviewedFinding]:
    merged: list[ReviewedFinding] = []
    for agent, finding in sorted(reported, key=lambda item: SEVERITY_ORDER[item[1].severity]):
        for existing in merged:
            if same_problem(existing.finding, finding):
                if agent not in existing.reported_by:
                    existing.reported_by.append(agent)
                existing.finding.evidence.extend(item for item in finding.evidence if item not in existing.finding.evidence)
                break
        else:
            merged.append(ReviewedFinding(id="", finding=finding.model_copy(deep=True), reported_by=[agent], status="unverified"))
    merged.sort(key=lambda item: (item.finding.path, item.finding.start_line))
    for number, item in enumerate(merged, start=1):
        item.id = f"f{number}"
    return merged


def apply_verdicts(candidates: list[ReviewedFinding], verdicts: list[Verdict], locator: Locator) -> None:
    by_id = {verdict.id.strip(): verdict for verdict in verdicts}
    for candidate in candidates:
        verdict = by_id.get(candidate.id)
        if verdict is None:
            candidate.reason = "the verifier returned no verdict"
            continue
        candidate.status = verdict.verdict
        candidate.reason = verdict.reason
        if verdict.verdict != "confirmed":
            continue
        candidate.finding = candidate.finding.model_copy(update={"severity": verdict.severity})
        corrected = locator.check(candidate.finding.model_copy(update={"start_line": verdict.start_line, "end_line": verdict.end_line}))
        if isinstance(corrected, Finding):
            candidate.finding = corrected
