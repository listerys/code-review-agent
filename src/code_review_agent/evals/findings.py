from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import openai
from pydantic import BaseModel, Field

from ..agent.findings import Finding, ReviewReport
from ..agent.orchestrator import Reviewer, ReviewFailed
from ..config import Settings
from ..demo import INJECTED_BUGS, NOT_A_BUG, create_demo_repo
from ..git.repo import GitRepo
from ..index.enrich import Enricher
from ..index.store import Store
from ..session import ReviewSession
from .golden import ExpectedFinding, GoldenCase
from .retrieval import case_session

LINE_SLACK = 3


def matches(finding: Finding, path: str, category: str, lines: tuple[int, int] | None = None) -> bool:
    if finding.path != path or finding.family() != category.split(".", 1)[0].lower():
        return False
    if lines is None:
        return True
    return finding.start_line <= lines[1] + LINE_SLACK and lines[0] <= finding.end_line + LINE_SLACK


def describe(finding: Finding) -> str:
    return f"{finding.path}:{finding.start_line} {finding.category}: {finding.title}"


class DemoScore(BaseModel):
    caught: list[str] = Field(default_factory=list)
    missed: list[str] = Field(default_factory=list)
    false_alarms: list[str] = Field(default_factory=list)
    extra: list[str] = Field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.missed and not self.false_alarms


def score_demo(report: ReviewReport) -> DemoScore:
    findings = [item.finding for item in report.findings]
    score = DemoScore()
    accounted: set[int] = set()
    for index, finding in enumerate(findings):
        text = f"{finding.title}\n{finding.explanation}".lower()
        for trap in NOT_A_BUG:
            if finding.path == trap["path"] and any(term in text for term in trap["terms"]):
                score.false_alarms.append(describe(finding))
                accounted.add(index)
                break
    for bug in INJECTED_BUGS:
        hits = [index for index, finding in enumerate(findings) if index not in accounted and matches(finding, bug["path"], bug["category"])]
        label = f"{bug['path']} {bug['category']}"
        (score.caught if hits else score.missed).append(label)
        accounted.update(hits)
    score.extra = [describe(finding) for index, finding in enumerate(findings) if index not in accounted]
    return score


def run_demo_eval(workdir: Path, enrich: bool = False, progress=None) -> tuple[ReviewReport, DemoScore]:
    repo = create_demo_repo(workdir / "tinyshop")
    session = ReviewSession.open(str(repo), base="main", use_llm=True, progress=progress)
    if enrich:
        enricher = Enricher(session.store, session.llm, session.settings, progress)
        enricher.summarize(session.view)
        enricher.embed(session.view)
    report = Reviewer(session, progress).run()
    return report, score_demo(report)


class CaseFindings(BaseModel):
    case_id: str
    subject: str
    expected: int
    matched_expected: int
    findings: int
    matched_findings: int
    cost_usd: float
    error: str | None = None


class FindingsEvalReport(BaseModel):
    created_at: str
    repo: str
    results: list[CaseFindings] = Field(default_factory=list)

    @property
    def recall(self) -> float:
        expected = sum(result.expected for result in self.results)
        return sum(result.matched_expected for result in self.results) / expected if expected else 0.0

    @property
    def precision(self) -> float:
        found = sum(result.findings for result in self.results)
        return sum(result.matched_findings for result in self.results) / found if found else 0.0


def score_case(findings: list[Finding], expected: list[ExpectedFinding]) -> tuple[int, int]:
    matched_expected = sum(
        1 for target in expected if any(matches(finding, target.path, target.category, (target.start_line, target.end_line)) for finding in findings)
    )
    matched_findings = sum(
        1 for finding in findings if any(matches(finding, target.path, target.category, (target.start_line, target.end_line)) for target in expected)
    )
    return matched_expected, matched_findings


def run_findings_eval(repo: GitRepo, settings: Settings, store: Store, cases: list[GoldenCase], progress=None) -> FindingsEvalReport:
    report = FindingsEvalReport(created_at=datetime.now(UTC).isoformat(timespec="seconds"), repo=str(repo.root))
    for case in cases:
        if not case.expected_findings:
            continue
        session = case_session(repo, settings, store, case, progress)
        try:
            review = Reviewer(session, progress).run()
        except (ReviewFailed, openai.APIError) as error:
            report.results.append(
                CaseFindings(
                    case_id=case.id,
                    subject=case.subject,
                    expected=len(case.expected_findings),
                    matched_expected=0,
                    findings=0,
                    matched_findings=0,
                    cost_usd=0.0,
                    error=str(error),
                )
            )
            continue
        findings = [item.finding for item in review.findings]
        matched_expected, matched_findings = score_case(findings, case.expected_findings)
        report.results.append(
            CaseFindings(
                case_id=case.id,
                subject=case.subject,
                expected=len(case.expected_findings),
                matched_expected=matched_expected,
                findings=len(findings),
                matched_findings=matched_findings,
                cost_usd=review.cost_usd,
            )
        )
    return report


def save_findings_report(cache_dir: Path, report: FindingsEvalReport) -> Path:
    directory = cache_dir / "evals"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"findings-{report.created_at.replace(':', '').replace('-', '')}.json"
    data = report.model_dump()
    data.update(recall=report.recall, precision=report.precision)
    path.write_text(json.dumps(data, indent=2))
    return path
