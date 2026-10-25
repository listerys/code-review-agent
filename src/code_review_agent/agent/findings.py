from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Severity = Literal["critical", "high", "medium", "low"]
SEVERITY_ORDER: dict[str, int] = {"critical": 0, "high": 1, "medium": 2, "low": 3}
Status = Literal["confirmed", "unverified", "rejected", "invalid"]


class Finding(BaseModel):
    path: str = Field(description="Repository-relative path of the file with the problem")
    start_line: int = Field(description="First line of the problem in the branch head version (merge-base version for deleted code)")
    end_line: int = Field(description="Last line of the problem, inclusive")
    severity: Severity
    category: str = Field(description="family.specific where family is bug, security, state, performance, or test_gap")
    title: str = Field(description="One-line statement of the problem")
    explanation: str = Field(description="What goes wrong, when, and why this change causes it")
    evidence: list[str] = Field(description="path:line locations or chunk refs (c123) you read that prove the problem")
    suggested_fix: str | None = Field(description="Short concrete fix, or null")

    def family(self) -> str:
        return self.category.split(".", 1)[0].strip().lower()


class SpecialistOutput(BaseModel):
    findings: list[Finding]


class Verdict(BaseModel):
    id: str = Field(description="Candidate id, e.g. f3")
    verdict: Literal["confirmed", "rejected"]
    reason: str = Field(description="One sentence citing what you checked")
    severity: Severity
    start_line: int
    end_line: int


class VerifierOutput(BaseModel):
    verdicts: list[Verdict]


class ReviewedFinding(BaseModel):
    id: str
    finding: Finding
    reported_by: list[str]
    status: Status
    reason: str | None = None


class AgentRun(BaseModel):
    name: str
    turns: int = 0
    tool_calls: int = 0
    findings: int = 0
    stopped: str | None = None
    error: str | None = None


class ReviewReport(BaseModel):
    repo: str
    base_sha: str
    head_sha: str | None
    model: str
    findings: list[ReviewedFinding] = Field(default_factory=list)
    dropped: list[ReviewedFinding] = Field(default_factory=list)
    agents: list[AgentRun] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
