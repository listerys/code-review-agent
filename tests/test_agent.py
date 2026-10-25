import json
import re

import pytest
from pydantic import BaseModel
from typer.testing import CliRunner

from code_review_agent.agent.findings import (
    Finding,
    ReviewedFinding,
    ReviewReport,
    SpecialistOutput,
    Verdict,
)
from code_review_agent.agent.loop import AgentError, AgentLoop
from code_review_agent.agent.orchestrator import Reviewer, ReviewFailed, ReviewOptions
from code_review_agent.agent.report import render_markdown
from code_review_agent.agent.verifier import Locator, apply_verdicts, merge_candidates
from code_review_agent.cli import app as cli
from code_review_agent.demo import FEATURE_BRANCH
from code_review_agent.evals.findings import score_case, score_demo
from code_review_agent.evals.golden import ExpectedFinding
from code_review_agent.llm.client import LLMClient
from code_review_agent.retrieval.profiles import PROFILES
from code_review_agent.session import ReviewSession
from code_review_agent.tools.executor import ToolExecutor
from code_review_agent.tools.impl import build_registry
from code_review_agent.tools.registry import ToolRegistry, ToolSpec

SECRET = "sk-proj-abcdefghijklmnopqrstuvwxyz0123456789"

TENANT = {"path": "shop/views.py", "start": 14, "end": 17, "severity": "critical", "category": "security.tenant_isolation", "title": "order_detail no longer scopes orders to request.tenant"}
VALIDATION = {"path": "shop/views.py", "start": 33, "end": 37, "severity": "medium", "category": "bug.input_validation", "title": "refund quantities are not validated"}
ON_COMMIT = {"path": "shop/services/refunds.py", "start": 48, "end": 48, "severity": "high", "category": "state.async_before_commit", "title": "refund email is enqueued inside the transaction"}
MONEY = {"path": "shop/services/refunds.py", "start": 22, "end": 29, "severity": "high", "category": "bug.money_calculation", "title": "calculate_refund ignores already refunded quantities"}
CACHE = {"path": "shop/services/refunds.py", "start": 45, "end": 47, "severity": "medium", "category": "state.cache_invalidation", "title": "invalidate_order_total call was removed so cached totals go stale"}
HALLUCINATED = {"path": "shop/payments.py", "start": 3, "end": 4, "severity": "low", "category": "bug.other", "title": "problem in a file that does not exist"}

SPECIALIST_FINDINGS = {
    "security_integrity": [TENANT],
    "bug_hunter": [MONEY, VALIDATION, {**TENANT, "severity": "high"}],
    "state_regression": [ON_COMMIT, CACHE],
    "impact_mapper": [HALLUCINATED],
    "test_verifier": [],
}


def finding(path="shop/views.py", start=14, end=17, severity="high", category="security.tenant_isolation", title="order_detail skips the tenant filter", evidence=None):
    return Finding(
        path=path,
        start_line=start,
        end_line=end,
        severity=severity,
        category=category,
        title=title,
        explanation="explained",
        evidence=evidence or [f"{path}:{start}"],
        suggested_fix=None,
    )


def rounds(body):
    return sum(1 for item in body["input"] if item.get("type") == "function_call_output")


def agent_of(body):
    if body["instructions"].startswith("You are the verifier"):
        return "verifier"
    return body["instructions"].removeprefix("You are the ").split(" specialist", 1)[0]


def final(findings):
    return {"final": {"findings": [item.model_dump() for item in findings]}}


def verify_candidates(body):
    verdicts = []
    for candidate_id, block in re.findall(r'<candidate id="(f\d+)"[^>]*>\n(.*?)\n</candidate>', body["input"][0]["content"], re.DOTALL):
        data = json.loads(block)
        rejected = "invalidate_order_total" in data["title"]
        verdicts.append(
            {
                "id": candidate_id,
                "verdict": "rejected" if rejected else "confirmed",
                "reason": "post_save receivers still invalidate the cache" if rejected else "confirmed by reading the code",
                "severity": data["severity"],
                "start_line": data["start_line"],
                "end_line": data["end_line"],
            }
        )
    return {"final": {"verdicts": verdicts}}


def demo_policy(body):
    agent = agent_of(body)
    if agent == "verifier":
        if rounds(body) == 0:
            return {"calls": [("signal_receivers", {"model": "Refund"})]}
        return verify_candidates(body)
    if rounds(body) == 0:
        return {"calls": [("get_diff", {"path": None, "offset": None})]}
    return final([finding(**spec) for spec in SPECIALIST_FINDINGS[agent]])


def tool_until_forced(body):
    if body["tool_choice"] == "none":
        return final([])
    return {"calls": [("get_diff", {"path": None, "offset": None})]}


def agent_requests(fake):
    return [request["body"] for request in fake.requests if "tools" in request["body"]]


def make_loop(session, registry=None, **kwargs):
    registry = registry or build_registry()
    executor = ToolExecutor(session, registry, session.settings.tools)
    return AgentLoop(session.llm, executor, registry.openai_definitions(), **kwargs)


def test_loop_runs_tools_and_passes_reasoning_back(llm_session, fake_openai):
    def policy(body):
        if rounds(body) == 0:
            return {"calls": [("find_symbol", {"name": "calculate_refund", "kind": None, "version": "current"})]}
        return final([])

    fake_openai.agent_policy = policy
    result = make_loop(llm_session).run("bug_hunter", "instructions", "review this", SpecialistOutput, max_turns=5)
    assert (result.turns, result.tool_calls, result.stopped) == (2, 1, None)
    first, second = agent_requests(fake_openai)
    assert first["model"] == "gpt-5" and first["store"] is False and first["tool_choice"] == "auto"
    assert first["include"] == ["reasoning.encrypted_content"] and first["reasoning"] == {"effort": "medium"}
    assert first["text"]["format"]["name"] == "SpecialistOutput"
    assert any(item.get("type") == "reasoning" and item["encrypted_content"].startswith("encrypted-") for item in second["input"])
    output = next(item for item in second["input"] if item.get("type") == "function_call_output")
    assert "RefundService.calculate_refund" in output["output"]


def test_loop_scrubs_prompt_and_tool_output(llm_session, fake_openai):
    class NoArgs(BaseModel):
        pass

    registry = ToolRegistry()
    registry.register(ToolSpec("leak", "returns a secret", NoArgs, lambda executor, arguments: f"KEY = '{SECRET}'", text_output=True))
    fake_openai.agent_policy = lambda body: {"calls": [("leak", {})]} if rounds(body) == 0 else final([])
    make_loop(llm_session, registry).run("bug_hunter", "instructions", f"token {SECRET}", SpecialistOutput, max_turns=3)
    sent = json.dumps(agent_requests(fake_openai), ensure_ascii=False)
    assert SECRET not in sent and "«REDACTED" in sent


def test_loop_forces_an_answer_at_the_turn_limit(llm_session, fake_openai):
    fake_openai.agent_policy = tool_until_forced
    result = make_loop(llm_session).run("bug_hunter", "instructions", "prompt", SpecialistOutput, max_turns=3)
    assert (result.turns, result.tool_calls, result.stopped) == (3, 2, "max_turns")
    assert agent_requests(fake_openai)[-1]["tool_choice"] == "none"


def test_loop_stops_calling_tools_when_over_budget(llm_session, fake_openai):
    fake_openai.agent_policy = tool_until_forced
    loop = make_loop(llm_session, over_budget=lambda: llm_session.llm.totals.calls >= 1)
    result = loop.run("bug_hunter", "instructions", "prompt", SpecialistOutput, max_turns=10)
    assert (result.turns, result.tool_calls, result.stopped) == (2, 1, "budget")


def test_unknown_tool_error_goes_back_to_the_model(llm_session, fake_openai):
    fake_openai.agent_policy = lambda body: {"calls": [("delete_repo", {})]} if rounds(body) == 0 else final([])
    make_loop(llm_session).run("bug_hunter", "instructions", "prompt", SpecialistOutput, max_turns=3)
    output = next(item for item in agent_requests(fake_openai)[-1]["input"] if item.get("type") == "function_call_output")
    assert "not available" in output["output"]


def test_answer_that_breaks_the_schema_raises(llm_session, fake_openai):
    fake_openai.agent_policy = lambda body: {"final": {"unexpected": True}}
    with pytest.raises(AgentError):
        make_loop(llm_session).run("bug_hunter", "instructions", "prompt", SpecialistOutput, max_turns=3)


def test_locator_normalizes_and_rejects_bad_locations(demo_session):
    locator = Locator(demo_session)
    checked = locator.check(finding(path="./shop/views.py", start=14, end=400))
    assert (checked.path, checked.end_line) == ("shop/views.py", 42)
    assert "not a file" in locator.check(finding(path="shop/nope.py"))
    assert "outside" in locator.check(finding(start=500, end=501))
    assert "traversal" in locator.check(finding(path="../etc/passwd"))


def test_merge_combines_the_same_problem_and_keeps_distinct_ones():
    merged = merge_candidates(
        [
            ("bug_hunter", finding(severity="medium", evidence=["c1"])),
            ("security_integrity", finding(start=15, end=16, severity="critical", evidence=["c2"])),
            ("state_regression", finding(category="bug.input_validation", start=15, end=16)),
            ("bug_hunter", finding(path="shop/services/refunds.py", start=48, end=48, category="state.async_before_commit")),
        ]
    )
    assert [(item.id, item.finding.path, item.finding.category) for item in merged] == [
        ("f1", "shop/services/refunds.py", "state.async_before_commit"),
        ("f2", "shop/views.py", "security.tenant_isolation"),
        ("f3", "shop/views.py", "bug.input_validation"),
    ]
    tenant = merged[1]
    assert tenant.finding.severity == "critical"
    assert tenant.reported_by == ["security_integrity", "bug_hunter"]
    assert tenant.finding.evidence == ["c2", "c1"]


def test_apply_verdicts_updates_status_severity_and_valid_lines(demo_session):
    candidates = merge_candidates(
        [
            ("a", finding(path="shop/services/refunds.py", start=48, end=48, category="state.async_before_commit")),
            ("a", finding(path="shop/services/refunds.py", start=45, end=47, category="state.cache_invalidation")),
            ("b", finding()),
            ("c", finding(category="bug.input_validation", start=33, end=37)),
        ]
    )
    assert [item.id for item in candidates] == ["f1", "f2", "f3", "f4"]
    verdicts = [
        Verdict(id="f1", verdict="rejected", reason="handled by a receiver", severity="high", start_line=48, end_line=48),
        Verdict(id="f3", verdict="confirmed", reason="checked", severity="critical", start_line=13, end_line=17),
        Verdict(id=" f4 ", verdict="confirmed", reason="checked", severity="medium", start_line=900, end_line=901),
    ]
    apply_verdicts(candidates, verdicts, Locator(demo_session))
    rejected, missing, tenant, validation = candidates
    assert (rejected.status, rejected.reason) == ("rejected", "handled by a receiver")
    assert (missing.status, missing.reason) == ("unverified", "the verifier returned no verdict")
    assert (tenant.status, tenant.finding.severity, tenant.finding.start_line) == ("confirmed", "critical", 13)
    assert (validation.finding.severity, validation.finding.start_line, validation.finding.end_line) == ("medium", 33, 37)


def test_review_end_to_end_on_demo(llm_session, fake_openai):
    fake_openai.agent_policy = demo_policy
    report = Reviewer(llm_session).run()
    assert {item.finding.category for item in report.findings} == {
        "security.tenant_isolation",
        "bug.input_validation",
        "state.async_before_commit",
        "bug.money_calculation",
    }
    assert all(item.status == "confirmed" for item in report.findings)
    top = report.findings[0]
    assert (top.finding.category, top.finding.severity) == ("security.tenant_isolation", "critical")
    assert sorted(top.reported_by) == ["bug_hunter", "security_integrity"]
    assert {(item.status, item.finding.path) for item in report.dropped} == {("rejected", "shop/services/refunds.py"), ("invalid", "shop/payments.py")}
    assert [run.name for run in report.agents] == [*PROFILES, "verifier"]
    assert all(run.error is None for run in report.agents)
    assert report.agents[-1].findings == 4
    assert report.cost_usd > 0 and report.llm_calls >= 12
    score = score_demo(report)
    assert score.passed and not score.extra


def test_one_failing_specialist_does_not_stop_the_review(llm_session, fake_openai):
    fake_openai.agent_policy = lambda body: {"final": {"unexpected": True}} if agent_of(body) == "bug_hunter" else demo_policy(body)
    report = Reviewer(llm_session).run()
    assert [run.name for run in report.agents if run.error] == ["bug_hunter"]
    assert "security.tenant_isolation" in {item.finding.category for item in report.findings}


def test_review_fails_when_every_specialist_fails(llm_session, fake_openai):
    fake_openai.agent_policy = lambda body: {"final": {"unexpected": True}}
    with pytest.raises(ReviewFailed):
        Reviewer(llm_session).run(ReviewOptions(profiles=["bug_hunter", "state_regression"]))


def test_verifier_failure_leaves_findings_unverified(llm_session, fake_openai):
    fake_openai.agent_policy = lambda body: {"final": {"unexpected": True}} if agent_of(body) == "verifier" else demo_policy(body)
    report = Reviewer(llm_session).run(ReviewOptions(profiles=["security_integrity"]))
    assert [(item.status, item.reason) for item in report.findings] == [("unverified", "verification failed")]
    assert any("verification failed" in warning for warning in report.warnings)


def test_no_verify_keeps_candidates_unverified(llm_session, fake_openai):
    fake_openai.agent_policy = demo_policy
    report = Reviewer(llm_session).run(ReviewOptions(profiles=["state_regression"], verify=False))
    assert sorted(item.finding.category for item in report.findings) == ["state.async_before_commit", "state.cache_invalidation"]
    assert {item.status for item in report.findings} == {"unverified"}
    assert [run.name for run in report.agents] == ["state_regression"]


def test_cost_cap_forces_specialists_to_answer_without_tools(llm_session, fake_openai):
    llm_session.settings.agent.max_review_cost_usd = 0.0
    fake_openai.agent_policy = tool_until_forced
    report = Reviewer(llm_session).run(ReviewOptions(profiles=["bug_hunter"]))
    run = report.agents[0]
    assert (run.stopped, run.tool_calls, run.turns) == ("budget", 0, 1)


def test_empty_diff_makes_no_model_calls(demo_repo, fake_openai, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-real-key-0000000000")
    session = ReviewSession.open(str(demo_repo), base=FEATURE_BRANCH, use_llm=True)
    session._llm = LLMClient(session.settings, session.store, http_client=fake_openai.http_client())
    session._llm_checked = True
    report = Reviewer(session).run()
    assert report.warnings == ["no reviewable changes between base and head"]
    assert fake_openai.requests == []


def test_markdown_report_lists_findings_and_escapes_tables():
    report = ReviewReport(
        repo="repo",
        base_sha="a" * 40,
        head_sha="b" * 40,
        model="gpt-5",
        findings=[ReviewedFinding(id="f1", finding=finding(), reported_by=["bug_hunter"], status="confirmed", reason="checked the view")],
        dropped=[ReviewedFinding(id="f2", finding=finding(title="a | b"), reported_by=["x"], status="rejected", reason="line\nbreak")],
    )
    text = render_markdown(report)
    assert text.startswith(f"# Code review: {'b' * 12} vs {'a' * 12}")
    assert "## 1. [HIGH] order_detail skips the tenant filter" in text
    assert "**Verifier:** checked the view" in text
    assert "a \\| b" in text and "line break" in text


def test_cli_run_requires_a_key(demo_repo):
    result = CliRunner().invoke(cli.app, ["run", "-r", str(demo_repo), "--base", "main"])
    assert result.exit_code == 2
    assert "OPENAI_API_KEY" in result.output


def test_cli_run_rejects_unknown_profile(demo_repo):
    result = CliRunner().invoke(cli.app, ["run", "-r", str(demo_repo), "--base", "main", "-p", "nitpicker"])
    assert result.exit_code == 2
    assert "unknown profile nitpicker" in result.output


def test_cli_run_writes_json_and_markdown(llm_session, fake_openai, monkeypatch, tmp_path):
    fake_openai.agent_policy = demo_policy
    monkeypatch.setattr(cli, "open_session", lambda *args, **kwargs: llm_session)
    json_path, markdown_path = tmp_path / "review.json", tmp_path / "review.md"
    arguments = ["run", "-r", str(llm_session.repo.root), "--base", "main", "--json", str(json_path), "--markdown", str(markdown_path)]
    result = CliRunner().invoke(cli.app, arguments)
    assert result.exit_code == 0, result.output
    assert len(json.loads(json_path.read_text())["findings"]) == 4
    assert markdown_path.read_text().startswith("# Code review:")
    assert "4 findings" in result.output and "LLM usage" in result.output


def test_eval_demo_requires_a_key():
    result = CliRunner().invoke(cli.app, ["eval", "demo"])
    assert result.exit_code == 2
    assert "OPENAI_API_KEY" in result.output


def test_score_case_matches_file_family_and_lines():
    expected = [ExpectedFinding(path="app/pricing.py", start_line=10, end_line=12, category="bug")]
    findings = [
        finding(path="app/pricing.py", start=11, end=11, category="bug.money_calculation"),
        finding(path="app/pricing.py", start=40, end=41, category="bug.other"),
        finding(path="app/pricing.py", start=10, end=10, category="security.tenant_isolation"),
    ]
    assert score_case(findings, expected) == (1, 1)


def test_score_demo_flags_the_planted_false_alarm():
    report = ReviewReport(
        repo="repo",
        base_sha="a" * 40,
        head_sha="b" * 40,
        model="gpt-5",
        findings=[ReviewedFinding(id="f1", finding=finding(**CACHE), reported_by=["state_regression"], status="confirmed")],
    )
    score = score_demo(report)
    assert len(score.false_alarms) == 1 and len(score.missed) == 4 and not score.passed
