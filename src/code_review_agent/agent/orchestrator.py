from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import openai

from ..llm.client import LLMUnavailable
from ..retrieval.profiles import PROFILES
from ..tools.executor import ToolExecutor
from ..tools.impl import build_registry
from .findings import (
    SEVERITY_ORDER,
    AgentRun,
    Finding,
    ReviewedFinding,
    ReviewReport,
    SpecialistOutput,
    VerifierOutput,
)
from .loop import AgentError, AgentLoop
from .prompts import (
    specialist_instructions,
    specialist_prompt,
    verifier_instructions,
    verifier_prompt,
)
from .verifier import Locator, apply_verdicts, merge_candidates

AGENT_ERRORS = (AgentError, openai.APIError)


class ReviewFailed(RuntimeError):
    pass


@dataclass
class ReviewOptions:
    profiles: list[str] = field(default_factory=lambda: list(PROFILES))
    verify: bool = True


class Reviewer:
    def __init__(self, session, progress=None):
        if session.llm is None:
            raise LLMUnavailable("OPENAI_API_KEY is not set")
        self.session = session
        self.llm = session.llm
        self.settings = session.settings
        self.progress = progress or (lambda message: None)
        self.registry = build_registry()
        self.tools = self.registry.openai_definitions()
        self.start_cost = 0.0

    def over_budget(self) -> bool:
        return self.llm.totals.cost_usd - self.start_cost >= self.settings.agent.max_review_cost_usd

    def loop(self) -> AgentLoop:
        executor = ToolExecutor(self.session, self.registry, self.settings.tools)
        return AgentLoop(self.llm, executor, self.tools, self.over_budget, self.progress)

    def run(self, options: ReviewOptions | None = None) -> ReviewReport:
        options = options or ReviewOptions()
        totals = self.llm.totals
        start_calls, start_input, start_output, self.start_cost = totals.calls, totals.input_tokens, totals.output_tokens, totals.cost_usd
        session = self.session
        report = ReviewReport(repo=str(session.repo.root), base_sha=session.base_sha, head_sha=session.head_sha, model=self.settings.models.review_model)
        if not any(file_diff.hunks for file_diff in session.diff.files):
            report.warnings.append("no reviewable changes between base and head")
            return report

        reported, runs = self.run_specialists(options.profiles)
        report.agents.extend(runs)
        if all(run.error for run in runs):
            raise ReviewFailed("every specialist failed: " + "; ".join(f"{run.name}: {run.error}" for run in runs))

        locator = Locator(session)
        valid: list[tuple[str, Finding]] = []
        for agent, finding in reported:
            checked = locator.check(finding)
            if isinstance(checked, str):
                report.dropped.append(ReviewedFinding(id=f"x{len(report.dropped) + 1}", finding=finding, reported_by=[agent], status="invalid", reason=checked))
            else:
                valid.append((agent, checked))
        candidates = merge_candidates(valid)
        if candidates and options.verify:
            report.agents.append(self.verify(candidates, locator, report))
        elif candidates:
            report.warnings.append("verification was skipped; findings are unverified")
        for candidate in candidates:
            (report.dropped if candidate.status == "rejected" else report.findings).append(candidate)
        report.findings.sort(key=lambda item: (SEVERITY_ORDER[item.finding.severity], item.finding.path, item.finding.start_line))

        report.llm_calls = totals.calls - start_calls
        report.input_tokens = totals.input_tokens - start_input
        report.output_tokens = totals.output_tokens - start_output
        report.cost_usd = totals.cost_usd - self.start_cost
        return report

    def run_specialists(self, profiles: list[str]) -> tuple[list[tuple[str, Finding]], list[AgentRun]]:
        diff = self.session.diff
        prompts = {}
        for name in profiles:
            self.progress(f"{name}: retrieving context")
            wave = self.session.engine.first_wave(diff, name)
            prompts[name] = specialist_prompt(diff, wave.pack.text, name, self.settings.agent.diff_prompt_tokens)
        workers = max(1, min(len(profiles), self.settings.models.llm_concurrency))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(lambda name: self.run_specialist(name, prompts[name]), profiles))
        reported = [(run.name, finding) for run, findings in results for finding in findings]
        return reported, [run for run, _ in results]

    def run_specialist(self, name: str, prompt: str) -> tuple[AgentRun, list[Finding]]:
        run = AgentRun(name=name)
        instructions = specialist_instructions(name, PROFILES[name].focus)
        try:
            result = self.loop().run(name, instructions, prompt, SpecialistOutput, self.settings.agent.specialist_max_turns)
        except AGENT_ERRORS as error:
            run.error = f"{type(error).__name__}: {error}"
            self.progress(f"{name}: failed ({run.error})")
            return run, []
        run.turns, run.tool_calls, run.stopped = result.turns, result.tool_calls, result.stopped
        run.findings = len(result.output.findings)
        self.progress(f"{name}: {run.findings} candidate findings after {run.turns} turns")
        return run, result.output.findings

    def verify(self, candidates: list[ReviewedFinding], locator: Locator, report: ReviewReport) -> AgentRun:
        run = AgentRun(name="verifier")
        size = self.settings.agent.verifier_batch_size
        batches = [candidates[index : index + size] for index in range(0, len(candidates), size)]

        def verify_batch(batch: list[ReviewedFinding]):
            listing = [(item.id, item.reported_by, item.finding.model_dump_json(indent=1)) for item in batch]
            prompt = verifier_prompt(self.session.diff, listing, self.settings.agent.diff_prompt_tokens)
            return self.loop().run("verifier", verifier_instructions(), prompt, VerifierOutput, self.settings.agent.verifier_max_turns)

        self.progress(f"verifier: checking {len(candidates)} candidates")
        with ThreadPoolExecutor(max_workers=max(1, min(len(batches), self.settings.models.llm_concurrency))) as pool:
            futures = [pool.submit(verify_batch, batch) for batch in batches]
        for batch, future in zip(batches, futures):
            try:
                result = future.result()
            except AGENT_ERRORS as error:
                run.error = f"{type(error).__name__}: {error}"
                report.warnings.append(f"verification failed for {', '.join(item.id for item in batch)} ({run.error}); those findings are unverified")
                for item in batch:
                    item.reason = "verification failed"
                continue
            apply_verdicts(batch, result.output.verdicts, locator)
            run.turns += result.turns
            run.tool_calls += result.tool_calls
            run.stopped = run.stopped or result.stopped
        run.findings = sum(1 for item in candidates if item.status == "confirmed")
        return run
