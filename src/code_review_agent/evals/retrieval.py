from __future__ import annotations

import json
import time
from datetime import datetime, UTC
from pathlib import Path

from pydantic import BaseModel, Field

from ..config import Settings
from ..git.repo import GitRepo
from ..index.enrich import Enricher
from ..index.store import Store
from ..retrieval.engine import RetrievalOptions
from ..retrieval.graph import resolve_symbol_rows
from ..session import ReviewSession
from .golden import GoldenCase

ABLATIONS: dict[str, RetrievalOptions] = {
    "graph": RetrievalOptions(use_lexical=False, use_dense=False, use_rerank=False),
    "lexical": RetrievalOptions(use_graph=False, use_dense=False, use_rerank=False),
    "graph+lexical": RetrievalOptions(use_dense=False, use_rerank=False),
    "dense": RetrievalOptions(use_graph=False, use_lexical=False, use_rerank=False),
    "dense-code-only": RetrievalOptions(use_graph=False, use_lexical=False, use_rerank=False, facets=["code"]),
    "hybrid": RetrievalOptions(use_rerank=False),
    "hybrid-code-only": RetrievalOptions(use_rerank=False, facets=["code"]),
    "hybrid+rerank": RetrievalOptions(),
}
NEEDS_DENSE = {"dense", "dense-code-only", "hybrid", "hybrid-code-only", "hybrid+rerank"}
NEEDS_LLM = {"hybrid+rerank"}
EVAL_KEEP = 40


class CaseResult(BaseModel):
    case_id: str
    profile: str
    ablation: str
    relevant: int
    ranks: list[int | None]
    recall: dict[str, float]
    reciprocal_rank: float
    pack_recall: float
    pack_tokens: int
    retrieved: int
    latency_ms: int


class AblationSummary(BaseModel):
    ablation: str
    profile: str
    cases: int
    recall: dict[str, float]
    mrr: float
    pack_recall: float
    mean_latency_ms: float


class RetrievalReport(BaseModel):
    created_at: str
    repo: str
    ks: list[int]
    skipped: list[str] = Field(default_factory=list)
    summaries: list[AblationSummary] = Field(default_factory=list)
    results: list[CaseResult] = Field(default_factory=list)


def ablation_options(name: str, dims: int | None = None, graph_scale: float = 1.0) -> RetrievalOptions:
    base = ABLATIONS[name]
    return RetrievalOptions(
        use_lexical=base.use_lexical,
        use_dense=base.use_dense,
        use_graph=base.use_graph,
        use_rerank=base.use_rerank,
        facets=list(base.facets),
        dims=dims,
        graph_scale=graph_scale,
    )


def resolve_ablations(
    requested: list[str],
    dense_ready: bool,
    llm_ready: bool,
    dims_variants: list[int],
    graph_scales: list[float] | None = None,
) -> tuple[list[tuple[str, RetrievalOptions]], list[str]]:
    chosen: list[tuple[str, RetrievalOptions]] = []
    skipped: list[str] = []
    for name in requested:
        if name not in ABLATIONS:
            skipped.append(f"{name}: unknown ablation")
            continue
        if name in NEEDS_DENSE and not dense_ready:
            skipped.append(f"{name}: needs embeddings (set OPENAI_API_KEY and run review index)")
            continue
        if name in NEEDS_LLM and not llm_ready:
            skipped.append(f"{name}: needs OPENAI_API_KEY")
            continue
        chosen.append((name, ablation_options(name)))
        if ABLATIONS[name].use_graph and (ABLATIONS[name].use_lexical or ABLATIONS[name].use_dense):
            for scale in graph_scales or []:
                chosen.append((f"{name}[graph x{scale:g}]", ablation_options(name, graph_scale=scale)))
        if name == "hybrid":
            for dims in dims_variants:
                chosen.append((f"hybrid@{dims}", ablation_options("hybrid", dims)))
    return chosen, skipped


def relevant_ids(session: ReviewSession, case: GoldenCase) -> list[str]:
    ids = [item.chunk_id for item in case.relevant]
    for symbol in case.extra_relevant_symbols:
        for row in resolve_symbol_rows(session.view, symbol):
            ids.append(row["chunk_id"])
    return list(dict.fromkeys(ids))


def score_case(ranked_chunk_ids: list[str], relevant: list[str], ks: list[int]) -> tuple[list[int | None], dict[str, float], float]:
    positions: dict[str, int] = {}
    for position, chunk_id in enumerate(ranked_chunk_ids, start=1):
        positions.setdefault(chunk_id, position)
    ranks = [positions.get(chunk_id) for chunk_id in relevant]
    recall = {}
    for k in ks:
        hits = sum(1 for rank in ranks if rank is not None and rank <= k)
        recall[f"@{k}"] = hits / len(relevant) if relevant else 0.0
    found = [rank for rank in ranks if rank is not None]
    reciprocal = 1.0 / min(found) if found else 0.0
    return ranks, recall, reciprocal


def case_session(repo: GitRepo, settings: Settings, store: Store, case: GoldenCase, progress=None) -> ReviewSession:
    return ReviewSession.synthetic(
        repo,
        settings,
        store,
        root_sha=case.fix_sha,
        overrides=case.view_overrides,
        label=f"golden:{case.id}",
        diff_overrides=case.diff_overrides,
        progress=progress,
    )


def run_retrieval_eval(
    repo: GitRepo,
    settings: Settings,
    store: Store,
    cases: list[GoldenCase],
    profiles: list[str],
    ablations: list[str],
    ks: list[int],
    enrich: bool,
    dims_variants: list[int] | None = None,
    progress=None,
    graph_scales: list[float] | None = None,
) -> RetrievalReport:
    report = RetrievalReport(created_at=datetime.now(UTC).isoformat(timespec="seconds"), repo=str(repo.root), ks=ks)
    results: list[CaseResult] = []
    usable = [case for case in cases if case.relevant or case.extra_relevant_symbols]
    for case in cases:
        if case not in usable:
            report.skipped.append(f"case {case.id}: no relevant chunks outside the injected file")
    chosen: list[tuple[str, RetrievalOptions]] | None = None
    for case in usable:
        session = case_session(repo, settings, store, case, progress)
        if enrich and session.llm is not None:
            enricher = Enricher(store, session.llm, settings, progress)
            enricher.summarize(session.view)
            enricher.embed(session.view)
            for dims in dims_variants or []:
                dims_settings = settings.model_copy(deep=True)
                dims_settings.models.embedding_dims = dims
                Enricher(store, session.llm, dims_settings, progress).embed(session.view)
        if chosen is None:
            chosen, skipped = resolve_ablations(ablations, session.engine.dense_available(), session.llm is not None, dims_variants or [], graph_scales)
            report.skipped.extend(skipped)
        relevant = relevant_ids(session, case)
        for profile in profiles:
            for name, options in chosen:
                started = time.monotonic()
                wave = session.engine.first_wave(session.diff, profile, options=options, pack=True, keep=EVAL_KEEP)
                latency = int((time.monotonic() - started) * 1000)
                ranked_ids = [item.chunk_id for item in wave.ranked]
                ranks, recall, reciprocal = score_case(ranked_ids, relevant, ks)
                packed_ids = {item.chunk_id for item in wave.pack.included}
                pack_recall = sum(1 for chunk_id in relevant if chunk_id in packed_ids) / len(relevant)
                results.append(
                    CaseResult(
                        case_id=case.id,
                        profile=profile,
                        ablation=name,
                        relevant=len(relevant),
                        ranks=ranks,
                        recall=recall,
                        reciprocal_rank=reciprocal,
                        pack_recall=pack_recall,
                        pack_tokens=wave.pack.tokens,
                        retrieved=len(ranked_ids),
                        latency_ms=latency,
                    )
                )
        if progress is not None:
            progress(f"evaluated case {case.id} ({len(relevant)} relevant chunks)")
    report.results = results
    report.summaries = summarize(results, ks)
    return report


def summarize(results: list[CaseResult], ks: list[int]) -> list[AblationSummary]:
    grouped: dict[tuple[str, str], list[CaseResult]] = {}
    for result in results:
        grouped.setdefault((result.profile, result.ablation), []).append(result)
    summaries = []
    for (profile, ablation), items in grouped.items():
        recall = {}
        for k in ks:
            recall[f"@{k}"] = sum(item.recall[f"@{k}"] for item in items) / len(items)
        summaries.append(
            AblationSummary(
                ablation=ablation,
                profile=profile,
                cases=len(items),
                recall=recall,
                mrr=sum(item.reciprocal_rank for item in items) / len(items),
                pack_recall=sum(item.pack_recall for item in items) / len(items),
                mean_latency_ms=sum(item.latency_ms for item in items) / len(items),
            )
        )
    return summaries


def save_report(cache_dir: Path, report: RetrievalReport) -> Path:
    directory = cache_dir / "evals"
    directory.mkdir(parents=True, exist_ok=True)
    stamp = report.created_at.replace(":", "").replace("-", "")
    path = directory / f"retrieval-{stamp}.json"
    path.write_text(json.dumps(report.model_dump(), indent=2))
    return path
