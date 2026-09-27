from __future__ import annotations

from dataclasses import dataclass, field

from ..config import Settings
from ..git.diff import Diff
from ..index.headers import render_header
from ..index.view import View
from ..llm.client import LLMClient
from ..util import IDENTIFIER, truncate_to_tokens
from .dense import QueryEmbedder, dense_memory_search, dense_search
from .fusion import apply_boosts, reciprocal_rank_fusion
from .graph import GraphIndex, related_to_hits, symbol_hits
from .lexical import lexical_search, memory_search
from .models import ContextPack, RankedList, ReviewQuery, ScoredChunk, SearchFilters
from .pack import Packer, load_summaries
from .profiles import RetrievalProfile, profile_for
from .queries import ChangedChunk, build_first_wave, changed_chunks, hunk_gists
from .rerank import rerank

MODES = ("hybrid", "lexical", "dense", "graph")


@dataclass
class RetrievalOptions:
    use_lexical: bool = True
    use_dense: bool = True
    use_graph: bool = True
    use_rerank: bool = True
    facets: list[str] = field(default_factory=lambda: ["code", "summary"])
    dims: int | None = None
    graph_scale: float = 1.0


@dataclass
class FirstWave:
    profile: str
    queries: list[ReviewQuery]
    changed: list[ChangedChunk]
    ranked: list[ScoredChunk]
    pack: ContextPack | None
    lists: list[RankedList] = field(default_factory=list)
    facts: list[ScoredChunk] = field(default_factory=list)


CONTAINER_KINDS = {"module_header", "config", "class_skeleton"}


def pinned_rows(changed: list[ChangedChunk]) -> list[tuple[int, str]]:
    pinned: list[tuple[int, str]] = []
    order = {"modified": 0, "added": 0, "deleted": 1, "before": 2}
    for item in sorted(changed, key=lambda entry: (order.get(entry.change, 3), entry.row["kind"] in CONTAINER_KINDS)):
        if item.change == "before" and item.row["kind"] in CONTAINER_KINDS:
            continue
        role = {"modified": "changed", "added": "added", "deleted": "deleted", "before": "before"}[item.change]
        pinned.append((item.row["id"], role))
    return pinned


def focus_lines_for(changed: list[ChangedChunk], diff: Diff) -> dict[int, set[int]]:
    focus: dict[int, set[int]] = {}
    for item in changed:
        file_diff = diff.file(item.row["path"])
        if file_diff is None:
            continue
        if item.change == "before" or item.change == "deleted":
            lines = file_diff.removed_line_numbers()
        else:
            lines = file_diff.added_line_numbers()
        focus[item.row["id"]] = set(lines)
    return focus


def looks_like_symbol(text: str) -> list[str]:
    stripped = text.strip()
    if " " not in stripped and IDENTIFIER.fullmatch(stripped.replace(".", "_")):
        return [stripped]
    names = []
    for match in IDENTIFIER.finditer(text):
        word = match.group(0)
        if "_" in word or any(character.isupper() for character in word[1:]):
            names.append(word)
    return names[:5]


class RetrievalEngine:
    def __init__(self, view: View, settings: Settings, llm: LLMClient | None = None):
        self.view = view
        self.settings = settings
        self.llm = llm
        self.embedder = QueryEmbedder(llm, settings)
        self.graph = GraphIndex(view)

    def dense_available(self, dims: int | None = None) -> bool:
        return self.embedder.available(self.view, dims)

    def rows_by_id(self, row_ids: list[int]) -> dict:
        return {row["id"]: row for row in self.view.chunks_for_row_ids(row_ids)}

    def headers_for(self, rows_by_id: dict) -> dict[int, str]:
        summaries = load_summaries(self.view, self.settings.models.mini_model, [row["content_hash"] for row in rows_by_id.values()])
        derived = self.view.derived_tags([row["chunk_id"] for row in rows_by_id.values()])
        headers = {}
        for row_id, row in rows_by_id.items():
            headers[row_id] = render_header(row, self.view.all_tags(row, derived), None, summaries.get(row["content_hash"]))
        return headers

    def boost(self, fused: list[ScoredChunk], profile: RetrievalProfile | None) -> list[ScoredChunk]:
        if profile is None or not fused:
            return fused
        rows_by_id = self.rows_by_id([item.row_id for item in fused])
        derived = self.view.derived_tags([item.chunk_id for item in fused])
        tags_by_chunk = {}
        for row in rows_by_id.values():
            tags_by_chunk[row["chunk_id"]] = set(self.view.all_tags(row, derived))
        return apply_boosts(fused, rows_by_id, tags_by_chunk, profile.kind_boosts, profile.tag_boosts)

    def maybe_rerank(self, query: str, focus: str, hunk_text: str | None, fused: list[ScoredChunk], keep: int, enabled: bool) -> list[ScoredChunk]:
        if not enabled or self.llm is None or not fused:
            return fused[:keep]
        candidates = fused[: self.settings.retrieval.rerank_candidates]
        rows_by_id = self.rows_by_id([item.row_id for item in candidates])
        headers = self.headers_for(rows_by_id)
        return rerank(self.llm, query, focus, hunk_text, candidates, rows_by_id, headers, keep)

    def search(
        self,
        query: str,
        filters: SearchFilters | None = None,
        mode: str = "hybrid",
        limit: int = 10,
        rerank_results: bool | None = None,
        hunk_text: str | None = None,
        profile: RetrievalProfile | None = None,
        options: RetrievalOptions | None = None,
    ) -> tuple[list[ScoredChunk], list[RankedList]]:
        filters = filters or SearchFilters()
        options = options or RetrievalOptions()
        retrieval = self.settings.retrieval
        per_list = retrieval.per_retriever_limit
        lists: list[RankedList] = []
        if mode in ("hybrid", "lexical") and options.use_lexical:
            lexical_weight = profile.lexical_weight if profile else retrieval.lexical_weight
            lists.append(RankedList("lexical", lexical_weight, lexical_search(self.view, query, filters, per_list), query))
        if mode in ("hybrid", "dense") and options.use_dense and self.dense_available(options.dims):
            dense_weight = profile.dense_weight if profile else retrieval.dense_weight
            hits = dense_search(self.view, self.embedder, query, filters, per_list, options.facets, options.dims)
            lists.append(RankedList("dense", dense_weight, hits, query))
        if mode in ("hybrid", "graph") and options.use_graph:
            names = looks_like_symbol(query)
            if names:
                graph_weight = profile.graph_weight if profile else retrieval.graph_weight
                lists.append(RankedList("graph", graph_weight, symbol_hits(self.view, names, filters, per_list), query))
        fused = reciprocal_rank_fusion(lists, retrieval.rrf_k)
        fused = self.boost(fused, profile)
        use_rerank = options.use_rerank if rerank_results is None else rerank_results
        focus = profile.focus if profile else "Answer the search query."
        ranked = self.maybe_rerank(query, focus, hunk_text, fused, max(limit, retrieval.rerank_keep) if use_rerank else limit, use_rerank)
        return ranked[:limit], lists

    def first_wave(
        self,
        diff: Diff,
        profile_name: str,
        budget_tokens: int | None = None,
        options: RetrievalOptions | None = None,
        pack: bool = True,
        exclude_changed_files: bool = False,
        keep: int | None = None,
    ) -> FirstWave:
        profile = profile_for(profile_name)
        options = options or RetrievalOptions()
        retrieval = self.settings.retrieval
        changed = changed_chunks(self.view, diff)
        gists = hunk_gists(self.llm, self.view, diff) if options.use_dense and self.dense_available(options.dims) else {}
        queries = build_first_wave(self.view, diff, changed, gists, retrieval.max_hunk_identifiers)
        pinned = pinned_rows(changed)
        pinned_ids = [row_id for row_id, _ in pinned]
        changed_chunk_ids = list(dict.fromkeys(item.row["chunk_id"] for item in changed))
        base_filters = SearchFilters(exclude_changed=exclude_changed_files)

        lists: list[RankedList] = []
        facts: list[ScoredChunk] = []
        fact_ids: set[int] = set()
        pinned_ids_set = {item.row["id"] for item in changed}
        seed_paths = {item.row["path"] for item in changed}
        if options.use_graph and changed_chunk_ids:
            for relation in profile.relations:
                related = self.graph.related(changed_chunk_ids, relation, profile.relation_depth, 200)
                hits = related_to_hits(self.view, related, base_filters, profile.relation_cap, seed_paths)
                taken = 0
                for hit in hits:
                    if taken >= profile.fact_quotas.get(relation, 0):
                        break
                    if hit.row_id in fact_ids or hit.row_id in pinned_ids_set:
                        continue
                    facts.append(ScoredChunk(row_id=hit.row_id, chunk_id=hit.chunk_id, score=0.0, sources=["graph"], reasons=[f"fact: {hit.reason}"]))
                    fact_ids.add(hit.row_id)
                    taken += 1
                if hits:
                    weight = profile.graph_weight * profile.relation_weights.get(relation, 0.5) * options.graph_scale
                    lists.append(RankedList("graph", weight, hits, f"{relation} of changed symbols"))
        if options.use_lexical:
            lexical_filters = base_filters.merged(profile.lexical_filters)
            for query in queries:
                if query.kind != "lexical":
                    continue
                hits = lexical_search(self.view, query.text, lexical_filters, 15)
                if hits:
                    lists.append(RankedList("lexical", profile.lexical_weight * query.weight, hits, query.text))
        if options.use_dense and self.dense_available(options.dims):
            dense_filters = base_filters.merged(profile.dense_filters)
            for query in queries:
                if query.kind != "dense":
                    continue
                hits = dense_search(self.view, self.embedder, query.text, dense_filters, 15, options.facets, options.dims)
                if hits:
                    lists.append(RankedList("dense", profile.dense_weight, hits, query.text))
            if profile.similar_code:
                similar_filters = SearchFilters(exclude_changed=True).merged(profile.dense_filters)
                for item in changed:
                    if item.change not in ("modified", "added") or item.row["kind"] not in ("function", "method"):
                        continue
                    code = truncate_to_tokens(item.row["text"], 1200)
                    hits = dense_search(self.view, self.embedder, code, similar_filters, 8, ["code"], options.dims)
                    if hits:
                        lists.append(RankedList("dense", profile.dense_weight, hits, f"similar to {item.row['symbol']}"))

        fused = reciprocal_rank_fusion(lists, retrieval.rrf_k)
        pinned_set = {item.row["id"] for item in changed}
        fused = [item for item in fused if item.row_id not in pinned_set]
        fused = self.boost(fused, profile)
        hunks_text = "\n\n".join(hunk.text() for hunk in diff.hunks())
        default_keep = retrieval.rerank_keep if options.use_rerank and self.llm is not None else 25
        ranked = self.maybe_rerank(
            f"Context needed to review this change ({profile.name})",
            profile.focus,
            hunks_text,
            fused,
            keep or default_keep,
            options.use_rerank,
        )

        history: list[str] = []
        if profile.include_memory:
            history = self.history_lines(queries, gists)
        context_pack = None
        if pack:
            summaries = load_summaries(self.view, self.settings.models.mini_model, self.all_hashes(pinned_ids, ranked + facts))
            packer = Packer(self.view, summaries, focus_lines_for(changed, diff))
            context_pack = packer.pack(ranked, budget_tokens or profile.budget_tokens, pinned=pinned, history=history, facts=facts)
        return FirstWave(profile=profile.name, queries=queries, changed=changed, ranked=ranked, pack=context_pack, lists=lists, facts=facts)

    def all_hashes(self, pinned_ids: list[int], ranked: list[ScoredChunk]) -> list[str]:
        rows = self.view.chunks_for_row_ids(list(dict.fromkeys(pinned_ids + [item.row_id for item in ranked])))
        return [row["content_hash"] for row in rows]

    def history_lines(self, queries: list[ReviewQuery], gists: dict[str, str]) -> list[str]:
        texts = [query.text for query in queries if query.kind in ("lexical", "dense")]
        rows_by_id: dict[int, dict] = {}
        for text in texts[:12]:
            for row in memory_search(self.view, text, 3):
                rows_by_id.setdefault(row["id"], row)
        if self.dense_available():
            for gist in list(gists.values())[:6]:
                for content_hash, _score in dense_memory_search(self.view, self.embedder, gist, 3):
                    row = self.view.store.query_one("SELECT * FROM memory WHERE content_hash = ?", (content_hash,))
                    if row is not None:
                        rows_by_id.setdefault(row["id"], dict(row))
        lines = []
        for row in list(rows_by_id.values())[:6]:
            date = f" ({row['date']})" if row.get("date") else ""
            lines.append(f"m{row['id']}{date} {row['source']}: {row['subject'][:120]} — touched {row.get('path') or '?'}")
        return lines

    def memory(self, text: str, limit: int = 5) -> list[dict]:
        rows = memory_search(self.view, text, limit)
        if self.dense_available():
            seen = {row["id"] for row in rows}
            for content_hash, _score in dense_memory_search(self.view, self.embedder, text, limit):
                row = self.view.store.query_one("SELECT * FROM memory WHERE content_hash = ?", (content_hash,))
                if row is not None and row["id"] not in seen:
                    rows.append(dict(row))
                    seen.add(row["id"])
        return rows[:limit]
