from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field

import networkx as nx

from ..index.view import CURRENT, View
from ..util import loads
from .lexical import filter_clause
from .models import Hit, SearchFilters

RELATIONS: dict[str, tuple[str, list[str]]] = {
    "callers": ("in", ["calls", "instantiates", "references", "enqueues", "routes", "decorated_by", "registers_viewset"]),
    "callees": ("out", ["calls", "instantiates", "references", "enqueues", "renders", "email_template", "sends_signal", "queries"]),
    "receivers": ("in", ["receives"]),
    "routes": ("in", ["routes"]),
    "templates": ("out", ["renders", "email_template", "extends", "includes"]),
    "rendered_by": ("in", ["renders", "email_template", "extends", "includes"]),
    "subclasses": ("in", ["inherits"]),
    "bases": ("out", ["inherits"]),
    "admin": ("in", ["admin_for", "model_ref"]),
    "queried_by": ("in", ["queries"]),
    "task_callers": ("in", ["enqueues"]),
    "signal_senders": ("in", ["sends_signal"]),
    "links": ("in", ["links_to", "requests"]),
    "imports": ("in", ["imports"]),
}
TEST_LINK_KINDS = ["calls", "instantiates", "references", "queries", "links_to", "requests", "enqueues"]
CONFIDENCE_ORDER = {"exact": 0, "heuristic": 1, "transitive": 2, "missing": 3, "unresolved": 4}


@dataclass
class Related:
    chunk_id: str
    distance: int
    relation: str
    edge_kind: str
    confidence: str
    line: int
    via: str
    extra: dict = field(default_factory=dict)


def resolve_symbol_rows(view: View, name: str, version: str = CURRENT, limit: int = 20) -> list[sqlite3.Row]:
    cleaned = name.strip()
    if ":" in cleaned and cleaned.rsplit(":", 1)[1].isdigit() and "/" in cleaned:
        path, line_text = cleaned.rsplit(":", 1)
        line = int(line_text)
        rows = view.chunk_rows("c.path = ? AND c.start_line <= ? AND c.end_line >= ?", (path, line, line), version=version)
        rows = sorted(rows, key=lambda row: row["end_line"] - row["start_line"])
        return rows[:1]
    rows = view.chunk_rows("c.symbol = ? AND c.part = 0", (cleaned,), version=version, limit=limit)
    if rows:
        return rows
    rows = view.chunk_rows("c.symbol LIKE ? AND c.part = 0", ("%." + cleaned,), version=version, limit=limit)
    if rows:
        return rows
    short = cleaned.rsplit(".", 1)[-1]
    rows = view.chunk_rows("c.name = ? AND c.part = 0 AND c.kind NOT IN ('doc', 'config', 'file_block')", (short,), version=version, limit=limit)
    return rows


class GraphIndex:
    def __init__(self, view: View):
        self.view = view

    def related(self, seeds: list[str], relation: str, depth: int = 1, limit: int = 60, version: str = CURRENT) -> list[Related]:
        if relation == "tests":
            return self.tests_for(seeds, version)[:limit]
        if relation == "cache_peers":
            return self.cache_peers(seeds, version)[:limit]
        if relation in ("model_receivers", "model_queries", "model_admin"):
            return self.through_models(seeds, relation, version)[:limit]
        direction, kinds = RELATIONS[relation]
        graph = nx.DiGraph()
        frontier = list(dict.fromkeys(seeds))
        for seed in frontier:
            graph.add_node(seed)
        found: dict[str, Related] = {}
        for distance in range(1, depth + 1):
            if not frontier:
                break
            if direction == "in":
                edges = self.view.edges_to(frontier, kinds, version)
            else:
                edges = self.view.edges_from(frontier, kinds, version)
            next_frontier = []
            for edge in edges:
                if direction == "in":
                    source_node, other = edge["dst_chunk_id"], edge["src_chunk_id"]
                else:
                    source_node, other = edge["src_chunk_id"], edge["dst_chunk_id"]
                if other is None or other in seeds:
                    continue
                graph.add_edge(source_node, other, kind=edge["kind"], confidence=edge["confidence"])
                candidate = Related(
                    chunk_id=other,
                    distance=distance,
                    relation=relation,
                    edge_kind=edge["kind"],
                    confidence=edge["confidence"],
                    line=edge["line"],
                    via=source_node,
                    extra=loads(edge["extra"], {}),
                )
                existing = found.get(other)
                if existing is None or (distance, CONFIDENCE_ORDER[candidate.confidence]) < (existing.distance, CONFIDENCE_ORDER[existing.confidence]):
                    found[other] = candidate
                    next_frontier.append(other)
            frontier = list(dict.fromkeys(next_frontier))
        ordered = sorted(found.values(), key=lambda item: (item.distance, CONFIDENCE_ORDER.get(item.confidence, 9), item.edge_kind))
        return ordered[:limit]

    def neighborhood(self, seeds: list[str], relations: list[str], depth: int = 2) -> nx.DiGraph:
        graph = nx.DiGraph()
        for seed in seeds:
            graph.add_node(seed, seed=True)
        for relation in relations:
            for item in self.related(seeds, relation, depth):
                graph.add_node(item.chunk_id)
                if RELATIONS.get(relation, ("in", []))[0] == "in":
                    graph.add_edge(item.chunk_id, item.via, kind=item.edge_kind, confidence=item.confidence, relation=relation)
                else:
                    graph.add_edge(item.via, item.chunk_id, kind=item.edge_kind, confidence=item.confidence, relation=relation)
        return graph

    def test_chunk_ids(self, candidate_ids: list[str], version: str) -> set[str]:
        if not candidate_ids:
            return set()
        rows = self.view.chunks_for_ids(list(set(candidate_ids)), version)
        tests = set()
        for row in rows:
            tags = loads(row["tags"], [])
            if row["kind"] == "test" or "test" in tags or "test_support" in tags or "fixture" in tags:
                tests.add(row["chunk_id"])
        return tests

    def tests_for(self, seeds: list[str], version: str = CURRENT) -> list[Related]:
        results: dict[str, Related] = {}

        def consider(edges, via_label: str, confidence_override: str | None = None, target_of: dict | None = None) -> None:
            sources = [edge["src_chunk_id"] for edge in edges]
            tests = self.test_chunk_ids(sources, version)
            for edge in edges:
                if edge["src_chunk_id"] not in tests:
                    continue
                confidence = confidence_override or edge["confidence"]
                via = target_of.get(edge["dst_chunk_id"], edge["dst_chunk_id"]) if target_of else edge["dst_chunk_id"]
                candidate = Related(edge["src_chunk_id"], 1, "tests", edge["kind"], confidence, edge["line"], via, {"via": via_label})
                existing = results.get(candidate.chunk_id)
                if existing is None or CONFIDENCE_ORDER[candidate.confidence] < CONFIDENCE_ORDER[existing.confidence]:
                    results[candidate.chunk_id] = candidate

        consider(self.view.edges_to(seeds, TEST_LINK_KINDS, version), "direct")

        url_edges = self.view.edges_to(seeds, ["routes"], version)
        url_to_view = {edge["src_chunk_id"]: edge["dst_chunk_id"] for edge in url_edges}
        if url_to_view:
            consider(self.view.edges_to(list(url_to_view), ["links_to"], version), "url_name", target_of=url_to_view)

        factory_edges = self.view.edges_to(seeds, ["model_ref"], version)
        factory_ids = [edge["src_chunk_id"] for edge in factory_edges]
        if factory_ids:
            factory_target = {edge["src_chunk_id"]: edge["dst_chunk_id"] for edge in factory_edges}
            consider(self.view.edges_to(factory_ids, ["instantiates", "calls"], version), "factory", "heuristic", factory_target)

        caller_edges = self.view.edges_to(seeds, ["calls", "instantiates", "references", "enqueues"], version)
        caller_ids = [edge["src_chunk_id"] for edge in caller_edges if edge["confidence"] == "exact"][:40]
        if caller_ids:
            caller_target = {edge["src_chunk_id"]: edge["dst_chunk_id"] for edge in caller_edges}
            consider(self.view.edges_to(caller_ids, TEST_LINK_KINDS, version), "transitive", "transitive", caller_target)

        return sorted(results.values(), key=lambda item: (CONFIDENCE_ORDER.get(item.confidence, 9), item.chunk_id))

    def touched_models(self, seeds: list[str], version: str) -> dict[str, str]:
        models: dict[str, str] = {}
        for edge in self.view.edges_from(seeds, ["queries", "model_ref", "instantiates"], version):
            if edge["dst_chunk_id"]:
                models.setdefault(edge["dst_chunk_id"], edge["src_chunk_id"])
        rows = self.view.chunks_for_ids(list(models) + list(seeds), version)
        model_ids = set()
        for row in rows:
            tags = loads(row["tags"], [])
            if row["kind"] == "model" or "model" in tags:
                model_ids.add(row["chunk_id"])
        derived = self.view.derived_tags(list(models))
        for chunk_id, tags in derived.items():
            if "model" in tags:
                model_ids.add(chunk_id)
        result = {}
        for chunk_id in model_ids:
            result[chunk_id] = models.get(chunk_id, chunk_id)
        return result

    def through_models(self, seeds: list[str], relation: str, version: str) -> list[Related]:
        models = self.touched_models(seeds, version)
        if not models:
            return []
        edge_kinds = {"model_receivers": ["receives"], "model_queries": ["queries"], "model_admin": ["admin_for", "model_ref"]}[relation]
        results: dict[str, Related] = {}
        for edge in self.view.edges_to(list(models), edge_kinds, version):
            source = edge["src_chunk_id"]
            if source in seeds or source in results:
                continue
            extra = loads(edge["extra"], {})
            extra["model"] = edge["dst_name"]
            results[source] = Related(source, 2, relation, edge["kind"], edge["confidence"], edge["line"], edge["dst_chunk_id"], extra)
        return sorted(results.values(), key=lambda item: (CONFIDENCE_ORDER.get(item.confidence, 9), item.chunk_id))

    def cache_peers(self, seeds: list[str], version: str = CURRENT) -> list[Related]:
        own = self.view.edges_from(seeds, ["cache"], version)
        keys = sorted({edge["dst_name"] for edge in own if not edge["dst_name"].startswith("cache:<")})
        if not keys:
            return []
        placeholders = ",".join("?" for _ in keys)
        peers = self.view.edge_rows(f"kind = 'cache' AND dst_name IN ({placeholders})", tuple(keys), version)
        results = []
        seen = set()
        for edge in peers:
            if edge["src_chunk_id"] in seeds or edge["src_chunk_id"] in seen:
                continue
            seen.add(edge["src_chunk_id"])
            extra = loads(edge["extra"], {})
            results.append(Related(edge["src_chunk_id"], 1, "cache_peers", "cache", "exact", edge["line"], edge["dst_name"], extra))
        return results


def path_proximity(path: str, seed_paths: set[str]) -> int:
    if path in seed_paths:
        return 0
    directory = path.rsplit("/", 1)[0]
    best = 3
    for seed in seed_paths:
        seed_directory = seed.rsplit("/", 1)[0]
        if directory == seed_directory:
            return 1
        if directory.split("/")[:2] == seed_directory.split("/")[:2]:
            best = min(best, 2)
    return best


def related_to_hits(view: View, related: list[Related], filters: SearchFilters, limit: int, seed_paths: set[str] | None = None) -> list[Hit]:
    if not related:
        return []
    where, parameters = filter_clause(view, filters)
    ids = [item.chunk_id for item in related]
    placeholders = ",".join("?" for _ in ids)
    rows = view.chunk_rows(f"c.chunk_id IN ({placeholders}) AND {where}", (*ids, *parameters), version=filters.version)
    row_by_chunk = {row["chunk_id"]: row for row in rows}
    if seed_paths:
        def order_key(item: Related) -> tuple:
            row = row_by_chunk.get(item.chunk_id)
            proximity = path_proximity(row["path"], seed_paths) if row is not None else 9
            return (item.distance, CONFIDENCE_ORDER.get(item.confidence, 9), proximity)

        related = sorted(related, key=order_key)
    hits = []
    for item in related:
        row = row_by_chunk.get(item.chunk_id)
        if row is None:
            continue
        reason = f"{item.relation}:{item.edge_kind}:{item.confidence}"
        hits.append(Hit(row_id=row["id"], chunk_id=row["chunk_id"], source="graph", rank=len(hits) + 1, score=1.0 / item.distance, reason=reason))
        if len(hits) >= limit:
            break
    return hits


def symbol_hits(view: View, names: list[str], filters: SearchFilters, limit: int) -> list[Hit]:
    hits: list[Hit] = []
    seen = set()
    where, parameters = filter_clause(view, filters)
    for name in names:
        for row in resolve_symbol_rows(view, name, filters.version):
            if row["id"] in seen:
                continue
            allowed = view.chunk_rows(f"c.id = ? AND {where}", (row["id"], *parameters), version=filters.version)
            if not allowed:
                continue
            seen.add(row["id"])
            hits.append(Hit(row_id=row["id"], chunk_id=row["chunk_id"], source="graph", rank=len(hits) + 1, score=1.0, reason=f"symbol:{name}"))
            if len(hits) >= limit:
                return hits
    return hits
