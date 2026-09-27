from __future__ import annotations

from ..config import Settings
from ..index.store import pack_vector
from ..index.view import View
from ..llm.client import LLMClient
from .lexical import filter_clause
from .models import Hit, SearchFilters


class QueryEmbedder:
    def __init__(self, llm: LLMClient | None, settings: Settings):
        self.llm = llm
        self.settings = settings
        self.cache: dict[tuple[str, int, str], list[float]] = {}

    def available(self, view: View, dims: int | None = None) -> bool:
        if self.llm is None:
            return False
        model = self.settings.models.embedding_model
        return view.store.vector_table(model, dims or self.settings.models.embedding_dims) is not None

    def vector(self, text: str, dims: int | None = None) -> list[float]:
        dims = dims or self.settings.models.embedding_dims
        model = self.settings.models.embedding_model
        key = (model, dims, text)
        if key not in self.cache:
            self.cache[key] = self.llm.embed([text], purpose="retrieval.query", model=model, dims=dims)[0]
        return self.cache[key]


def knn_content_hashes(view: View, table: str, vector: list[float], k: int, facets: list[str]) -> list[tuple[str, str, float]]:
    placeholders = ",".join("?" for _ in facets)
    rows = view.store.query(
        f"WITH knn AS (SELECT rowid, distance FROM {table} WHERE embedding MATCH ? AND k = ?) "
        f"SELECT vr.content_hash AS content_hash, vr.facet AS facet, knn.distance AS distance "
        f"FROM knn JOIN vector_rows vr ON vr.rowid = knn.rowid "
        f"WHERE vr.table_name = ? AND vr.facet IN ({placeholders}) ORDER BY knn.distance",
        (pack_vector(vector), k, table, *facets),
    )
    return [(row["content_hash"], row["facet"], row["distance"]) for row in rows]


def dense_search(
    view: View,
    embedder: QueryEmbedder,
    text: str,
    filters: SearchFilters,
    limit: int,
    facets: list[str] | None = None,
    dims: int | None = None,
) -> list[Hit]:
    if not embedder.available(view, dims):
        return []
    settings = embedder.settings
    dims = dims or settings.models.embedding_dims
    table = view.store.vector_table(settings.models.embedding_model, dims)
    facets = facets or ["code", "summary"]
    vector = embedder.vector(text, dims)
    best_distance: dict[str, tuple[float, str]] = {}
    k = max(limit * settings.retrieval.dense_overfetch, 40)
    for _ in range(3):
        for content_hash, facet, distance in knn_content_hashes(view, table, vector, k, facets):
            if content_hash not in best_distance or distance < best_distance[content_hash][0]:
                best_distance[content_hash] = (distance, facet)
        hits = rows_for_hashes(view, best_distance, filters, limit)
        if len(hits) >= limit or k >= 4000:
            return hits
        k *= 4
    return rows_for_hashes(view, best_distance, filters, limit)


def rows_for_hashes(view: View, best_distance: dict[str, tuple[float, str]], filters: SearchFilters, limit: int) -> list[Hit]:
    if not best_distance:
        return []
    where, parameters = filter_clause(view, filters)
    hashes = list(best_distance)
    placeholders = ",".join("?" for _ in hashes)
    rows = view.chunk_rows(f"c.content_hash IN ({placeholders}) AND {where}", (*hashes, *parameters), version=filters.version)
    ordered = sorted(rows, key=lambda row: best_distance[row["content_hash"]][0])
    hits = []
    seen = set()
    for row in ordered:
        if row["id"] in seen:
            continue
        seen.add(row["id"])
        distance, facet = best_distance[row["content_hash"]]
        hits.append(Hit(row_id=row["id"], chunk_id=row["chunk_id"], source="dense", rank=len(hits) + 1, score=1 - distance, reason=f"dense:{facet}"))
        if len(hits) >= limit:
            break
    return hits


def dense_memory_search(view: View, embedder: QueryEmbedder, text: str, limit: int) -> list[tuple[str, float]]:
    if not embedder.available(view):
        return []
    settings = embedder.settings
    table = view.store.vector_table(settings.models.embedding_model, settings.models.embedding_dims)
    vector = embedder.vector(text)
    results = knn_content_hashes(view, table, vector, max(limit * 4, 20), ["memory"])
    return [(content_hash, 1 - distance) for content_hash, _, distance in results[:limit]]
