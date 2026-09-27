from __future__ import annotations

from .models import RankedList, ScoredChunk


def reciprocal_rank_fusion(lists: list[RankedList], k: int) -> list[ScoredChunk]:
    fused: dict[int, ScoredChunk] = {}
    for ranked in lists:
        for hit in ranked.hits:
            contribution = ranked.weight / (k + hit.rank)
            scored = fused.get(hit.row_id)
            if scored is None:
                scored = ScoredChunk(row_id=hit.row_id, chunk_id=hit.chunk_id, score=0.0)
                fused[hit.row_id] = scored
            scored.score += contribution
            if ranked.source not in scored.sources:
                scored.sources.append(ranked.source)
            reason = f"{hit.reason}" + (f" <- {ranked.query[:60]}" if ranked.query else "")
            if reason and reason not in scored.reasons and len(scored.reasons) < 6:
                scored.reasons.append(reason)
    return sorted(fused.values(), key=lambda item: item.score, reverse=True)


def apply_boosts(scored: list[ScoredChunk], rows_by_id: dict, tags_by_chunk: dict[str, set[str]], kind_boosts: dict[str, float], tag_boosts: dict[str, float]) -> list[ScoredChunk]:
    for item in scored:
        row = rows_by_id.get(item.row_id)
        if row is None:
            continue
        multiplier = 1.0 + kind_boosts.get(row["kind"], 0.0)
        for tag in tags_by_chunk.get(item.chunk_id, set()):
            multiplier += tag_boosts.get(tag, 0.0)
            prefix = tag.split(":", 1)[0] + ":*"
            multiplier += tag_boosts.get(prefix, 0.0)
        item.score *= multiplier
    return sorted(scored, key=lambda entry: entry.score, reverse=True)
