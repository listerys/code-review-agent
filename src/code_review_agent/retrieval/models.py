from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field


class SearchFilters(BaseModel):
    kinds: list[str] | None = None
    exclude_kinds: list[str] | None = None
    tags_any: list[str] | None = None
    path_prefix: str | None = None
    version: Literal["current", "base", "head"] = "current"
    exclude_changed: bool = False

    def merged(self, other: SearchFilters | None) -> SearchFilters:
        if other is None:
            return self
        data = self.model_dump()
        for key, value in other.model_dump(exclude_defaults=True).items():
            data[key] = value
        return SearchFilters.model_validate(data)


@dataclass
class Hit:
    row_id: int
    chunk_id: str
    source: str
    rank: int
    score: float = 0.0
    reason: str = ""


@dataclass
class RankedList:
    source: str
    weight: float
    hits: list[Hit]
    query: str = ""


@dataclass
class ScoredChunk:
    row_id: int
    chunk_id: str
    score: float
    sources: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    relevance: int | None = None


@dataclass
class ReviewQuery:
    kind: Literal["exact", "lexical", "dense"]
    text: str
    origin: str
    weight: float = 1.0


class PackedChunk(BaseModel):
    ref: str
    chunk_id: str
    path: str
    start_line: int
    end_line: int
    symbol: str
    kind: str
    version: str
    role: str
    reasons: list[str] = Field(default_factory=list)
    tokens: int = 0


class OmittedChunk(BaseModel):
    ref: str
    path: str
    start_line: int
    end_line: int
    symbol: str
    reason: str


class ContextPack(BaseModel):
    text: str
    included: list[PackedChunk] = Field(default_factory=list)
    omitted: list[OmittedChunk] = Field(default_factory=list)
    history: list[str] = Field(default_factory=list)
    tokens: int = 0
    budget: int = 0


def chunk_ref(row_id: int) -> str:
    return f"c{row_id}"


def parse_chunk_ref(ref: str) -> int | None:
    cleaned = ref.strip()
    if cleaned.startswith("c") and cleaned[1:].isdigit():
        return int(cleaned[1:])
    if cleaned.isdigit():
        return int(cleaned)
    return None
