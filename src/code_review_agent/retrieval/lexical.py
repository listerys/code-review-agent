from __future__ import annotations

import re

from ..index.view import View
from ..util import IDENTIFIER, split_identifier
from .models import Hit, SearchFilters

STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "of",
    "to",
    "in",
    "on",
    "for",
    "is",
    "are",
    "be",
    "by",
    "with",
    "where",
    "what",
    "which",
    "who",
    "how",
    "when",
    "that",
    "this",
    "it",
    "as",
    "at",
    "from",
    "into",
    "does",
    "do",
    "self",
    "def",
    "return",
    "import",
    "none",
    "true",
    "false",
    "if",
    "else",
    "not",
}


def filter_clause(view: View, filters: SearchFilters) -> tuple[str, list]:
    clauses: list[str] = []
    parameters: list = []
    if filters.kinds:
        clauses.append(f"c.kind IN ({','.join('?' for _ in filters.kinds)})")
        parameters.extend(filters.kinds)
    if filters.exclude_kinds:
        clauses.append(f"c.kind NOT IN ({','.join('?' for _ in filters.exclude_kinds)})")
        parameters.extend(filters.exclude_kinds)
    if filters.path_prefix:
        clauses.append("c.path LIKE ?")
        parameters.append(filters.path_prefix.rstrip("*") + "%")
    if filters.exclude_changed:
        clauses.append(f"c.path NOT IN (SELECT path FROM {view.overridden_table})")
    if filters.tags_any:
        placeholders = ",".join("?" for _ in filters.tags_any)
        clauses.append(
            f"(EXISTS (SELECT 1 FROM json_each(c.tags) WHERE value IN ({placeholders})) "
            f"OR EXISTS (SELECT 1 FROM chunk_tags t WHERE t.view_key = ? AND t.chunk_id = c.chunk_id AND t.tag IN ({placeholders})))"
        )
        parameters.extend(filters.tags_any)
        parameters.append(view.tags_key())
        parameters.extend(filters.tags_any)
    if not clauses:
        return "1=1", parameters
    return " AND ".join(clauses), parameters


def quote_term(term: str) -> str:
    return '"' + term.replace('"', '""') + '"'


def query_terms(text: str) -> list[str]:
    terms: list[str] = []
    seen = set()
    for match in IDENTIFIER.finditer(text):
        word = match.group(0)
        pieces = split_identifier(word)
        if not pieces:
            continue
        if len(pieces) > 1:
            phrase = " ".join(pieces)
            if phrase not in seen:
                seen.add(phrase)
                terms.append(phrase)
            continue
        lowered = pieces[0]
        if lowered in STOPWORDS or len(lowered) < 2 or lowered in seen:
            continue
        seen.add(lowered)
        terms.append(lowered)
    return terms[:32]


def fts_match_expression(text: str) -> str | None:
    literal = text.strip()
    if literal and re.search(r"[{}:/%.]", literal) and " " not in literal and len(literal) <= 120:
        pieces = [piece for piece in re.split(r"[^A-Za-z0-9_]+", literal) if piece]
        words = []
        for piece in pieces:
            words.extend(split_identifier(piece) or [piece.lower()])
        if words:
            return quote_term(" ".join(words))
    terms = query_terms(text)
    if not terms:
        return None
    return " OR ".join(quote_term(term) for term in terms)


def lexical_search(view: View, text: str, filters: SearchFilters, limit: int) -> list[Hit]:
    expression = fts_match_expression(text)
    if expression is None:
        return []
    where, parameters = filter_clause(view, filters)
    sql = (
        f"SELECT c.id AS id, c.chunk_id AS chunk_id, bm25(chunk_fts) AS score FROM chunk_fts "
        f"JOIN chunks c ON c.id = chunk_fts.rowid "
        f"JOIN {view.active_table} a ON c.path = a.path AND c.blob_sha = a.blob_sha "
        f"WHERE chunk_fts MATCH ? AND {view.version_clause(filters.version)} AND {where} "
        f"ORDER BY score LIMIT ?"
    )
    rows = view.store.query(sql, (expression, *parameters, limit))
    hits = []
    for rank, row in enumerate(rows, start=1):
        hits.append(Hit(row_id=row["id"], chunk_id=row["chunk_id"], source="lexical", rank=rank, score=-row["score"], reason="bm25"))
    return hits


def document_frequency(view: View, term: str) -> int:
    expression = fts_match_expression(term)
    if expression is None:
        return 0
    row = view.store.query_one("SELECT count(*) AS total FROM chunk_fts WHERE chunk_fts MATCH ?", (expression,))
    return row["total"] if row else 0


def memory_search(view: View, text: str, limit: int) -> list[dict]:
    expression = fts_match_expression(text)
    if expression is None:
        return []
    rows = view.store.query(
        "SELECT m.*, bm25(memory_fts) AS score FROM memory_fts JOIN memory m ON m.id = memory_fts.rowid "
        "WHERE memory_fts MATCH ? ORDER BY score LIMIT ?",
        (expression, limit),
    )
    return [dict(row) for row in rows]
