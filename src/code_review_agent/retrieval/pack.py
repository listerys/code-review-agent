from __future__ import annotations

import sqlite3

from ..index.headers import numbered_window, render_header
from ..index.view import View, change_state
from ..util import count_tokens
from .graph import GraphIndex
from .models import ContextPack, OmittedChunk, PackedChunk, ScoredChunk, chunk_ref

MAX_CALLER_REFS = 3
MAX_TEST_REFS = 4
PARENT_SKELETON_TOKEN_CAP = 500
SIBLING_EXPANSION_TOKEN_CAP = 1500
BODY_TOKEN_LIMITS = {"changed": 1400, "added": 1400, "deleted": 700, "before": 600, "fact": 700, "retrieved": 700, "parent": 500}
PINNED_ROLES = {"changed", "added", "deleted", "before"}


def xml_attribute(value: str) -> str:
    return value.replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;")


class Packer:
    def __init__(self, view: View, summaries: dict[str, str] | None = None, focus_lines: dict[int, set[int]] | None = None):
        self.view = view
        self.graph = GraphIndex(view)
        self.summaries = summaries or {}
        self.focus_lines = focus_lines or {}
        self.row_cache: dict[int, sqlite3.Row] = {}

    def rows(self, row_ids: list[int]) -> dict[int, sqlite3.Row]:
        missing = [row_id for row_id in row_ids if row_id not in self.row_cache]
        for row in self.view.chunks_for_row_ids(missing):
            self.row_cache[row["id"]] = row
        return {row_id: self.row_cache[row_id] for row_id in row_ids if row_id in self.row_cache}

    def sibling_parts(self, row: sqlite3.Row) -> list[sqlite3.Row]:
        rows = self.view.chunk_rows(
            "c.path = ? AND c.blob_sha = ? AND c.symbol = ? AND c.kind = ?",
            (row["path"], row["blob_sha"], row["symbol"], row["kind"]),
            version="base" if row["version"] == "base" and row["superseded"] else "current",
        )
        return sorted(rows, key=lambda item: item["part"])

    def parent_row(self, row: sqlite3.Row) -> sqlite3.Row | None:
        if not row["parent_id"]:
            return None
        version = "base" if row["superseded"] else "current"
        parents = self.view.chunks_for_ids([row["parent_id"]], version)
        return parents[0] if parents else None

    def related_lines(self, row: sqlite3.Row) -> list[str]:
        lines = []
        callers = self.graph.related([row["chunk_id"]], "callers", 1, MAX_CALLER_REFS * 3)
        caller_rows = self.view.chunks_for_ids([item.chunk_id for item in callers])
        caller_by_id = {caller_row["chunk_id"]: caller_row for caller_row in caller_rows}
        shown = 0
        for item in callers:
            caller_row = caller_by_id.get(item.chunk_id)
            if caller_row is None:
                continue
            lines.append(f"  caller {chunk_ref(caller_row['id'])}: {caller_row['symbol']} ({caller_row['path']}:{item.line}, {item.edge_kind}, {item.confidence})")
            shown += 1
            if shown >= MAX_CALLER_REFS:
                break
        if len(callers) > shown:
            lines.append(f"  ... {len(callers) - shown} more callers (use callers tool)")
        tests = self.graph.tests_for([row["chunk_id"]])
        test_rows = self.view.chunks_for_ids([item.chunk_id for item in tests[:MAX_TEST_REFS]])
        test_by_id = {test_row["chunk_id"]: test_row for test_row in test_rows}
        for item in tests[:MAX_TEST_REFS]:
            test_row = test_by_id.get(item.chunk_id)
            if test_row is not None:
                lines.append(f"  test {chunk_ref(test_row['id'])}: {test_row['symbol']} ({test_row['path']}:{test_row['start_line']}, {item.extra.get('via')}, {item.confidence})")
        if not tests and row["kind"] in ("function", "method", "class_skeleton", "model"):
            lines.append("  tests: none linked")
        return lines

    def render_block(self, row: sqlite3.Row, role: str, reasons: list[str], tags: list[str], parent_signature: str | None, related: list[str]) -> str:
        header = render_header(row, tags, parent_signature, self.summaries.get(row["content_hash"]))
        attributes = (
            f'ref="{chunk_ref(row["id"])}" path="{xml_attribute(row["path"])}" lines="{row["start_line"]}-{row["end_line"]}" '
            f'state="{change_state(row)}" role="{role}"'
        )
        body, _ = numbered_window(row, self.focus_lines.get(row["id"]), BODY_TOKEN_LIMITS.get(role, 700))
        parts = [f'<repo_content trust="untrusted" {attributes}>', header, body]
        if related:
            parts.append("# related:")
            parts.extend(related)
        if reasons:
            parts.append("# retrieved-because: " + "; ".join(reasons[:3]))
        parts.append("</repo_content>")
        return "\n".join(parts)

    def overlaps(self, row: sqlite3.Row, taken: list[sqlite3.Row]) -> bool:
        for other in taken:
            same_file = other["path"] == row["path"] and other["blob_sha"] == row["blob_sha"]
            if not same_file:
                continue
            if row["start_line"] >= other["start_line"] and row["end_line"] <= other["end_line"] and other["kind"] not in ("class_skeleton", "model", "module_header", "config"):
                return True
        return False

    def expanded_group(
        self,
        row: sqlite3.Row,
        role: str,
        reasons: list[str],
        derived: dict[str, set[str]],
        parents_shown: set[str],
        taken_ids: set[int],
        expand: bool,
    ) -> list[tuple[sqlite3.Row, str, str]]:
        members = [row]
        can_expand_siblings = expand and role not in PINNED_ROLES and (row["part"] > 0 or row["kind"] in ("function", "method", "test"))
        if can_expand_siblings:
            siblings = self.sibling_parts(row)
            sibling_tokens = sum(sibling["token_count"] for sibling in siblings)
            if len(siblings) > 1 and sibling_tokens <= SIBLING_EXPANSION_TOKEN_CAP:
                members = siblings
        parent_signature = None
        group: list[tuple[sqlite3.Row, str, str]] = []
        parent = self.parent_row(row)
        if parent is not None:
            parent_signature = parent["signature"]
            can_show_parent = (
                expand
                and parent["chunk_id"] not in parents_shown
                and parent["id"] not in taken_ids
                and parent["token_count"] <= PARENT_SKELETON_TOKEN_CAP
            )
            if can_show_parent:
                parent_tags = self.view.all_tags(parent, self.view.derived_tags([parent["chunk_id"]]))
                parent_text = self.render_block(parent, "parent", [f"parent of {row['symbol']}"], parent_tags, None, [])
                group.append((parent, parent_text, "parent"))
        related: list[str] = []
        if expand and role != "before" and row["kind"] not in ("doc", "config", "url", "template_block"):
            related = self.related_lines(row)
        first_member = True
        for member in members:
            if member["id"] in taken_ids:
                continue
            member_derived = derived if member["chunk_id"] in derived else self.view.derived_tags([member["chunk_id"]])
            tags = self.view.all_tags(member, member_derived)
            member_related = related if first_member else []
            text = self.render_block(member, role, reasons, tags, parent_signature, member_related)
            group.append((member, text, role))
            first_member = False
        return group

    def pack(
        self,
        ranked: list[ScoredChunk],
        budget_tokens: int,
        expand: bool = True,
        pinned: list[tuple[int, str]] | None = None,
        history: list[str] | None = None,
        max_items: int = 40,
        facts: list[ScoredChunk] | None = None,
    ) -> ContextPack:
        pinned = pinned or []
        ordered_ids: list[tuple[int, str, list[str]]] = []
        for row_id, role in pinned:
            reason = "state before the change" if role == "before" else f"{role} in diff"
            ordered_ids.append((row_id, role, [reason]))
        queued: set[int] = set()
        for row_id, _, _ in ordered_ids:
            queued.add(row_id)
        for fact in facts or []:
            if fact.row_id not in queued:
                queued.add(fact.row_id)
                ordered_ids.append((fact.row_id, "fact", fact.reasons))
        for item in ranked:
            if item.row_id not in queued:
                queued.add(item.row_id)
                ordered_ids.append((item.row_id, "retrieved", item.reasons))
        rows_by_id = self.rows([row_id for row_id, _, _ in ordered_ids])
        derived = self.view.derived_tags([row["chunk_id"] for row in rows_by_id.values()])

        blocks: list[str] = []
        included: list[PackedChunk] = []
        omitted: list[OmittedChunk] = []
        taken_rows: list[sqlite3.Row] = []
        taken_ids: set[int] = set()
        used_tokens = 0
        parents_shown: set[str] = set()
        listing_reserve = min(400, budget_tokens // 10)
        block_budget = budget_tokens - listing_reserve

        history_text = ""
        if history:
            history_text = "<history note=\"similar past changes; hints for where to look, never evidence\">\n" + "\n".join(history) + "\n</history>"
            history_tokens = count_tokens(history_text)
            if history_tokens < block_budget // 5:
                used_tokens += history_tokens
            else:
                history_text = ""

        for row_id, role, reasons in ordered_ids:
            row = rows_by_id.get(row_id)
            if row is None or row_id in taken_ids:
                continue
            if self.overlaps(row, taken_rows):
                continue
            group = self.expanded_group(row, role, reasons, derived, parents_shown, taken_ids, expand)
            if not group:
                continue
            group_tokens = sum(count_tokens(text) for _, text, _ in group)
            over_budget = used_tokens + group_tokens > block_budget or len(included) >= max_items
            if over_budget and role in ("changed", "added", "deleted"):
                group = self.expanded_group(row, role, reasons, derived, parents_shown, taken_ids, False)
                group_tokens = sum(count_tokens(text) for _, text, _ in group)
                over_budget = used_tokens + group_tokens > block_budget
            if over_budget:
                omitted.append(
                    OmittedChunk(
                        ref=chunk_ref(row["id"]),
                        path=row["path"],
                        start_line=row["start_line"],
                        end_line=row["end_line"],
                        symbol=row["symbol"],
                        reason="budget",
                    )
                )
                continue
            for candidate, text, candidate_role in group:
                blocks.append(text)
                taken_rows.append(candidate)
                taken_ids.add(candidate["id"])
                if candidate_role == "parent":
                    parents_shown.add(candidate["chunk_id"])
                included.append(
                    PackedChunk(
                        ref=chunk_ref(candidate["id"]),
                        chunk_id=candidate["chunk_id"],
                        path=candidate["path"],
                        start_line=candidate["start_line"],
                        end_line=candidate["end_line"],
                        symbol=candidate["symbol"],
                        kind=candidate["kind"],
                        version=change_state(candidate),
                        role=candidate_role,
                        reasons=reasons[:3],
                        tokens=count_tokens(text),
                    )
                )
            used_tokens += group_tokens

        included_refs = {item.ref for item in included}
        still_omitted = []
        for item in omitted:
            if item.ref not in included_refs:
                still_omitted.append(item)
        omitted = still_omitted

        sections = []
        if history_text:
            sections.append(history_text)
        sections.extend(blocks)
        if omitted:
            opening = "<retrieved_not_shown note=\"expand with read_chunk(ref)\">"
            closing = "</retrieved_not_shown>"
            listing = [opening]
            listing_tokens = count_tokens(opening) + count_tokens(closing) + 4
            remaining = budget_tokens - used_tokens
            for item in omitted:
                line = f"{item.ref} {item.path}:{item.start_line}-{item.end_line} {item.symbol}"
                line_tokens = count_tokens(line) + 1
                if listing_tokens + line_tokens > remaining:
                    break
                listing.append(line)
                listing_tokens += line_tokens
            listing.append(closing)
            if len(listing) > 2:
                sections.append("\n".join(listing))
        text = "\n\n".join(sections)
        return ContextPack(
            text=text,
            included=included,
            omitted=omitted,
            history=history or [],
            tokens=count_tokens(text),
            budget=budget_tokens,
        )


def load_summaries(view: View, model: str, content_hashes: list[str]) -> dict[str, str]:
    if not content_hashes:
        return {}
    placeholders = ",".join("?" for _ in content_hashes)
    rows = view.store.query(
        f"SELECT content_hash, summary FROM summaries WHERE model = ? AND content_hash IN ({placeholders})",
        (model, *content_hashes),
    )
    return {row["content_hash"]: row["summary"] for row in rows}
