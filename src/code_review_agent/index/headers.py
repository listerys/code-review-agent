from __future__ import annotations

import sqlite3
import textwrap

from ..util import count_tokens, loads

HEADER_TAG_LIMIT = 12


def display_tags(tags: list[str]) -> list[str]:
    ordered = sorted(tags, key=lambda tag: (":" in tag, tag))
    return ordered[:HEADER_TAG_LIMIT]


def render_header(
    row: sqlite3.Row | dict,
    tags: list[str],
    parent_signature: str | None,
    summary: str | None,
) -> str:
    lines = [
        f"# path: {row['path']}  L{row['start_line']}-{row['end_line']}",
        f"# symbol: {row['symbol']} ({row['kind']})",
    ]
    if parent_signature:
        lines.append(f"# parent: {parent_signature.splitlines()[0].strip().rstrip(':')}")
    decorators = loads(row["decorators"], [])
    if decorators:
        lines.append("# decorators: " + ", ".join("@" + decorator for decorator in decorators[:6]))
    if tags:
        lines.append("# tags: " + ", ".join(display_tags(tags)))
    if summary:
        wrapped = textwrap.wrap(summary, width=100)
        lines.append("# summary: " + wrapped[0])
        for continuation in wrapped[1:3]:
            lines.append("#          " + continuation)
    return "\n".join(lines)


def numbered_text(row: sqlite3.Row | dict) -> str:
    text_lines = row["text"].split("\n")
    line_numbers = loads(row["line_numbers"], None)
    if not line_numbers:
        line_numbers = list(range(row["start_line"], row["start_line"] + len(text_lines)))
    width = len(str(max(line_numbers))) if line_numbers else 3
    output = []
    previous = None
    for number, text in zip(line_numbers, text_lines, strict=False):
        if previous is not None and number != previous + 1:
            output.append(" " * width + " | ...")
        output.append(f"{number:>{width}} | {text}")
        previous = number
    return "\n".join(output)


def row_line_numbers(row: sqlite3.Row | dict) -> list[int]:
    text_lines = row["text"].split("\n")
    line_numbers = loads(row["line_numbers"], None)
    if not line_numbers:
        line_numbers = list(range(row["start_line"], row["start_line"] + len(text_lines)))
    return line_numbers


def numbered_window(row: sqlite3.Row | dict, focus_lines: set[int] | None, max_tokens: int, context: int = 12) -> tuple[str, bool]:
    full = numbered_text(row)
    if count_tokens(full) <= max_tokens:
        return full, False
    text_lines = row["text"].split("\n")
    line_numbers = row_line_numbers(row)
    keep: set[int] = set()
    for index in range(min(3, len(line_numbers))):
        keep.add(index)
    focus_indexes = [index for index, number in enumerate(line_numbers) if focus_lines and number in focus_lines]
    if not focus_indexes:
        focus_indexes = [3]
    for focus in focus_indexes:
        for index in range(max(focus - context, 0), min(focus + context + 1, len(line_numbers))):
            keep.add(index)
    width = len(str(max(line_numbers))) if line_numbers else 3
    output = []
    previous = None
    used = 0
    shown = 0
    for index in sorted(keep):
        rendered = f"{line_numbers[index]:>{width}} | {text_lines[index]}"
        cost = count_tokens(rendered) + 1
        if used + cost > max_tokens:
            break
        if previous is not None and index != previous + 1:
            output.append(" " * width + " | ...")
        output.append(rendered)
        used += cost
        shown += 1
        previous = index
    output.append(f"# truncated: showing {shown} of {len(line_numbers)} lines; call read_chunk for the full text")
    return "\n".join(output), True
