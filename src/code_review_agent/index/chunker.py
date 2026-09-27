from __future__ import annotations

import re
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from ..util import IDENTIFIER, content_hash, count_tokens, split_identifier, stable_chunk_id
from .files import is_settings_module, is_test_path
from .python_parser import ParsedPython, SymbolInfo, parse_python
from .templates import parse_template

HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
FENCE = re.compile(r"^\s*(```|~~~)")
JS_SYMBOL = re.compile(
    r"(?:function\s+([A-Za-z_$][\w$]*)|class\s+([A-Za-z_$][\w$]*)|(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?(?:function|\())"
)
GENERIC_SYMBOL = re.compile(r"^\s*(?:pub\s+)?(?:func|fn|def|class|interface|struct|type|module)\s+([A-Za-z_][\w]*)")


class Chunk(BaseModel):
    chunk_id: str
    content_hash: str
    path: str
    blob_sha: str
    start_line: int
    end_line: int
    line_numbers: list[int] | None = None
    kind: str
    symbol: str
    name: str
    parent_id: str | None = None
    decorators: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    template_vars: list[str] = Field(default_factory=list)
    signature: str | None = None
    text: str
    token_count: int
    part: int = 0
    extra: dict = Field(default_factory=dict)


class ChunkRef(BaseModel):
    src_chunk_id: str
    kind: str
    target: str
    line: int
    extra: dict = Field(default_factory=dict)


class ImportRow(BaseModel):
    alias: str
    target: str
    is_star: bool
    line: int


class FileIndex(BaseModel):
    path: str
    blob_sha: str
    language: str
    module: str | None = None
    chunks: list[Chunk] = Field(default_factory=list)
    refs: list[ChunkRef] = Field(default_factory=list)
    imports: list[ImportRow] = Field(default_factory=list)
    migration: dict | None = None
    syntax_errors: int = 0


@dataclass
class ChunkLimits:
    max_tokens: int = 1500
    target_tokens: int = 600
    small_template_tokens: int = 600
    money_pattern: str = ""


@dataclass
class ChunkBuilder:
    path: str
    blob_sha: str
    lines: list[str]
    limits: ChunkLimits
    chunks: list[Chunk] = field(default_factory=list)
    seen_ids: dict[str, int] = field(default_factory=dict)

    def line_text(self, numbers: list[int]) -> str:
        output = []
        for number in numbers:
            if 1 <= number <= len(self.lines):
                output.append(self.lines[number - 1])
        return "\n".join(output)

    def unique_id(self, symbol: str, kind: str) -> str:
        base_id = stable_chunk_id(self.path, symbol, kind)
        count = self.seen_ids.get(base_id, 0) + 1
        self.seen_ids[base_id] = count
        if count == 1:
            return base_id
        return f"{base_id}.{count}"

    def add(
        self,
        kind: str,
        symbol: str,
        name: str,
        numbers: list[int],
        part: int = 0,
        chunk_id: str | None = None,
        **fields,
    ) -> Chunk:
        text = self.line_text(numbers)
        contiguous = numbers == list(range(numbers[0], numbers[-1] + 1)) if numbers else True
        identifier = chunk_id or self.unique_id(symbol, kind)
        chunk = Chunk(
            chunk_id=identifier,
            content_hash=content_hash(text),
            path=self.path,
            blob_sha=self.blob_sha,
            start_line=numbers[0] if numbers else 1,
            end_line=numbers[-1] if numbers else 1,
            line_numbers=None if contiguous else numbers,
            kind=kind,
            symbol=symbol,
            name=name,
            text=text,
            token_count=count_tokens(text),
            part=part,
            **fields,
        )
        self.chunks.append(chunk)
        return chunk


def line_tokens(lines: list[str], numbers: list[int]) -> list[int]:
    return [count_tokens(lines[number - 1]) + 1 if 1 <= number <= len(lines) else 1 for number in numbers]


def group_lines_by_budget(lines: list[str], numbers: list[int], max_tokens: int) -> list[list[int]]:
    groups: list[list[int]] = []
    current: list[int] = []
    current_tokens = 0
    for number, tokens in zip(numbers, line_tokens(lines, numbers), strict=True):
        if current and current_tokens + tokens > max_tokens:
            groups.append(current)
            current = []
            current_tokens = 0
        current.append(number)
        current_tokens += tokens
    if current:
        groups.append(current)
    return groups


def windows_by_blank_lines(lines: list[str], start: int, end: int, target_tokens: int, max_tokens: int) -> list[list[int]]:
    paragraphs: list[list[int]] = []
    current: list[int] = []
    for number in range(start, end + 1):
        text = lines[number - 1] if number <= len(lines) else ""
        if not text.strip():
            if current:
                paragraphs.append(current)
                current = []
            continue
        current.append(number)
    if current:
        paragraphs.append(current)

    windows: list[list[int]] = []
    window: list[int] = []
    window_tokens = 0
    for paragraph in paragraphs:
        paragraph_tokens = sum(line_tokens(lines, paragraph))
        if paragraph_tokens > max_tokens:
            if window:
                windows.append(window)
                window = []
                window_tokens = 0
            windows.extend(group_lines_by_budget(lines, paragraph, max_tokens))
            continue
        if window and window_tokens + paragraph_tokens > target_tokens:
            windows.append(window)
            window = []
            window_tokens = 0
        window.extend(paragraph)
        window_tokens += paragraph_tokens
    if window:
        windows.append(window)
    return windows


MONEY_TAG_KINDS = {"function", "method", "test", "model", "class_skeleton", "template_block", "file_block"}


def money_signal(text: str, pattern: re.Pattern) -> bool:
    hits = 0
    distinct = set()
    for match in IDENTIFIER.finditer(text):
        for word in split_identifier(match.group(0)):
            if pattern.fullmatch(word):
                hits += 1
                distinct.add(word)
    if "Decimal(" in text or "quantize(" in text:
        distinct.add("decimal")
        hits += 1
    return len(distinct) >= 2 or hits >= 4


def apply_money_tags(chunks: list[Chunk], limits: ChunkLimits) -> None:
    if not limits.money_pattern:
        return
    pattern = re.compile(limits.money_pattern)
    for chunk in chunks:
        if chunk.kind not in MONEY_TAG_KINDS:
            continue
        is_money_model = False
        for field_info in chunk.extra.get("fields", []):
            if field_info["type"] in ("DecimalField", "MoneyField") or pattern.fullmatch(field_info["name"].split("_")[-1]):
                is_money_model = True
        if is_money_model or money_signal(chunk.text, pattern):
            if "money" not in chunk.tags:
                chunk.tags.append("money")


def function_parts(symbol: SymbolInfo, lines: list[str], limits: ChunkLimits) -> list[list[int]]:
    all_numbers = list(range(symbol.start_line, symbol.end_line + 1))
    total_tokens = sum(line_tokens(lines, all_numbers))
    if total_tokens <= limits.max_tokens or not symbol.body_statements:
        return [all_numbers]
    signature = list(range(symbol.signature_lines[0], symbol.signature_lines[1] + 1))
    signature_tokens = sum(line_tokens(lines, signature))
    budget = max(limits.max_tokens - signature_tokens, 200)

    units: list[list[int]] = []
    for statement_start, statement_end in symbol.body_statements:
        statement_numbers = list(range(statement_start, statement_end + 1))
        if sum(line_tokens(lines, statement_numbers)) > budget:
            units.extend(group_lines_by_budget(lines, statement_numbers, budget))
        else:
            units.append(statement_numbers)

    pieces: list[list[int]] = []
    current: list[int] = []
    current_tokens = 0
    for unit in units:
        unit_tokens = sum(line_tokens(lines, unit))
        if current and current_tokens + unit_tokens > budget:
            pieces.append(current)
            current = []
            current_tokens = 0
        current.extend(unit)
        current_tokens += unit_tokens
    if current:
        pieces.append(current)

    parts = []
    for piece in pieces:
        numbers = sorted(set(signature) | set(piece))
        parts.append(numbers)
    return parts


def symbol_chunk_kind(symbol: SymbolInfo) -> str:
    if symbol.kind == "class":
        return "class_skeleton"
    return symbol.kind


def chunk_python(parsed: ParsedPython, path: str, blob_sha: str, source: bytes, limits: ChunkLimits, is_migration: bool) -> FileIndex:
    lines = source.decode("utf-8", "replace").split("\n")
    builder = ChunkBuilder(path=path, blob_sha=blob_sha, lines=lines, limits=limits)
    index = FileIndex(path=path, blob_sha=blob_sha, language="python", module=parsed.module, syntax_errors=parsed.syntax_errors)
    symbol_to_chunk: dict[str, str] = {}
    is_test_file = is_test_path(path)

    if parsed.header_lines and not is_migration:
        header_kind = "config" if is_settings_module(path) else "module_header"
        header_groups = group_lines_by_budget(lines, parsed.header_lines, limits.max_tokens)
        first_header_id = None
        module_tags = sorted(parsed.module_tags | ({"test"} if is_test_file else set()))
        for part_number, numbers in enumerate(header_groups):
            chunk_id = None
            if part_number > 0 and first_header_id:
                chunk_id = f"{first_header_id}.p{part_number + 1}"
            chunk = builder.add(
                header_kind,
                parsed.module,
                parsed.module.rsplit(".", 1)[-1],
                numbers,
                part=part_number,
                chunk_id=chunk_id,
                tags=list(module_tags),
                extra={"app_name": parsed.app_name, "signal_defs": parsed.signal_defs, "variables": parsed.module_variables[:200]}
                if part_number == 0
                else {},
            )
            if first_header_id is None:
                first_header_id = chunk.chunk_id
        symbol_to_chunk[parsed.module] = first_header_id

    ordered_symbols = sorted(parsed.symbols, key=lambda item: (item.start_line, -item.end_line))
    for symbol in ordered_symbols:
        if is_migration and "data_migration" not in symbol.tags:
            continue
        kind = symbol_chunk_kind(symbol)
        parent_id = symbol_to_chunk.get(symbol.parent) if symbol.parent else None
        signature = builder.line_text(list(range(symbol.signature_lines[0], symbol.signature_lines[1] + 1))).strip()
        tags = sorted(symbol.tags)
        if kind in ("class_skeleton", "model"):
            skeleton = symbol.skeleton_lines or list(range(symbol.start_line, symbol.end_line + 1))
            groups = group_lines_by_budget(lines, skeleton, limits.max_tokens)
            extra = {
                "bases": symbol.bases,
                "fields": [
                    {"name": item.name, "type": item.field_type, "related": item.related, "line": item.line, "options": item.options}
                    for item in symbol.fields
                ],
                "managers": symbol.managers,
                "meta": symbol.meta,
                "span": [symbol.start_line, symbol.end_line],
            }
        else:
            groups = function_parts(symbol, lines, limits)
            extra = {
                "is_async": symbol.is_async,
                "tx": symbol.tx,
                "task_name": symbol.task_name,
                "returns_pattern": symbol.returns_pattern,
                "span": [symbol.start_line, symbol.end_line],
            }
        first_id = None
        for part_number, numbers in enumerate(groups):
            chunk_id = f"{first_id}.p{part_number + 1}" if part_number > 0 and first_id else None
            chunk = builder.add(
                kind,
                symbol.qualname,
                symbol.name,
                numbers,
                part=part_number,
                chunk_id=chunk_id,
                parent_id=parent_id,
                decorators=symbol.decorators,
                tags=tags,
                signature=signature,
                extra=extra if part_number == 0 else {"span": extra["span"]},
            )
            if first_id is None:
                first_id = chunk.chunk_id
        symbol_to_chunk.setdefault(symbol.qualname, first_id)

    for pattern in parsed.url_patterns:
        url_symbol = pattern.symbol
        numbers = list(range(pattern.start_line, pattern.end_line + 1))
        chunk = builder.add(
            "url",
            url_symbol,
            pattern.name or pattern.route or f"pattern{pattern.index}",
            numbers,
            tags=["url"],
            extra={
                "route": pattern.route,
                "name": pattern.name,
                "view_ref": pattern.view_ref,
                "include_ref": pattern.include_ref,
                "namespace": pattern.namespace,
                "is_regex": pattern.is_regex,
                "app_name": parsed.app_name,
                "wrappers": pattern.wrappers,
                "function": pattern.function,
            },
        )
        symbol_to_chunk[url_symbol] = chunk.chunk_id

    fallback_chunk = symbol_to_chunk.get(parsed.module)
    for raw in parsed.refs:
        source_chunk = symbol_to_chunk.get(raw.src)
        if source_chunk is None:
            owner = raw.src
            while "." in owner and source_chunk is None:
                owner = owner.rsplit(".", 1)[0]
                source_chunk = symbol_to_chunk.get(owner)
        if source_chunk is None:
            source_chunk = fallback_chunk
        if source_chunk is None:
            continue
        index.refs.append(ChunkRef(src_chunk_id=source_chunk, kind=raw.kind, target=raw.target, line=raw.line, extra=raw.extra))

    for entry in parsed.imports:
        index.imports.append(ImportRow(alias=entry.alias, target=entry.target, is_star=entry.is_star, line=entry.line))
    index.migration = parsed.migration
    apply_money_tags(builder.chunks, limits)
    index.chunks = builder.chunks
    return index


def chunk_template(path: str, blob_sha: str, source: bytes, template_name: str, limits: ChunkLimits) -> FileIndex:
    parsed = parse_template(path, template_name, source)
    lines = source.decode("utf-8", "replace").split("\n")
    builder = ChunkBuilder(path=path, blob_sha=blob_sha, lines=lines, limits=limits)
    index = FileIndex(path=path, blob_sha=blob_sha, language="django_template", module=None)
    base_tags = ["template"] + (["email"] if parsed.is_email else [])
    file_symbol = f"tpl:{template_name}"
    all_numbers = list(range(1, len(lines) + 1))
    total_tokens = count_tokens(source.decode("utf-8", "replace"))
    base_extra = {"template_name": template_name, "extends": parsed.extends, "includes": parsed.includes}

    if total_tokens <= limits.small_template_tokens or not parsed.blocks:
        groups = [all_numbers] if total_tokens <= limits.max_tokens else windows_by_blank_lines(
            lines, 1, len(lines), limits.target_tokens, limits.max_tokens
        )
        first_id = None
        for part_number, numbers in enumerate(groups):
            chunk_id = f"{first_id}.p{part_number + 1}" if part_number > 0 and first_id else None
            chunk = builder.add(
                "template_block",
                file_symbol,
                template_name,
                numbers,
                part=part_number,
                chunk_id=chunk_id,
                tags=list(base_tags),
                template_vars=parsed.variables if part_number == 0 else [],
                extra=dict(base_extra, block=None),
            )
            if first_id is None:
                first_id = chunk.chunk_id
        canonical = first_id
    else:
        inside_blocks = set()
        for block in parsed.blocks:
            inside_blocks.update(range(block.start_line, block.end_line + 1))
        header_numbers = [number for number in all_numbers if number not in inside_blocks and lines[number - 1].strip()]
        if not header_numbers:
            header_numbers = [1]
        header = builder.add(
            "template_block",
            file_symbol,
            template_name,
            header_numbers[:400],
            tags=list(base_tags),
            extra=dict(base_extra, block="__header__"),
        )
        canonical = header.chunk_id
        for block in parsed.blocks:
            block_numbers = list(range(block.start_line, block.end_line + 1))
            block_tokens = sum(line_tokens(lines, block_numbers))
            groups = [block_numbers] if block_tokens <= limits.max_tokens else windows_by_blank_lines(
                lines, block.start_line, block.end_line, limits.target_tokens, limits.max_tokens
            )
            first_id = None
            for part_number, numbers in enumerate(groups):
                chunk_id = f"{first_id}.p{part_number + 1}" if part_number > 0 and first_id else None
                chunk = builder.add(
                    "template_block",
                    f"{file_symbol}#{block.block_name}",
                    block.block_name or template_name,
                    numbers,
                    part=part_number,
                    chunk_id=chunk_id,
                    parent_id=canonical,
                    tags=list(base_tags),
                    template_vars=block.variables if part_number == 0 else [],
                    extra=dict(base_extra, block=block.block_name),
                )
                if first_id is None:
                    first_id = chunk.chunk_id

    for extended in parsed.extends:
        index.refs.append(ChunkRef(src_chunk_id=canonical, kind="template_extends", target=extended, line=1))
    for included in parsed.includes:
        index.refs.append(ChunkRef(src_chunk_id=canonical, kind="template_include", target=included, line=1))
    for url_name, line in parsed.url_names:
        owner = canonical
        for chunk in builder.chunks:
            if chunk.start_line <= line <= chunk.end_line and chunk.chunk_id != canonical:
                owner = chunk.chunk_id
        index.refs.append(ChunkRef(src_chunk_id=owner, kind="url_name", target=url_name, line=line))
    apply_money_tags(builder.chunks, limits)
    index.chunks = builder.chunks
    return index


def slugify(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", text.strip().lower()).strip("-")
    return slug[:60] or "section"


def chunk_markdown(path: str, blob_sha: str, source: bytes, limits: ChunkLimits) -> FileIndex:
    lines = source.decode("utf-8", "replace").split("\n")
    builder = ChunkBuilder(path=path, blob_sha=blob_sha, lines=lines, limits=limits)
    index = FileIndex(path=path, blob_sha=blob_sha, language="markdown")
    sections: list[tuple[list[str], int, int]] = []
    heading_stack: list[tuple[int, str]] = []
    current_start = 1
    current_path: list[str] = []
    in_fence = False
    for number, line in enumerate(lines, start=1):
        if FENCE.match(line):
            in_fence = not in_fence
        match = None if in_fence else HEADING.match(line)
        if match:
            if number > current_start:
                sections.append((list(current_path), current_start, number - 1))
            level = len(match.group(1))
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, match.group(2)))
            current_path = [title for _, title in heading_stack]
            current_start = number
    if current_start <= len(lines):
        sections.append((list(current_path), current_start, len(lines)))

    for heading_path, start, end in sections:
        section_numbers = [number for number in range(start, end + 1) if lines[number - 1].strip()]
        if not section_numbers:
            continue
        title = " > ".join(heading_path) if heading_path else path
        symbol = f"doc:{path}#{slugify(heading_path[-1]) if heading_path else 'top'}"
        tokens = sum(line_tokens(lines, section_numbers))
        groups = [section_numbers] if tokens <= limits.max_tokens else windows_by_blank_lines(
            lines, start, end, limits.target_tokens, limits.max_tokens
        )
        for part_number, numbers in enumerate(groups):
            builder.add(
                "doc",
                symbol if part_number == 0 else f"{symbol}.p{part_number + 1}",
                title,
                numbers,
                part=part_number,
                tags=["doc"],
                extra={"heading_path": heading_path},
            )
    index.chunks = builder.chunks
    return index


def chunk_windows(path: str, blob_sha: str, source: bytes, limits: ChunkLimits, language: str, kind: str) -> FileIndex:
    lines = source.decode("utf-8", "replace").split("\n")
    builder = ChunkBuilder(path=path, blob_sha=blob_sha, lines=lines, limits=limits)
    index = FileIndex(path=path, blob_sha=blob_sha, language=language)
    windows = windows_by_blank_lines(lines, 1, len(lines), limits.target_tokens, limits.max_tokens)
    prefix = "cfg" if kind == "config" else "file"
    for numbers in windows:
        text = builder.line_text(numbers)
        name = f"{path}#L{numbers[0]}"
        found = JS_SYMBOL.search(text) if language in ("javascript", "typescript") else GENERIC_SYMBOL.search(text)
        if found:
            name = next_group(found) or name
        builder.add(kind, f"{prefix}:{path}#L{numbers[0]}", name, numbers, tags=[kind])
    apply_money_tags(builder.chunks, limits)
    index.chunks = builder.chunks
    return index


def next_group(match: re.Match) -> str | None:
    for value in match.groups():
        if value:
            return value
    return None


def index_file(
    path: str,
    blob_sha: str,
    data: bytes,
    handler: str,
    language: str,
    module: str | None,
    template_name: str | None,
    limits: ChunkLimits,
) -> FileIndex:
    if handler in ("python", "migration"):
        is_migration = handler == "migration"
        parsed = parse_python(path, data, module or path, is_test_path(path), is_migration)
        return chunk_python(parsed, path, blob_sha, data, limits, is_migration)
    if handler == "template":
        return chunk_template(path, blob_sha, data, template_name or path, limits)
    if handler == "doc":
        return chunk_markdown(path, blob_sha, data, limits)
    if handler in ("config", "fixture"):
        return chunk_windows(path, blob_sha, data, limits, language, "config")
    return chunk_windows(path, blob_sha, data, limits, language, "file_block")
