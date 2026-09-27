from __future__ import annotations

import re
from dataclasses import dataclass, field

TAG_PATTERN = re.compile(r"{%-?\s*(\w+)\s*(.*?)\s*-?%}", re.S)
VARIABLE_PATTERN = re.compile(r"{{-?\s*(.*?)\s*-?}}", re.S)
DOTTED_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z0-9_]+)*")
QUOTED = re.compile(r"""^\s*["']([^"']+)["']""")
TEMPLATE_KEYWORDS = {
    "and",
    "or",
    "not",
    "in",
    "is",
    "as",
    "with",
    "only",
    "True",
    "False",
    "None",
    "forloop",
    "block",
    "csrf_token",
    "request",
    "empty",
    "reversed",
}


@dataclass
class TemplateRegion:
    kind: str
    block_name: str | None
    start_line: int
    end_line: int
    variables: list[str] = field(default_factory=list)


@dataclass
class ParsedTemplate:
    path: str
    template_name: str
    line_count: int
    extends: list[str] = field(default_factory=list)
    includes: list[str] = field(default_factory=list)
    url_names: list[tuple[str, int]] = field(default_factory=list)
    blocks: list[TemplateRegion] = field(default_factory=list)
    variables: list[str] = field(default_factory=list)
    is_email: bool = False


def line_number_at(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def expression_variables(expression: str) -> list[str]:
    head = expression.split("|")[0]
    found = []
    for match in DOTTED_IDENTIFIER.finditer(head):
        word = match.group(0)
        if word.split(".")[0] in TEMPLATE_KEYWORDS:
            continue
        found.append(word)
    return found


def tag_variables(tag_name: str, arguments: str) -> list[str]:
    if tag_name in ("if", "elif"):
        return expression_variables(arguments)
    if tag_name == "for":
        parts = arguments.split(" in ", 1)
        if len(parts) == 2:
            return expression_variables(parts[1])
    if tag_name == "with":
        variables = []
        for assignment in arguments.split():
            if "=" in assignment:
                variables.extend(expression_variables(assignment.split("=", 1)[1]))
        return variables
    return []


def variables_between(text: str, start_offset: int, end_offset: int) -> list[str]:
    region = text[start_offset:end_offset]
    names: list[str] = []
    for match in VARIABLE_PATTERN.finditer(region):
        names.extend(expression_variables(match.group(1)))
    for match in TAG_PATTERN.finditer(region):
        names.extend(tag_variables(match.group(1), match.group(2)))
    unique = []
    seen = set()
    for name in names:
        if name not in seen:
            seen.add(name)
            unique.append(name)
    return unique[:60]


def parse_template(path: str, template_name: str, source: bytes) -> ParsedTemplate:
    text = source.decode("utf-8", "replace")
    line_count = text.count("\n") + 1
    parsed = ParsedTemplate(path=path, template_name=template_name, line_count=line_count)
    lower_path = path.lower()
    parsed.is_email = "email" in lower_path or "/mail" in lower_path or "mails/" in lower_path

    open_blocks: list[tuple[str, int, int]] = []
    for match in TAG_PATTERN.finditer(text):
        tag_name = match.group(1)
        arguments = match.group(2)
        if tag_name == "extends":
            quoted = QUOTED.match(arguments)
            if quoted:
                parsed.extends.append(quoted.group(1))
        elif tag_name == "include":
            quoted = QUOTED.match(arguments)
            if quoted:
                parsed.includes.append(quoted.group(1))
        elif tag_name == "url":
            quoted = QUOTED.match(arguments)
            if quoted:
                parsed.url_names.append((quoted.group(1), line_number_at(text, match.start())))
        elif tag_name == "block":
            block_name = arguments.split()[0] if arguments.split() else "block"
            open_blocks.append((block_name, match.start(), len(open_blocks)))
        elif tag_name == "endblock" and open_blocks:
            block_name, start_offset, depth = open_blocks.pop()
            if depth == 0:
                parsed.blocks.append(
                    TemplateRegion(
                        kind="block",
                        block_name=block_name,
                        start_line=line_number_at(text, start_offset),
                        end_line=line_number_at(text, match.end()),
                        variables=variables_between(text, start_offset, match.end()),
                    )
                )
    parsed.blocks.sort(key=lambda region: region.start_line)
    parsed.variables = variables_between(text, 0, len(text))
    return parsed
