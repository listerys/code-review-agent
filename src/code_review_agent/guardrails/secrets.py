from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass

TOKEN_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("aws_access_key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("github_token", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{50,})\b")),
    ("anthropic_key", re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{20,}")),
    ("openai_key", re.compile(r"\bsk-(?:proj-|svcacct-|admin-)?[A-Za-z0-9_\-]{20,}")),
    ("slack_token", re.compile(r"\bxox[abposr]-[A-Za-z0-9-]{10,}")),
    ("stripe_key", re.compile(r"\b(?:sk|rk|pk)_(?:live|test)_[A-Za-z0-9]{16,}")),
    ("google_api_key", re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}")),
    ("url_credentials", re.compile(r"(?<=://)[^\s:/@'\"]+:[^\s@/'\"]{3,}(?=@)")),
]

SECRET_NAME = (
    r"[A-Za-z0-9_\.\-]*(?:secret|passwd|password|pwd|token|api[_\-]?key|apikey|access[_\-]?key|private[_\-]?key|"
    r"client[_\-]?secret|auth[_\-]?key|signing[_\-]?key|credentials?)[A-Za-z0-9_\.\-]*"
)

QUOTED_ASSIGNMENT = re.compile(
    rf"""(?ix)
    (?P<name>{SECRET_NAME})
    (?P<sep>["']?\s*[:=]\s*[bru]?)
    (?P<quote>["'])
    (?P<value>(?:(?!(?P=quote))[^\n]){{6,}}?)
    (?P=quote)
    """
)

UNQUOTED_ASSIGNMENT = re.compile(
    rf"""(?ix)
    ^(?P<prefix>\s*(?:export\s+)?)
    (?P<name>{SECRET_NAME})
    (?P<sep>\s*[:=]\s*)
    (?P<value>[^\s"'#,;()\[\]{{}}]{{8,}})
    \s*$
    """
)

PRIVATE_KEY_BEGIN = re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY(?: BLOCK)?-----")
PRIVATE_KEY_END = re.compile(r"-----END [A-Z0-9 ]*PRIVATE KEY(?: BLOCK)?-----")
KEY_BODY_LINE = re.compile(r"^\s*(?:[A-Za-z0-9+/]{8,}={0,2}|=[A-Za-z0-9+/]{2,8}|[A-Za-z][A-Za-z\-]*: [^()]*|)\s*$")

PLACEHOLDER_VALUES = re.compile(
    r"(?i)^(?:x+|\*+|changeme|change_me|password|secret|example|dummy|test|testing|placeholder|your[_\-].*|<.*>|\$\{.*\}|\{\{.*\}\}|%\(.*\)s|none|null|todo)$"
)
CODE_LIKE_VALUE = re.compile(r"^(?:os\.|env\(|getenv|settings\.|self\.|config\.|request\.)|^[a-z_][a-z0-9_]*(?:\.[a-z_][a-z0-9_]*)+$|^[a-z_]+$")


@dataclass
class ScrubResult:
    text: str
    redactions: list[str]


def shannon_entropy(value: str) -> float:
    if not value:
        return 0.0
    counts: dict[str, int] = {}
    for character in value:
        counts[character] = counts.get(character, 0) + 1
    entropy = 0.0
    for count in counts.values():
        probability = count / len(value)
        entropy -= probability * math.log2(probability)
    return entropy


def character_classes(value: str) -> int:
    classes = 0
    if any(character.islower() for character in value):
        classes += 1
    if any(character.isupper() for character in value):
        classes += 1
    if any(character.isdigit() for character in value):
        classes += 1
    if any(not character.isalnum() for character in value):
        classes += 1
    return classes


def marker(kind: str, secret: str) -> str:
    digest = hashlib.sha1(secret.encode("utf-8", "replace")).hexdigest()[:8]
    return f"«REDACTED:{kind}:{digest}»"


def looks_like_secret_value(value: str) -> bool:
    stripped = value.strip()
    if len(stripped) < 8 or any(character.isspace() for character in stripped):
        return False
    if PLACEHOLDER_VALUES.match(stripped) or CODE_LIKE_VALUE.match(stripped):
        return False
    return shannon_entropy(stripped) >= 3.0 or character_classes(stripped) >= 3


def scrub_line(line: str, redactions: list[str]) -> str:
    for kind, pattern in TOKEN_PATTERNS:
        def replace_token(match: re.Match, kind=kind) -> str:
            redactions.append(kind)
            return marker(kind, match.group(0))

        line = pattern.sub(replace_token, line)

    def replace_quoted(match: re.Match) -> str:
        value = match.group("value")
        if "«REDACTED" in value or not looks_like_secret_value(value):
            return match.group(0)
        redactions.append("assigned_secret")
        return f"{match.group('name')}{match.group('sep')}{match.group('quote')}{marker('assigned_secret', value)}{match.group('quote')}"

    line = QUOTED_ASSIGNMENT.sub(replace_quoted, line)

    unquoted = UNQUOTED_ASSIGNMENT.match(line)
    if unquoted and "«REDACTED" not in unquoted.group("value") and looks_like_secret_value(unquoted.group("value")):
        redactions.append("assigned_secret")
        line = f"{unquoted.group('prefix')}{unquoted.group('name')}{unquoted.group('sep')}{marker('assigned_secret', unquoted.group('value'))}"
    return line


def scrub_text(text: str) -> ScrubResult:
    if not text:
        return ScrubResult(text=text, redactions=[])
    redactions: list[str] = []
    output_lines = []
    inside_private_key = False
    for line in text.split("\n"):
        if inside_private_key:
            end_match = PRIVATE_KEY_END.search(line)
            if end_match:
                inside_private_key = False
                output_lines.append(end_match.group(0))
                continue
            if KEY_BODY_LINE.match(line):
                output_lines.append("«REDACTED:private_key_body»" if line.strip() else line)
                continue
            inside_private_key = False
        begin_match = PRIVATE_KEY_BEGIN.search(line)
        if begin_match:
            redactions.append("private_key")
            if not PRIVATE_KEY_END.search(line):
                inside_private_key = True
            output_lines.append(scrub_line(line, redactions))
            continue
        output_lines.append(scrub_line(line, redactions))
    return ScrubResult(text="\n".join(output_lines), redactions=redactions)


def scrub(text: str) -> str:
    return scrub_text(text).text
