from __future__ import annotations

import hashlib
import json
import re
from functools import lru_cache

import tiktoken

CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", "surrogatepass")).hexdigest()


def normalize_code(text: str) -> str:
    lines = text.replace("\r\n", "\n").split("\n")
    stripped = [line.rstrip() for line in lines]
    return "\n".join(stripped).strip("\n")


def content_hash(text: str) -> str:
    return sha1_text(normalize_code(text))


def stable_chunk_id(path: str, qualname: str, kind: str) -> str:
    return sha1_text(f"{path}\x00{qualname}\x00{kind}")[:20]


def split_identifier(word: str) -> list[str]:
    parts = []
    for piece in word.split("_"):
        if not piece:
            continue
        for sub in CAMEL_BOUNDARY.split(piece):
            if sub:
                parts.append(sub.lower())
    return parts


def expand_identifiers(text: str) -> str:
    extra_words = []
    for match in IDENTIFIER.finditer(text):
        word = match.group(0)
        if "_" in word or CAMEL_BOUNDARY.search(word):
            pieces = split_identifier(word)
            if len(pieces) > 1:
                extra_words.append(" ".join(pieces))
    if not extra_words:
        return text
    return text + "\n" + " ".join(extra_words)


@lru_cache(maxsize=1)
def token_encoder():
    return tiktoken.get_encoding("o200k_base")


def count_tokens(text: str) -> int:
    if not text:
        return 0
    return len(token_encoder().encode(text, disallowed_special=()))


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    encoder = token_encoder()
    tokens = encoder.encode(text, disallowed_special=())
    if len(tokens) <= max_tokens:
        return text
    return encoder.decode(tokens[:max_tokens])


def dumps(value) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False, default=str)


def loads(text: str | None, default=None):
    if not text:
        return default
    return json.loads(text)
