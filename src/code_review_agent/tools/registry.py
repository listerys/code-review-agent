from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from openai.lib._pydantic import to_strict_json_schema
from pydantic import BaseModel


@dataclass
class ToolSpec:
    name: str
    description: str
    arguments: type[BaseModel]
    handler: Callable[[Any, BaseModel], Any]
    timeout_seconds: float | None = None
    max_chars: int | None = None
    text_output: bool = False


def strip_schema_noise(schema: Any) -> Any:
    if isinstance(schema, dict):
        cleaned = {}
        for key, value in schema.items():
            if key in ("default", "title"):
                continue
            cleaned[key] = strip_schema_noise(value)
        return cleaned
    if isinstance(schema, list):
        return [strip_schema_noise(item) for item in schema]
    return schema


class ToolRegistry:
    def __init__(self):
        self.specs: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        self.specs[spec.name] = spec

    def get(self, name: str) -> ToolSpec | None:
        return self.specs.get(name)

    def names(self) -> list[str]:
        return sorted(self.specs)

    def openai_definitions(self, allowlist: list[str] | None = None) -> list[dict]:
        definitions = []
        for name in sorted(self.specs):
            if allowlist is not None and name not in allowlist:
                continue
            spec = self.specs[name]
            definitions.append(
                {
                    "type": "function",
                    "name": spec.name,
                    "description": spec.description,
                    "parameters": strip_schema_noise(to_strict_json_schema(spec.arguments)),
                    "strict": True,
                }
            )
        return definitions
