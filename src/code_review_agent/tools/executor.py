from __future__ import annotations

import json
import threading
import time
from pathlib import PurePosixPath

from pydantic import BaseModel, Field, ValidationError

from ..config import ToolSettings
from ..index.files import SECRET_FILE_PATTERNS, matches_any
from .registry import ToolRegistry


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict = Field(default_factory=dict)


class ToolResult(BaseModel):
    call_id: str
    name: str
    ok: bool
    content: str
    truncated: bool = False
    duration_ms: int = 0
    error: str | None = None


class ToolError(Exception):
    pass


def normalize_repo_path(raw: str) -> str:
    if not isinstance(raw, str) or not raw.strip():
        raise ToolError("path must be a non-empty string")
    if "\x00" in raw or "\\" in raw:
        raise ToolError("path contains forbidden characters")
    cleaned = raw.strip()
    while cleaned.startswith("./"):
        cleaned = cleaned[2:]
    pure = PurePosixPath(cleaned)
    if pure.is_absolute():
        raise ToolError("absolute paths are not allowed; use a repository-relative path")
    if any(part == ".." for part in pure.parts):
        raise ToolError("path traversal is not allowed")
    return str(pure)


def ensure_allowed_path(path: str, known_paths: set[str]) -> str:
    normalized = normalize_repo_path(path)
    if matches_any(normalized, SECRET_FILE_PATTERNS):
        raise ToolError("reading secret-like files is not allowed")
    if normalized not in known_paths:
        raise ToolError(f"{normalized} is not a file in this revision")
    return normalized


def ensure_safe_glob(pattern: str) -> str:
    if "\x00" in pattern or pattern.startswith("/") or ".." in PurePosixPath(pattern).parts:
        raise ToolError("path_glob must be repository-relative without '..'")
    return pattern


def truncate_output(text: str, max_chars: int) -> tuple[str, bool]:
    if len(text) <= max_chars:
        return text, False
    cut = text[:max_chars]
    last_newline = cut.rfind("\n")
    if last_newline > max_chars * 0.7:
        cut = cut[:last_newline]
    note = f"\n[truncated: showing {len(cut)} of {len(text)} characters; narrow the query or use offset/start to page]"
    return cut + note, True


class ToolExecutor:
    def __init__(self, session, registry: ToolRegistry, settings: ToolSettings, allowlist: list[str] | None = None):
        self.session = session
        self.registry = registry
        self.settings = settings
        self.allowlist = allowlist
        self.known_paths: set[str] | None = None

    def paths(self) -> set[str]:
        if self.known_paths is None:
            known = set()
            for view_file in self.session.view.load_files().values():
                known.add(view_file.path)
            self.known_paths = known
        return self.known_paths

    def check_path(self, path: str) -> str:
        return ensure_allowed_path(path, self.paths())

    def run(self, call: ToolCall) -> ToolResult:
        started = time.monotonic()
        spec = self.registry.get(call.name)
        if spec is None or (self.allowlist is not None and call.name not in self.allowlist):
            return self.failure(call, started, f"tool '{call.name}' is not available to this agent")
        try:
            arguments = spec.arguments.model_validate(call.arguments)
        except ValidationError as error:
            return self.failure(call, started, f"invalid arguments: {error.errors(include_url=False)}")
        timeout = spec.timeout_seconds or self.settings.default_timeout_seconds
        outcome: dict = {}

        def target() -> None:
            try:
                outcome["value"] = spec.handler(self, arguments)
            except Exception as error:
                outcome["error"] = error

        worker = threading.Thread(target=target, name=f"tool-{call.name}", daemon=True)
        worker.start()
        worker.join(timeout)
        if worker.is_alive():
            return self.failure(call, started, f"timed out after {timeout:.0f}s")
        if "error" in outcome:
            error = outcome["error"]
            if isinstance(error, ToolError):
                return self.failure(call, started, str(error))
            return self.failure(call, started, f"{type(error).__name__}: {error}")
        output = outcome.get("value")
        if spec.text_output and isinstance(output, str):
            text = output
        else:
            text = json.dumps(output, separators=(",", ":"), ensure_ascii=False, default=str)
        limit = spec.max_chars or self.settings.max_output_chars
        text, truncated = truncate_output(text, limit)
        elapsed = int((time.monotonic() - started) * 1000)
        return ToolResult(call_id=call.id, name=call.name, ok=True, content=text, truncated=truncated, duration_ms=elapsed)

    def failure(self, call: ToolCall, started: float, message: str) -> ToolResult:
        elapsed = int((time.monotonic() - started) * 1000)
        return ToolResult(call_id=call.id, name=call.name, ok=False, content=f"error: {message}", duration_ms=elapsed, error=message)
