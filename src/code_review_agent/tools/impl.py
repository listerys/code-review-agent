from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field

from ..index.headers import numbered_text, render_header
from ..index.files import SECRET_FILE_PATTERNS, matches_any
from ..index.view import change_state
from ..retrieval.graph import GraphIndex, resolve_symbol_rows
from ..retrieval.models import SearchFilters, chunk_ref, parse_chunk_ref
from ..retrieval.pack import load_summaries
from ..util import loads
from .executor import ToolError, ToolExecutor, ensure_safe_glob
from .registry import ToolRegistry, ToolSpec

Version = Literal["current", "base", "head"]
MAX_CANDIDATES = 12


class GetDiffArgs(BaseModel):
    path: str | None = Field(description="Limit to one changed file; null for the whole diff")
    offset: int | None = Field(description="Character offset for paging a long diff; null to start at the beginning")


class ReadFileArgs(BaseModel):
    path: str = Field(description="Repository-relative file path")
    start: int | None = Field(description="First line (1-based); null for 1")
    end: int | None = Field(description="Last line inclusive; null for start + 200")
    version: Version = Field(description="'current' = the branch head state, 'base' = before the change, 'head' = only files changed on the branch")


class ReadChunkArgs(BaseModel):
    ref: str = Field(description="Chunk reference such as c123 from retrieved context or search results")


class SymbolArgs(BaseModel):
    symbol: str = Field(description="Qualified name (pkg.module.Class.method), short name, or path:line")
    version: Version = Field(description="Which side of the change to inspect")


class FindSymbolArgs(BaseModel):
    name: str = Field(description="Qualified name, short name, or path:line")
    kind: str | None = Field(description="Optional kind filter: function, method, class_skeleton, model, test, url, template_block")
    version: Version = Field(description="Which side of the change to inspect")


class CallersArgs(BaseModel):
    symbol: str = Field(description="Qualified name, short name, or path:line")
    depth: int = Field(description="Hops to walk outward, 1-3")
    version: Version = Field(description="'base' shows callers before the change")


class ModelArgs(BaseModel):
    model: str = Field(description="Model class name or qualified name")


class TaskArgs(BaseModel):
    task: str = Field(description="Celery task function name or qualified name")


class TemplateArgs(BaseModel):
    template: str = Field(description="Template name as used in render(), e.g. emails/receipt.html")


class CacheArgs(BaseModel):
    key: str = Field(description="Cache key or key pattern, e.g. order_total:{}")


class GrepArgs(BaseModel):
    pattern: str = Field(description="Extended regular expression")
    path_glob: str | None = Field(description="Optional pathspec glob such as '*.py' or 'shop/**'")
    version: Literal["current", "base"] = Field(description="Search the branch head or the merge base")
    max_matches: int | None = Field(description="Cap on matches, default 50")


class SemanticSearchArgs(BaseModel):
    query: str = Field(description="Natural-language or code-shaped query")
    kinds: list[str] | None = Field(description="Optional chunk kinds to include")
    tags: list[str] | None = Field(description="Optional tags, any of which must match (e.g. money, tenant_scoped, celery_task)")
    path_prefix: str | None = Field(description="Optional path prefix")
    exclude_changed: bool = Field(description="Exclude files changed on this branch")
    k: int | None = Field(description="Number of results, default 8")


class GitLogArgs(BaseModel):
    path: str = Field(description="Repository-relative file path")
    n: int | None = Field(description="Number of commits, default 10")


class MemoryArgs(BaseModel):
    query: str = Field(description="What to look for in past bug fixes and confirmed findings")


def rows_for_symbol(executor: ToolExecutor, symbol: str, version: str) -> list:
    rows = resolve_symbol_rows(executor.session.view, symbol, version)
    if not rows:
        raise ToolError(f"no symbol matches '{symbol}' in version '{version}'; try find_symbol or semantic_search")
    return rows


def describe_row(row, extra_fields: dict | None = None) -> dict:
    described = {
        "ref": chunk_ref(row["id"]),
        "symbol": row["symbol"],
        "kind": row["kind"],
        "at": f"{row['path']}:{row['start_line']}-{row['end_line']}",
        "state": change_state(row),
    }
    if extra_fields:
        described.update(extra_fields)
    return described


def get_diff(executor: ToolExecutor, arguments: GetDiffArgs) -> str:
    diff = executor.session.diff
    if arguments.path:
        if diff.file(arguments.path) is None:
            raise ToolError(f"{arguments.path} is not changed on this branch; changed files: {[item.path for item in diff.files][:30]}")
        text = diff.text(arguments.path)
    else:
        sections = []
        for file_diff in diff.files:
            hunk_ids = ", ".join(hunk.id for hunk in file_diff.hunks)
            sections.append(f"# {file_diff.status} {file_diff.path}  hunks: {hunk_ids}\n{file_diff.text()}")
        text = "\n\n".join(sections)
    offset = arguments.offset or 0
    if offset:
        text = f"[diff from offset {offset}]\n" + text[offset:]
    return text


def read_file(executor: ToolExecutor, arguments: ReadFileArgs) -> str:
    path = executor.check_path(arguments.path)
    view_file = executor.session.view.file(path, arguments.version)
    if view_file is None:
        raise ToolError(f"{path} does not exist in version '{arguments.version}'")
    lines = executor.session.repo.read_text(view_file.blob_sha).split("\n")
    start = max(arguments.start or 1, 1)
    max_lines = executor.settings.read_file_max_lines
    end = arguments.end or start + 200
    end = min(end, len(lines), start + max_lines - 1)
    if start > len(lines):
        raise ToolError(f"{path} has only {len(lines)} lines")
    width = len(str(end))
    output = [f"# {path} ({arguments.version}, lines {start}-{end} of {len(lines)})"]
    for number in range(start, end + 1):
        output.append(f"{number:>{width}} | {lines[number - 1]}")
    if end < len(lines):
        output.append(f"# more: call read_file with start={end + 1}")
    return "\n".join(output)


def read_chunk(executor: ToolExecutor, arguments: ReadChunkArgs) -> str:
    row_id = parse_chunk_ref(arguments.ref)
    if row_id is None:
        raise ToolError("ref must look like c123")
    view = executor.session.view
    row = view.chunk_by_row_id(row_id)
    if row is None:
        raise ToolError(f"{arguments.ref} is not part of this review's index")
    derived = view.derived_tags([row["chunk_id"]])
    summaries = load_summaries(view, executor.session.settings.models.mini_model, [row["content_hash"]])
    parent_signature = None
    if row["parent_id"]:
        parents = view.chunks_for_ids([row["parent_id"]], "base" if row["superseded"] else "current")
        if parents:
            parent_signature = parents[0]["signature"]
    header = render_header(row, view.all_tags(row, derived), parent_signature, summaries.get(row["content_hash"]))
    parts = [header, numbered_text(row)]
    siblings = view.chunk_rows("c.path = ? AND c.blob_sha = ? AND c.symbol = ? AND c.id != ?", (row["path"], row["blob_sha"], row["symbol"], row["id"]))
    if siblings:
        parts.append("# other parts: " + ", ".join(chunk_ref(sibling["id"]) for sibling in siblings))
    return "\n".join(parts)


def find_symbol(executor: ToolExecutor, arguments: FindSymbolArgs) -> dict:
    rows = resolve_symbol_rows(executor.session.view, arguments.name, arguments.version, limit=40)
    if arguments.kind:
        rows = [row for row in rows if row["kind"] == arguments.kind]
    derived = executor.session.view.derived_tags([row["chunk_id"] for row in rows])
    candidates = []
    for row in rows[:MAX_CANDIDATES]:
        candidates.append(
            describe_row(
                row,
                {
                    "signature": (row["signature"] or "").split("\n")[0][:160],
                    "tags": executor.session.view.all_tags(row, derived)[:12],
                },
            )
        )
    return {"count": len(rows), "candidates": candidates}


def graph_listing(executor: ToolExecutor, symbol: str, relation: str, depth: int, version: str) -> dict:
    view = executor.session.view
    seeds = rows_for_symbol(executor, symbol, version)
    seed_ids = [row["chunk_id"] for row in seeds[:5]]
    related = GraphIndex(view).related(seed_ids, relation, max(1, min(depth, 3)), 80, version)
    rows = {row["chunk_id"]: row for row in view.chunks_for_ids([item.chunk_id for item in related], version)}
    listing = []
    for item in related:
        row = rows.get(item.chunk_id)
        if row is None:
            continue
        entry = describe_row(row, {"line": item.line, "edge": item.edge_kind, "confidence": item.confidence, "distance": item.distance})
        if item.extra:
            compact = {key: value for key, value in item.extra.items() if key in ("in_atomic", "in_on_commit", "method", "signals", "via", "tenant_filtered", "model")}
            if compact:
                entry["details"] = compact
        listing.append(entry)
    result = {"seeds": [describe_row(row) for row in seeds[:5]], "results": listing[: executor.settings.max_items]}
    missing = view.edge_rows(
        "kind IN ('calls', 'enqueues', 'imports') AND confidence = 'missing' AND dst_name LIKE ?",
        ("%" + symbol.rsplit(".", 1)[-1],),
        version,
    )
    if relation == "callers" and missing:
        result["references_to_missing_symbol"] = [
            {"from": edge["src_path"], "line": edge["line"], "target": edge["dst_name"]} for edge in missing[:10]
        ]
    if len(listing) > executor.settings.max_items:
        result["more"] = len(listing) - executor.settings.max_items
    return result


def callers(executor: ToolExecutor, arguments: CallersArgs) -> dict:
    return graph_listing(executor, arguments.symbol, "callers", arguments.depth, arguments.version)


def callees(executor: ToolExecutor, arguments: SymbolArgs) -> dict:
    return graph_listing(executor, arguments.symbol, "callees", 1, arguments.version)


def tests_for(executor: ToolExecutor, arguments: SymbolArgs) -> dict:
    return graph_listing(executor, arguments.symbol, "tests", 1, arguments.version)


def route_for_view(executor: ToolExecutor, arguments: SymbolArgs) -> dict:
    view = executor.session.view
    seeds = rows_for_symbol(executor, arguments.symbol, arguments.version)
    seed_ids = [row["chunk_id"] for row in seeds[:5]]
    routes = []
    for route in view.url_routes_for(seed_ids):
        url_rows = view.chunks_for_ids([route["url_chunk_id"]])
        location = f"{url_rows[0]['path']}:{url_rows[0]['start_line']}" if url_rows else None
        routes.append({"route": "/" + route["route"], "name": route["name"], "defined_at": location, "url_ref": chunk_ref(url_rows[0]["id"]) if url_rows else None})
    derived = view.derived_tags(seed_ids)
    gates = []
    for row in seeds[:5]:
        tags = view.all_tags(row, derived)
        gates.append({"symbol": row["symbol"], "auth_gate": "auth_gate" in tags, "no_auth_gate": "no_auth_gate" in tags, "csrf_exempt": "csrf_exempt" in tags})
    return {"view": [describe_row(row) for row in seeds[:5]], "routes": routes, "access": gates}


def signal_receivers(executor: ToolExecutor, arguments: ModelArgs) -> dict:
    view = executor.session.view
    models = [row for row in resolve_symbol_rows(view, arguments.model) if row["kind"] in ("model", "class_skeleton")]
    model_ids = [row["chunk_id"] for row in models]
    edges = view.edges_to(model_ids, ["receives"]) if model_ids else []
    short = arguments.model.rsplit(".", 1)[-1]
    edges += view.edge_rows("kind = 'receives' AND dst_chunk_id IS NULL AND (dst_name = '*' OR dst_name = ? OR dst_name LIKE ?)", (short, "%." + short))
    rows = {row["chunk_id"]: row for row in view.chunks_for_ids([edge["src_chunk_id"] for edge in edges])}
    receivers = []
    for edge in edges:
        row = rows.get(edge["src_chunk_id"])
        if row is None:
            continue
        extra = loads(edge["extra"], {})
        receivers.append(
            describe_row(
                row,
                {"signals": extra.get("signals"), "sender": edge["dst_name"], "via": extra.get("via"), "line": edge["line"], "confidence": edge["confidence"]},
            )
        )
    senders = []
    for edge in view.edge_rows("kind = 'sends_signal'"):
        senders.append({"from": edge["src_path"], "line": edge["line"], "signal": edge["dst_name"]})
    return {"model": [describe_row(row) for row in models[:3]], "receivers": receivers, "custom_signal_senders": senders[:20]}


def task_call_sites(executor: ToolExecutor, arguments: TaskArgs) -> dict:
    view = executor.session.view
    tasks = rows_for_symbol(executor, arguments.task, "current")
    task_ids = [row["chunk_id"] for row in tasks[:5]]
    edges = view.edges_to(task_ids, ["enqueues"])
    rows = {row["chunk_id"]: row for row in view.chunks_for_ids([edge["src_chunk_id"] for edge in edges])}
    sites = []
    for edge in edges:
        row = rows.get(edge["src_chunk_id"])
        if row is None:
            continue
        extra = loads(edge["extra"], {})
        sites.append(
            describe_row(
                row,
                {
                    "line": edge["line"],
                    "method": extra.get("method"),
                    "inside_atomic": bool(extra.get("in_atomic")),
                    "deferred_with_on_commit": bool(extra.get("in_on_commit")),
                    "confidence": edge["confidence"],
                },
            )
        )
    return {"task": [describe_row(row, {"decorators": loads(row["decorators"], [])}) for row in tasks[:5]], "call_sites": sites}


def template_usages(executor: ToolExecutor, arguments: TemplateArgs) -> dict:
    view = executor.session.view
    name = arguments.template.removeprefix("tpl:")
    edges = view.edge_rows("kind IN ('renders', 'email_template', 'extends', 'includes') AND dst_name = ?", (f"tpl:{name}",))
    rows = {row["chunk_id"]: row for row in view.chunks_for_ids([edge["src_chunk_id"] for edge in edges])}
    usages = []
    for edge in edges:
        row = rows.get(edge["src_chunk_id"])
        if row is not None:
            usages.append(describe_row(row, {"line": edge["line"], "edge": edge["kind"]}))
    template_rows = view.chunk_rows("c.symbol = ? OR c.symbol LIKE ?", (f"tpl:{name}", f"tpl:{name}#%"))
    variables = []
    for row in template_rows:
        variables.extend(loads(row["template_vars"], []))
    return {"template": [describe_row(row) for row in template_rows[:6]], "variables": sorted(set(variables))[:60], "used_by": usages}


def cache_key_usage(executor: ToolExecutor, arguments: CacheArgs) -> dict:
    view = executor.session.view
    pattern = arguments.key.strip()
    like = "cache:%" + pattern.replace("{}", "%").replace("*", "%") + "%"
    edges = view.edge_rows("kind = 'cache' AND dst_name LIKE ?", (like,))
    rows = {row["chunk_id"]: row for row in view.chunks_for_ids([edge["src_chunk_id"] for edge in edges])}
    sites = []
    for edge in edges:
        row = rows.get(edge["src_chunk_id"])
        if row is None:
            continue
        extra = loads(edge["extra"], {})
        sites.append(describe_row(row, {"key": edge["dst_name"].removeprefix("cache:"), "op": extra.get("op"), "line": edge["line"], "key_helper": extra.get("key_helper")}))
    return {"sites": sites}


NESTED_QUANTIFIER = re.compile(r"\([^)]*[+*][^)]*\)\s*[+*{]")


def secret_exclusion_pathspecs() -> list[str]:
    pathspecs = []
    for pattern in SECRET_FILE_PATTERNS:
        pathspecs.append(f":(exclude,glob)**/{pattern}")
    return pathspecs


def grep(executor: ToolExecutor, arguments: GrepArgs) -> dict:
    session = executor.session
    limit = min(arguments.max_matches or 50, 200)
    try:
        compiled = re.compile(arguments.pattern)
    except re.error as error:
        raise ToolError(f"invalid regex: {error}") from error
    positive = [ensure_safe_glob(arguments.path_glob)] if arguments.path_glob else []
    revision = session.base_sha if arguments.version == "base" else session.grep_revision()
    matches = []
    if revision is not None:
        pathspecs = (positive or ["."]) + secret_exclusion_pathspecs()
        found = session.repo.grep(revision, arguments.pattern, pathspecs, limit, timeout=executor.settings.default_timeout_seconds)
        for match in found:
            if matches_any(match.path, SECRET_FILE_PATTERNS):
                continue
            matches.append({"at": f"{match.path}:{match.line}", "text": match.text.strip()[:200]})
    else:
        if NESTED_QUANTIFIER.search(arguments.pattern):
            raise ToolError("patterns with nested quantifiers are not allowed here; simplify the regex")
        for path in session.view.all_paths("current"):
            if matches_any(path, SECRET_FILE_PATTERNS):
                continue
            if positive and not re.fullmatch(glob_to_regex(positive[0]), path):
                continue
            view_file = session.view.file(path)
            if view_file is None or not view_file.indexed:
                continue
            text = session.read_file_text(path)
            if text is None:
                continue
            for number, line in enumerate(text.split("\n"), start=1):
                if compiled.search(line[:500]):
                    matches.append({"at": f"{path}:{number}", "text": line.strip()[:200]})
                    if len(matches) >= limit:
                        break
            if len(matches) >= limit:
                break
    return {"matches": matches, "capped": len(matches) >= limit}


def glob_to_regex(pattern: str) -> str:
    escaped = re.escape(pattern).replace(r"\*\*", ".*").replace(r"\*", "[^/]*").replace(r"\?", ".")
    if "/" not in pattern:
        return r"(.*/)?" + escaped
    return escaped


def semantic_search(executor: ToolExecutor, arguments: SemanticSearchArgs) -> dict:
    session = executor.session
    filters = SearchFilters(
        kinds=arguments.kinds or None,
        tags_any=arguments.tags or None,
        path_prefix=arguments.path_prefix,
        exclude_changed=arguments.exclude_changed,
    )
    limit = min(arguments.k or 8, 20)
    ranked, _ = session.engine.search(arguments.query, filters, "hybrid", limit)
    rows = {row["id"]: row for row in session.view.chunks_for_row_ids([item.row_id for item in ranked])}
    summaries = load_summaries(session.view, session.settings.models.mini_model, [row["content_hash"] for row in rows.values()])
    derived = session.view.derived_tags([row["chunk_id"] for row in rows.values()])
    results = []
    for item in ranked:
        row = rows.get(item.row_id)
        if row is None:
            continue
        preview = summaries.get(row["content_hash"]) or (row["signature"] or row["text"].split("\n")[0])[:160]
        results.append(
            describe_row(
                row,
                {
                    "matched_by": item.sources,
                    "why": item.reasons[:2],
                    "preview": preview,
                    "tags": session.view.all_tags(row, derived)[:10],
                },
            )
        )
    return {"results": results, "dense_enabled": session.engine.dense_available(), "hint": "expand any result with read_chunk(ref)"}


def git_log(executor: ToolExecutor, arguments: GitLogArgs) -> dict:
    path = executor.check_path(arguments.path)
    session = executor.session
    revision = session.head_sha or session.base_sha
    commits = session.repo.log_for_path(revision, path, min(arguments.n or 10, 30))
    return {"path": path, "commits": [{"sha": commit.sha[:10], "date": commit.date, "subject": commit.subject[:160]} for commit in commits]}


def similar_past_fixes(executor: ToolExecutor, arguments: MemoryArgs) -> dict:
    rows = executor.session.engine.memory(arguments.query, 5)
    history = []
    for row in rows:
        history.append(
            {
                "ref": f"m{row['id']}",
                "source": row["source"],
                "date": row.get("date"),
                "subject": row["subject"][:160],
                "path": row.get("path"),
                "hunk": (row.get("hunk") or "")[:600],
            }
        )
    return {"note": "history is a hint for where to look, never evidence for a finding", "history": history}


def build_registry() -> ToolRegistry:
    registry = ToolRegistry()
    specs = [
        ToolSpec("get_diff", "Show the branch diff with hunk ids, optionally for one file.", GetDiffArgs, get_diff, text_output=True, max_chars=20000),
        ToolSpec("read_file", "Read numbered lines of a repository file at the branch head or merge base.", ReadFileArgs, read_file, text_output=True),
        ToolSpec("read_chunk", "Expand a retrieved chunk reference (c123) into its full numbered text with header.", ReadChunkArgs, read_chunk, text_output=True),
        ToolSpec("find_symbol", "Locate functions, classes, models, URL patterns, or templates by name.", FindSymbolArgs, find_symbol),
        ToolSpec("callers", "List call sites and other inbound references of a symbol from the static graph, with confidence.", CallersArgs, callers),
        ToolSpec("callees", "List what a symbol calls, queries, renders, or enqueues.", SymbolArgs, callees),
        ToolSpec("route_for_view", "Show URL routes that reach a view and whether it has an auth gate.", SymbolArgs, route_for_view),
        ToolSpec("signal_receivers", "List signal receivers registered for a model (post_save, pre_delete, custom signals).", ModelArgs, signal_receivers),
        ToolSpec("task_call_sites", "List where a Celery task is enqueued and whether each call is inside a transaction or deferred with on_commit.", TaskArgs, task_call_sites),
        ToolSpec("tests_for", "List tests that exercise a symbol directly, through its URL, through factories, or transitively.", SymbolArgs, tests_for),
        ToolSpec("template_usages", "Show which views or emails render a template and the variables it uses.", TemplateArgs, template_usages),
        ToolSpec("cache_key_usage", "List every read, write, and delete of a cache key pattern.", CacheArgs, cache_key_usage),
        ToolSpec("grep", "Regex search over the repository at the branch head or merge base.", GrepArgs, grep, timeout_seconds=20),
        ToolSpec("semantic_search", "Hybrid search (graph + BM25 + embeddings) for code related to a question; returns references to expand.", SemanticSearchArgs, semantic_search, timeout_seconds=45),
        ToolSpec("git_log", "Recent commits that touched a file.", GitLogArgs, git_log),
        ToolSpec("similar_past_fixes", "Search past bug-fix commits and confirmed review findings for similar changes (hints only).", MemoryArgs, similar_past_fixes, timeout_seconds=30),
    ]
    for spec in specs:
        registry.register(spec)
    return registry

