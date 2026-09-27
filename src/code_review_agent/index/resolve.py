from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..config import Settings
from ..util import dumps, loads
from .python_parser import READ_WRITE_QUERY_METHODS
from .store import Store
from .view import View

HEURISTIC_STOPLIST = {
    "save",
    "get",
    "set",
    "delete",
    "update",
    "create",
    "filter",
    "exclude",
    "all",
    "first",
    "last",
    "count",
    "clean",
    "full_clean",
    "__init__",
    "__str__",
    "__repr__",
    "render",
    "run",
    "handle",
    "process",
    "send",
    "close",
    "open",
    "read",
    "write",
    "items",
    "keys",
    "values",
    "append",
    "extend",
    "add",
    "remove",
    "pop",
    "format",
    "join",
    "split",
    "strip",
    "replace",
    "lower",
    "upper",
    "copy",
    "exists",
    "is_valid",
    "dispatch",
    "post",
    "put",
    "patch",
    "setUp",
    "tearDown",
    "execute",
    "call",
    "load",
    "dump",
    "dumps",
    "loads",
    "encode",
    "decode",
    "startswith",
    "endswith",
    "refresh_from_db",
    "as_view",
}
HEURISTIC_MAX_CANDIDATES = 3
AUTO_TENANT_NAMES = {"Tenant", "Organization", "Organisation", "Realm", "Workspace"}
AUTH_WRAPPERS = re.compile(r"(?i)(login_required|permission_required|staff_member_required|user_passes_test|_required$)")
PATH_CONVERTER = re.compile(r"<(?:[^:>]+:)?([^>]+)>")


@dataclass
class SymbolEntry:
    chunk_id: str
    kind: str
    symbol: str
    name: str
    path: str
    tags: set[str]
    extra: dict
    parent_id: str | None


@dataclass
class EdgeRow:
    src_chunk_id: str
    src_path: str
    dst_chunk_id: str | None
    dst_name: str
    kind: str
    confidence: str
    line: int
    extra: dict = field(default_factory=dict)


@dataclass
class Resolution:
    chunk_id: str | None
    confidence: str
    name: str


def route_to_regex(route: str, is_regex: bool) -> str:
    if is_regex:
        return route.lstrip("^").rstrip("$")
    escaped = ""
    position = 0
    for match in PATH_CONVERTER.finditer(route):
        escaped += re.escape(route[position : match.start()]) + "[^/]+"
        position = match.end()
    escaped += re.escape(route[position:])
    return escaped


class ViewResolver:
    def __init__(self, store: Store, view: View, settings: Settings):
        self.store = store
        self.view = view
        self.settings = settings
        self.symbols: dict[str, SymbolEntry] = {}
        self.by_chunk: dict[str, SymbolEntry] = {}
        self.methods_by_name: dict[str, list[str]] = {}
        self.module_of_path: dict[str, str] = {}
        self.path_of_module: dict[str, str] = {}
        self.imports: dict[str, dict[str, str]] = {}
        self.star_imports: dict[str, list[str]] = {}
        self.templates: dict[str, str] = {}
        self.url_by_name: dict[str, list[str]] = {}
        self.signal_defs: set[str] = set()
        self.task_names: dict[str, str] = {}
        self.models_by_name: dict[str, list[str]] = {}
        self.base_cache: dict[str, list[str]] = {}
        self.model_cache: dict[str, bool] = {}
        self.load()

    def load(self) -> None:
        rows = self.view.chunk_rows("c.part = 0")
        for row in rows:
            tags = set(loads(row["tags"], []))
            extra = loads(row["extra"], {})
            entry = SymbolEntry(
                chunk_id=row["chunk_id"],
                kind=row["kind"],
                symbol=row["symbol"],
                name=row["name"],
                path=row["path"],
                tags=tags,
                extra=extra,
                parent_id=row["parent_id"],
            )
            self.by_chunk[entry.chunk_id] = entry
            if entry.kind in ("module_header", "config") and entry.path.endswith(".py"):
                self.module_of_path[entry.path] = entry.symbol
                self.path_of_module.setdefault(entry.symbol, entry.path)
            if entry.symbol not in self.symbols or entry.kind in ("module_header", "config"):
                self.symbols[entry.symbol] = entry
            if entry.kind in ("method", "function", "test"):
                self.methods_by_name.setdefault(entry.name, []).append(entry.symbol)
            if entry.kind == "model":
                self.models_by_name.setdefault(entry.name, []).append(entry.symbol)
            if entry.kind == "template_block" and extra.get("block") in (None, "__header__"):
                template_name = extra.get("template_name")
                if template_name:
                    self.templates.setdefault(template_name, entry.chunk_id)
            if entry.kind == "url" and extra.get("name"):
                self.url_by_name.setdefault(extra["name"], []).append(entry.chunk_id)
                if extra.get("app_name"):
                    self.url_by_name.setdefault(f"{extra['app_name']}:{extra['name']}", []).append(entry.chunk_id)
            if "celery_task" in tags:
                if extra.get("task_name"):
                    self.task_names[extra["task_name"]] = entry.symbol
                self.task_names.setdefault(entry.symbol, entry.symbol)
            if entry.kind in ("module_header", "config"):
                for variable in extra.get("variables", []) or []:
                    qualname = f"{entry.symbol}.{variable}"
                    if qualname not in self.symbols:
                        self.symbols[qualname] = SymbolEntry(entry.chunk_id, "variable", qualname, variable, entry.path, set(), {}, None)
            for signal_name in extra.get("signal_defs", []) or []:
                qualname = f"{entry.symbol}.{signal_name}"
                self.signal_defs.add(qualname)
                self.symbols.setdefault(
                    qualname,
                    SymbolEntry(entry.chunk_id, "signal", qualname, signal_name, entry.path, {"signal_definition"}, {}, None),
                )

        module_rows = self.store.query(
            f"SELECT f.path, f.module FROM file_versions f JOIN {self.view.active_table} a "
            f"ON f.path = a.path AND f.blob_sha = a.blob_sha WHERE a.superseded = 0 AND f.module IS NOT NULL"
        )
        for row in module_rows:
            self.module_of_path.setdefault(row["path"], row["module"])
            self.path_of_module.setdefault(row["module"], row["path"])

        import_rows = self.store.query(
            f"SELECT i.path, i.alias, i.target, i.is_star FROM imports i "
            f"JOIN {self.view.active_table} a ON i.path = a.path AND i.blob_sha = a.blob_sha WHERE a.superseded = 0"
        )
        for row in import_rows:
            module = self.module_of_path.get(row["path"])
            if module is None:
                continue
            if row["is_star"]:
                self.star_imports.setdefault(module, []).append(row["target"])
            else:
                self.imports.setdefault(module, {}).setdefault(row["alias"], row["target"])

    def module_for_chunk(self, entry: SymbolEntry) -> str | None:
        return self.module_of_path.get(entry.path)

    def enclosing_class_of(self, entry: SymbolEntry) -> str | None:
        if entry.kind in ("class_skeleton", "model"):
            return entry.symbol
        if entry.parent_id and entry.parent_id in self.by_chunk:
            parent = self.by_chunk[entry.parent_id]
            if parent.kind in ("class_skeleton", "model"):
                return parent.symbol
        return None

    def is_missing_symbol(self, dotted: str) -> bool:
        parts = dotted.split(".")
        for size in range(len(parts) - 1, 0, -1):
            module = ".".join(parts[:size])
            if module not in self.path_of_module:
                continue
            name = parts[size]
            if f"{module}.{name}" in self.symbols:
                return False
            if name in self.imports.get(module, {}) or self.star_imports.get(module):
                return False
            if f"{module}.{name}" in self.path_of_module:
                return False
            return True
        return False

    def lookup(self, qualname: str, depth: int = 0) -> str | None:
        if qualname in self.symbols:
            return qualname
        if depth > 4:
            return None
        parts = qualname.split(".")
        for size in range(len(parts) - 1, 0, -1):
            prefix = ".".join(parts[:size])
            remainder = parts[size:]
            if prefix in self.path_of_module:
                module_imports = self.imports.get(prefix, {})
                if remainder[0] in module_imports:
                    forwarded = ".".join([module_imports[remainder[0]], *remainder[1:]])
                    return self.lookup(forwarded, depth + 1)
                for star_module in self.star_imports.get(prefix, []):
                    candidate = self.lookup(".".join([star_module, *remainder]), depth + 1)
                    if candidate:
                        return candidate
                return None
            entry = self.symbols.get(prefix)
            if entry is not None and entry.kind in ("class_skeleton", "model") and len(remainder) == 1:
                return self.find_in_class(prefix, remainder[0])
        return None

    def bases_of(self, class_qualname: str) -> list[str]:
        if class_qualname in self.base_cache:
            return self.base_cache[class_qualname]
        self.base_cache[class_qualname] = []
        entry = self.symbols.get(class_qualname)
        resolved: list[str] = []
        if entry is not None:
            module = self.module_for_chunk(entry) or ""
            for base in entry.extra.get("bases", []) or []:
                resolution = self.resolve(base, module, None, allow_heuristic=False)
                if resolution.chunk_id:
                    target = self.by_chunk[resolution.chunk_id]
                    if target.kind in ("class_skeleton", "model"):
                        resolved.append(target.symbol)
        self.base_cache[class_qualname] = resolved
        return resolved

    def find_in_class(self, class_qualname: str, name: str, depth: int = 0) -> str | None:
        candidate = f"{class_qualname}.{name}"
        if candidate in self.symbols:
            return candidate
        if depth > 6:
            return None
        for base in self.bases_of(class_qualname):
            found = self.find_in_class(base, name, depth + 1)
            if found:
                return found
        return None

    def is_model(self, qualname: str, depth: int = 0) -> bool:
        if qualname in self.model_cache:
            return self.model_cache[qualname]
        entry = self.symbols.get(qualname)
        result = False
        if entry is not None:
            if entry.kind == "model":
                result = True
            elif entry.kind == "class_skeleton" and depth < 6:
                for base in self.bases_of(qualname):
                    if self.is_model(base, depth + 1):
                        result = True
                        break
        self.model_cache[qualname] = result
        return result

    def heuristic(self, name: str) -> list[str]:
        if name in HEURISTIC_STOPLIST or name.startswith("__"):
            return []
        candidates = self.methods_by_name.get(name, [])
        if 1 <= len(candidates) <= HEURISTIC_MAX_CANDIDATES:
            return candidates
        return []

    def resolve(
        self,
        dotted: str,
        module: str,
        enclosing_class: str | None,
        allow_heuristic: bool = True,
        is_local: bool = False,
    ) -> Resolution:
        parts = dotted.split(".")
        head = parts[0]
        if head in ("self", "cls") and enclosing_class and len(parts) >= 2:
            if len(parts) == 2:
                found = self.find_in_class(enclosing_class, parts[1])
                if found:
                    return Resolution(self.symbols[found].chunk_id, "exact", found)
            if allow_heuristic:
                guesses = self.heuristic(parts[-1])
                if len(guesses) == 1:
                    return Resolution(self.symbols[guesses[0]].chunk_id, "heuristic", guesses[0])
            return Resolution(None, "unresolved", parts[-1])
        if head == "super()" and enclosing_class and len(parts) == 2:
            for base in self.bases_of(enclosing_class):
                found = self.find_in_class(base, parts[1])
                if found:
                    return Resolution(self.symbols[found].chunk_id, "exact", found)
            return Resolution(None, "unresolved", parts[1])
        if is_local:
            if allow_heuristic and len(parts) >= 2:
                guesses = self.heuristic(parts[-1])
                if len(guesses) == 1:
                    return Resolution(self.symbols[guesses[0]].chunk_id, "heuristic", guesses[0])
            return Resolution(None, "unresolved", dotted)

        candidates = []
        explicit = []
        local = f"{module}.{head}"
        if local in self.symbols:
            candidates.append(".".join([local, *parts[1:]]))
        module_imports = self.imports.get(module, {})
        if head in module_imports:
            imported = ".".join([module_imports[head], *parts[1:]])
            candidates.append(imported)
            explicit.append(imported)
        for star_module in self.star_imports.get(module, []):
            candidates.append(".".join([star_module, *parts]))
        if "." in dotted and head not in module_imports and local not in self.symbols:
            candidates.append(dotted)
            explicit.append(dotted)
        for candidate in candidates:
            found = self.lookup(candidate)
            if found:
                return Resolution(self.symbols[found].chunk_id, "exact", found)
        for candidate in explicit:
            if self.is_missing_symbol(candidate):
                return Resolution(None, "missing", candidate)
        if allow_heuristic and len(parts) >= 2 and head not in module_imports:
            guesses = self.heuristic(parts[-1])
            if len(guesses) == 1:
                return Resolution(self.symbols[guesses[0]].chunk_id, "heuristic", guesses[0])
        return Resolution(None, "unresolved", dotted)

    def tenant_models(self) -> set[str]:
        configured = set(self.settings.django.tenant_models)
        tenants = set()
        for entry in self.symbols.values():
            if entry.kind != "model":
                continue
            if entry.name in configured or entry.symbol in configured or entry.name in AUTO_TENANT_NAMES:
                tenants.add(entry.name)
            for field_info in entry.extra.get("fields", []) or []:
                related = (field_info.get("related") or "").split(".")[-1]
                if field_info["type"] in ("OneToOneField", "ForeignKey") and related == "Site":
                    tenants.add(entry.name)
        tenants.add("Site")
        return tenants

    def tenant_fields_by_model(self) -> dict[str, set[str]]:
        tenant_names = self.tenant_models()
        configured_fields = set(self.settings.django.tenant_fields)
        result: dict[str, set[str]] = {}
        for entry in self.symbols.values():
            if entry.kind != "model":
                continue
            names = set()
            for field_info in entry.extra.get("fields", []) or []:
                related = (field_info.get("related") or "").split(".")[-1]
                if field_info["type"] in ("ForeignKey", "OneToOneField", "ManyToManyField") and related in tenant_names:
                    names.add(field_info["name"])
                if field_info["name"] in configured_fields:
                    names.add(field_info["name"])
            if entry.name in tenant_names:
                names.update({"pk", "id"})
            if names:
                result[entry.symbol] = names
        return result


def resolve_view_edges(store: Store, view: View, resolver: ViewResolver) -> list[EdgeRow]:
    tenant_fields = resolver.tenant_fields_by_model()
    all_tenant_field_names = set()
    for names in tenant_fields.values():
        all_tenant_field_names.update(names - {"pk", "id"})

    where = "a.superseded = 0"
    parameters: tuple = ()
    ref_rows = store.query(
        f"SELECT r.* FROM refs r JOIN {view.active_table} a ON r.path = a.path AND r.blob_sha = a.blob_sha WHERE {where}",
        parameters,
    )
    edges: dict[tuple, EdgeRow] = {}

    def add_edge(src_entry: SymbolEntry, dst: Resolution, kind: str, line: int, extra: dict | None = None, src_override: SymbolEntry | None = None) -> None:
        source = src_override or src_entry
        dst_name = dst.name
        operation = (extra or {}).get("op") if kind == "cache" else None
        key = (source.chunk_id, dst.chunk_id or dst_name, kind, operation)
        existing = edges.get(key)
        if existing is not None:
            lines = existing.extra.setdefault("lines", [existing.line])
            if len(lines) < 6 and line not in lines:
                lines.append(line)
            for flag in ("in_atomic", "in_on_commit"):
                if extra and extra.get(flag) is False and existing.extra.get(flag):
                    existing.extra[flag] = False
            return
        edges[key] = EdgeRow(source.chunk_id, source.path, dst.chunk_id, dst_name, kind, dst.confidence, line, dict(extra or {}))

    for ref in ref_rows:
        src = resolver.by_chunk.get(ref["src_chunk_id"])
        if src is None:
            continue
        module = resolver.module_for_chunk(src) or ""
        enclosing_class = resolver.enclosing_class_of(src)
        kind = ref["kind"]
        target = ref["target"]
        line = ref["line"]
        extra = loads(ref["extra"], {})

        if kind == "import":
            found = resolver.lookup(target)
            if found:
                add_edge(src, Resolution(resolver.symbols[found].chunk_id, "exact", found), "imports", line)
            elif resolver.is_missing_symbol(target):
                add_edge(src, Resolution(None, "missing", target), "imports", line)
        elif kind == "call":
            resolution = resolver.resolve(target, module, enclosing_class, is_local=bool(extra.get("local")))
            if resolution.chunk_id:
                dst_entry = resolver.by_chunk[resolution.chunk_id]
                edge_kind = "instantiates" if dst_entry.kind in ("class_skeleton", "model") else "calls"
                add_edge(src, resolution, edge_kind, line, extra)
            elif resolution.confidence == "missing":
                add_edge(src, resolution, "calls", line, extra)
        elif kind == "decorator":
            resolution = resolver.resolve(target, module, None, allow_heuristic=False)
            if resolution.chunk_id:
                add_edge(src, resolution, "decorated_by", line)
        elif kind == "base":
            resolution = resolver.resolve(target, module, None, allow_heuristic=False)
            if resolution.chunk_id or resolution.confidence == "missing":
                add_edge(src, resolution, "inherits", line)
        elif kind == "query":
            resolution = resolver.resolve(target, module, None, allow_heuristic=False)
            if resolution.chunk_id:
                model_symbol = resolver.by_chunk[resolution.chunk_id].symbol
                fields_for_model = tenant_fields.get(model_symbol, set())
                keyword_roots = set()
                for keyword in extra.get("kwargs", []):
                    for segment in keyword.split("__"):
                        keyword_roots.add(segment)
                tenant_filtered = bool(keyword_roots & (fields_for_model | all_tenant_field_names))
                extra["tenant_owned"] = bool(fields_for_model)
                extra["tenant_filtered"] = tenant_filtered
                add_edge(src, resolution, "queries", line, extra)
        elif kind == "task_call":
            resolution = resolver.resolve(target, module, enclosing_class)
            if resolution.chunk_id:
                dst_entry = resolver.by_chunk[resolution.chunk_id]
                confidence = "exact" if "celery_task" in dst_entry.tags else "heuristic"
                add_edge(src, Resolution(resolution.chunk_id, confidence, resolution.name), "enqueues", line, extra)
            elif resolution.confidence == "missing":
                add_edge(src, resolution, "enqueues", line, extra)
        elif kind == "send_task_name":
            qualname = resolver.task_names.get(target)
            if qualname and qualname in resolver.symbols:
                add_edge(src, Resolution(resolver.symbols[qualname].chunk_id, "exact", qualname), "enqueues", line, extra)
        elif kind == "receiver":
            if target == "*":
                add_edge(src, Resolution(None, "exact", "*"), "receives", line, extra)
            else:
                resolution = resolver.resolve(target, module, None, allow_heuristic=False)
                add_edge(src, resolution, "receives", line, extra)
        elif kind == "signal_connect":
            handler = resolver.resolve(target, module, enclosing_class, allow_heuristic=False)
            if handler.chunk_id is None:
                continue
            handler_entry = resolver.by_chunk[handler.chunk_id]
            sender_text = extra.get("sender") or "*"
            if sender_text == "*":
                sender = Resolution(None, "exact", "*")
            else:
                sender = resolver.resolve(sender_text, module, enclosing_class, allow_heuristic=False)
            edge_extra = {"signals": [extra.get("signal")], "via": "connect", "registered_in": src.chunk_id}
            add_edge(src, sender, "receives", line, edge_extra, src_override=handler_entry)
        elif kind == "signal_send":
            resolution = resolver.resolve(target, module, enclosing_class, allow_heuristic=False)
            if resolution.name in resolver.signal_defs:
                add_edge(src, resolution, "sends_signal", line)
        elif kind == "template":
            template_chunk = resolver.templates.get(target)
            edge_kind = "email_template" if "sends_email" in src.tags else "renders"
            if template_chunk:
                add_edge(src, Resolution(template_chunk, "exact", f"tpl:{target}"), edge_kind, line, extra)
            else:
                add_edge(src, Resolution(None, "unresolved", f"tpl:{target}"), edge_kind, line, extra)
        elif kind in ("template_extends", "template_include"):
            template_chunk = resolver.templates.get(target)
            edge_kind = "extends" if kind == "template_extends" else "includes"
            add_edge(src, Resolution(template_chunk, "exact" if template_chunk else "unresolved", f"tpl:{target}"), edge_kind, line)
        elif kind == "url_view":
            resolution = resolver.resolve(target, module, None, allow_heuristic=False)
            add_edge(src, resolution, "routes", line, extra)
        elif kind == "url_include":
            found = resolver.lookup(target)
            if found:
                add_edge(src, Resolution(resolver.symbols[found].chunk_id, "exact", found), "includes_urls", line)
        elif kind == "url_name":
            name = target
            candidates = resolver.url_by_name.get(name) or resolver.url_by_name.get(name.split(":")[-1], [])
            for url_chunk in candidates[:3]:
                add_edge(src, Resolution(url_chunk, "exact" if len(candidates) == 1 else "heuristic", name), "links_to", line)
            if not candidates:
                add_edge(src, Resolution(None, "unresolved", f"url:{name}"), "links_to", line)
        elif kind == "http_path":
            add_edge(src, Resolution(None, "unresolved", f"path:{target}"), "requests", line, extra)
        elif kind == "cache_op":
            key = target
            if target.startswith("<call:"):
                helper = resolver.resolve(target[len("<call:") : -1], module, enclosing_class, allow_heuristic=False)
                if helper.chunk_id and resolver.by_chunk[helper.chunk_id].extra.get("returns_pattern"):
                    key = resolver.by_chunk[helper.chunk_id].extra["returns_pattern"]
                    extra["key_helper"] = helper.name
            add_edge(src, Resolution(None, "exact", f"cache:{key}"), "cache", line, extra)
        elif kind == "admin_register":
            model = resolver.resolve(target, module, None, allow_heuristic=False)
            admin_source = src
            if extra.get("admin_class"):
                admin_resolution = resolver.resolve(extra["admin_class"], module, None, allow_heuristic=False)
                if admin_resolution.chunk_id:
                    admin_source = resolver.by_chunk[admin_resolution.chunk_id]
            if model.chunk_id:
                add_edge(src, model, "admin_for", line, src_override=admin_source)
        elif kind == "meta_model":
            model = resolver.resolve(target, module, None, allow_heuristic=False)
            if model.chunk_id is None and not target.startswith("self"):
                same_name = resolver.models_by_name.get(target.split(".")[-1], [])
                if len(same_name) == 1:
                    model = Resolution(resolver.symbols[same_name[0]].chunk_id, "heuristic", same_name[0])
            if model.chunk_id:
                add_edge(src, model, "model_ref", line, extra)
        elif kind == "router_register":
            viewset = resolver.resolve(target, module, None, allow_heuristic=False)
            if viewset.chunk_id:
                add_edge(src, viewset, "registers_viewset", line, extra)
        elif kind == "reference":
            resolution = resolver.resolve(target, module, enclosing_class, allow_heuristic=False)
            if resolution.chunk_id:
                add_edge(src, resolution, "references", line, extra)

    resolve_http_paths(view, edges)
    return list(edges.values())


def resolve_http_paths(view: View, edges: dict) -> None:
    routes = view.all_url_routes()
    compiled = []
    for route in routes:
        try:
            compiled.append((re.compile("^" + route["route"] + "$"), route))
        except re.error:
            continue
    for edge in edges.values():
        if edge.kind != "requests" or not edge.dst_name.startswith("path:"):
            continue
        request_path = edge.dst_name[len("path:") :].split("?")[0].lstrip("/")
        for pattern, route in compiled:
            if pattern.match(request_path) and route["view_chunk_id"]:
                edge.dst_chunk_id = route["view_chunk_id"]
                edge.confidence = "exact"
                break


def compose_url_routes(store: Store, view: View, resolver: ViewResolver) -> list[tuple[str, str | None, str, str | None, str | None]]:
    url_rows = view.chunk_rows("c.kind = 'url'")
    patterns_by_module: dict[str, list] = {}
    for row in url_rows:
        module = resolver.module_of_path.get(row["path"])
        if module is None:
            continue
        patterns_by_module.setdefault(module, []).append(row)
    for module_rows in patterns_by_module.values():
        module_rows.sort(key=lambda item: item["start_line"])

    root_modules = []
    root_rows = store.query(
        f"SELECT DISTINCT r.target FROM refs r JOIN {view.active_table} a ON r.path = a.path AND r.blob_sha = a.blob_sha "
        f"WHERE a.superseded = 0 AND r.kind = 'root_urlconf'"
    )
    for row in root_rows:
        if row["target"] in patterns_by_module:
            root_modules.append(row["target"])
    included = set()
    for module_rows in patterns_by_module.values():
        for row in module_rows:
            include_ref = loads(row["extra"], {}).get("include_ref")
            if include_ref:
                included.add(include_ref)
    if not root_modules:
        for module in patterns_by_module:
            if module not in included:
                root_modules.append(module)

    router_routes: dict[tuple[str, str], list[tuple[str, str | None]]] = {}
    router_rows = store.query(
        f"SELECT r.path, r.target, r.extra FROM refs r JOIN {view.active_table} a ON r.path = a.path AND r.blob_sha = a.blob_sha "
        f"WHERE a.superseded = 0 AND r.kind = 'router_register'"
    )
    for row in router_rows:
        module = resolver.module_of_path.get(row["path"])
        extra = loads(row["extra"], {})
        viewset = resolver.resolve(row["target"], module or "", None, allow_heuristic=False)
        router_routes.setdefault((module or "", extra.get("router") or ""), []).append((extra.get("route") or "", viewset.chunk_id))

    results: list[tuple[str, str | None, str, str | None, str | None]] = []
    seen: set[tuple[str, str, str]] = set()
    visited: set[tuple[str, str, str]] = set()

    def compose(module: str, prefix: str, namespace: str | None, depth: int) -> None:
        state = (module, prefix, namespace or "")
        if state in visited or depth > 8 or len(results) > 20000:
            return
        visited.add(state)
        for row in patterns_by_module.get(module, []):
            extra = loads(row["extra"], {})
            route_fragment = route_to_regex(extra.get("route") or "", bool(extra.get("is_regex")))
            full_route = prefix + route_fragment
            include_ref = extra.get("include_ref")
            if include_ref:
                if include_ref in patterns_by_module:
                    child_namespace = extra.get("namespace")
                    if child_namespace is None:
                        first_rows = patterns_by_module[include_ref]
                        child_namespace = loads(first_rows[0]["extra"], {}).get("app_name") if first_rows else None
                    combined = ":".join(part for part in (namespace, child_namespace) if part) or None
                    compose(include_ref, full_route, combined, depth + 1)
                elif include_ref.endswith(".urls"):
                    router_name = include_ref[: -len(".urls")]
                    for route, viewset_chunk in router_routes.get((module, router_name), []):
                        key = (row["chunk_id"], viewset_chunk or "", full_route + route)
                        if key not in seen:
                            seen.add(key)
                            results.append((row["chunk_id"], viewset_chunk, full_route + route_to_regex(route, False), None, None))
                continue
            view_ref = extra.get("view_ref")
            if not view_ref:
                continue
            resolution = resolver.resolve(view_ref, module, None, allow_heuristic=False)
            name = extra.get("name")
            full_name = f"{namespace}:{name}" if namespace and name else name
            key = (row["chunk_id"], resolution.chunk_id or view_ref, full_route)
            if key in seen:
                continue
            seen.add(key)
            results.append((row["chunk_id"], resolution.chunk_id, full_route, full_name, view_ref))

    for module in root_modules:
        compose(module, "", None, 0)
    return results


def derive_tags(view: View, resolver: ViewResolver) -> list[tuple[str, str]]:
    tags: set[tuple[str, str]] = set()
    edge_rows = view.edge_rows("1=1")
    auth_tagged = set()
    for row in view.chunk_rows("c.part = 0 AND c.tags LIKE '%auth_gate%'"):
        auth_tagged.add(row["chunk_id"])
    routed_views: dict[str, list[str]] = {}
    for route in view.all_url_routes():
        if route["view_chunk_id"]:
            tags.add((route["view_chunk_id"], "view"))
            routed_views.setdefault(route["view_chunk_id"], []).append(route["url_chunk_id"])
    url_wrappers: dict[str, list[str]] = {}
    for row in view.chunk_rows("c.kind = 'url'"):
        url_wrappers[row["chunk_id"]] = loads(row["extra"], {}).get("wrappers") or []

    inherited_auth = set()
    for row in edge_rows:
        if row["kind"] == "inherits" and row["dst_chunk_id"] in auth_tagged:
            inherited_auth.add(row["src_chunk_id"])

    for view_chunk, url_chunks in routed_views.items():
        wrapped_with_auth = False
        for url_chunk in url_chunks:
            for wrapper in url_wrappers.get(url_chunk, []):
                if AUTH_WRAPPERS.search(wrapper):
                    wrapped_with_auth = True
        if view_chunk not in auth_tagged and view_chunk not in inherited_auth and not wrapped_with_auth:
            tags.add((view_chunk, "no_auth_gate"))

    for entry in resolver.by_chunk.values():
        if entry.kind == "class_skeleton" and resolver.is_model(entry.symbol):
            tags.add((entry.chunk_id, "model"))

    for row in edge_rows:
        kind = row["kind"]
        src = row["src_chunk_id"]
        dst_name = row["dst_name"]
        short = dst_name.rsplit(".", 1)[-1]
        extra = loads(row["extra"], {})
        if kind == "receives":
            tags.add((src, "signal_receiver"))
        elif kind == "admin_for":
            tags.add((src, "admin"))
        elif kind == "enqueues":
            tags.add((src, f"calls_task:{short}"))
            if extra.get("in_atomic") and not extra.get("in_on_commit"):
                tags.add((src, "task_in_atomic_without_on_commit"))
        elif kind in ("renders", "email_template"):
            tags.add((src, f"renders:{dst_name.removeprefix('tpl:')}"))
        elif kind == "queries":
            tags.add((src, f"queries:{short}"))
            if extra.get("tenant_filtered"):
                tags.add((src, "tenant_scoped"))
            methods = set(extra.get("methods", []))
            if extra.get("tenant_owned") and not extra.get("tenant_filtered") and methods & READ_WRITE_QUERY_METHODS:
                tags.add((src, "unscoped_query"))
        elif kind == "sends_signal":
            tags.add((src, f"sends_signal:{short}"))
        elif kind == "cache":
            operation = extra.get("op", "")
            tags.add((src, "cache_write" if operation not in ("get", "get_many", "has_key") else "cache_read"))
        elif kind in ("calls", "enqueues", "imports") and row["confidence"] == "missing":
            tags.add((src, "references_missing_symbol"))
    return sorted(tags)


def write_resolution(store: Store, view: View, settings: Settings) -> dict:
    resolver = ViewResolver(store, view, settings)
    with store.lock:
        store.connection.execute("DELETE FROM url_routes WHERE view_key = ?", (view.view_key,))
    routes = compose_url_routes(store, view, resolver)
    with store.lock:
        store.connection.executemany(
            "INSERT INTO url_routes(view_key, url_chunk_id, view_chunk_id, route, name, view_ref) VALUES (?, ?, ?, ?, ?, ?)",
            [(view.view_key, url_chunk, view_chunk, route, name, view_ref) for url_chunk, view_chunk, route, name, view_ref in routes],
        )
        store.connection.commit()

    edges = resolve_view_edges(store, view, resolver)
    with store.lock:
        store.connection.execute("DELETE FROM edges WHERE view_key = ?", (view.view_key,))
        store.connection.executemany(
            "INSERT INTO edges(view_key, src_chunk_id, src_path, dst_chunk_id, dst_name, kind, confidence, line, extra) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (view.view_key, edge.src_chunk_id, edge.src_path, edge.dst_chunk_id, edge.dst_name, edge.kind, edge.confidence, edge.line, dumps(edge.extra))
                for edge in edges
            ],
        )
        store.connection.commit()

    tags = derive_tags(view, resolver)
    with store.lock:
        store.connection.execute("DELETE FROM chunk_tags WHERE view_key = ?", (view.view_key,))
        store.connection.executemany(
            "INSERT OR IGNORE INTO chunk_tags(view_key, chunk_id, tag) VALUES (?, ?, ?)",
            [(view.view_key, chunk_id, tag) for chunk_id, tag in tags],
        )
        store.connection.execute("UPDATE views SET resolved = 1 WHERE view_key = ?", (view.view_key,))
        store.connection.commit()
    return {"edges": len(edges), "routes": len(routes), "derived_tags": len(tags)}
