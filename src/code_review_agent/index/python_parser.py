from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache

import tree_sitter_python
from tree_sitter import Language, Node, Parser

QUERYSET_METHODS = {
    "filter",
    "get",
    "exclude",
    "all",
    "create",
    "update",
    "delete",
    "get_or_create",
    "update_or_create",
    "bulk_create",
    "bulk_update",
    "values",
    "values_list",
    "annotate",
    "aggregate",
    "select_for_update",
    "select_related",
    "prefetch_related",
    "raw",
    "extra",
    "count",
    "first",
    "last",
    "exists",
    "in_bulk",
    "iterator",
    "order_by",
    "distinct",
    "only",
    "defer",
    "none",
    "latest",
    "earliest",
    "using",
}

READ_WRITE_QUERY_METHODS = QUERYSET_METHODS - {"create", "bulk_create", "none", "raw"}

TASK_METHODS = {"delay", "apply_async", "s", "si", "signature", "map", "starmap", "chunks"}
RENDER_FUNCTIONS = {
    "render",
    "render_to_response",
    "render_to_string",
    "get_template",
    "select_template",
    "TemplateResponse",
    "SimpleTemplateResponse",
}
EMAIL_FUNCTIONS = {
    "send_mail",
    "send_mass_mail",
    "mail_admins",
    "mail_managers",
    "EmailMessage",
    "EmailMultiAlternatives",
    "send_templated_mail",
    "send_email",
}
CACHE_OPERATIONS = {
    "get",
    "set",
    "add",
    "delete",
    "get_or_set",
    "incr",
    "decr",
    "set_many",
    "get_many",
    "delete_many",
    "touch",
    "delete_pattern",
    "has_key",
}
CACHE_WRITE_OPERATIONS = {"set", "add", "delete", "get_or_set", "incr", "decr", "set_many", "delete_many", "touch", "delete_pattern"}
URL_FUNCTIONS = {"path", "re_path", "url"}
REVERSE_FUNCTIONS = {"reverse", "reverse_lazy", "redirect", "resolve_url"}
TEST_CLIENT_METHODS = {"get", "post", "put", "patch", "delete", "head", "options", "generic"}
SKIPPED_CALLS = {
    "print",
    "len",
    "str",
    "int",
    "float",
    "bool",
    "list",
    "dict",
    "set",
    "tuple",
    "isinstance",
    "issubclass",
    "getattr",
    "setattr",
    "hasattr",
    "range",
    "enumerate",
    "zip",
    "sorted",
    "reversed",
    "min",
    "max",
    "sum",
    "any",
    "all",
    "map",
    "filter",
    "super",
    "type",
    "repr",
    "round",
    "abs",
    "open",
    "iter",
    "next",
    "format",
    "id",
    "hash",
    "vars",
    "dir",
    "callable",
    "Exception",
    "ValueError",
    "TypeError",
    "KeyError",
    "RuntimeError",
    "NotImplementedError",
}

MODEL_BASE_NAMES = {
    "Model",
    "AbstractUser",
    "AbstractBaseUser",
    "PolymorphicModel",
    "MPTTModel",
    "TimeStampedModel",
    "SoftDeletableModel",
}
RELATION_FIELDS = {"ForeignKey", "OneToOneField", "ManyToManyField", "TreeForeignKey", "ParentalKey", "GenericRelation"}
AUTH_DECORATOR = re.compile(
    r"(?i)(login_required|permission_required|user_passes_test|staff_member_required|superuser_required|_required$|^requires?_(login|perm|staff|auth|admin|role)|has_perm|permission)"
)
AUTH_MIXIN = re.compile(r"(LoginRequired|PermissionRequired|UserPassesTest|StaffRequired|SuperuserRequired|AccessMixin|IsAuthenticated)")
VIEW_BASE = re.compile(r"(View|ViewSet|APIView|Consumer)$")
TEST_BASE = re.compile(r"(TestCase|TestCaseMixin|APITestCase|LiveServerTestCase|TransactionTestCase)$")
VIEW_DECORATORS = {"api_view", "action", "require_http_methods", "require_POST", "require_GET", "require_safe"}
TASK_DECORATORS = {"shared_task", "task", "periodic_task"}
SIGNAL_SENDER_BLOCKLIST = {"self", "request", "mail", "email", "message", "msg", "connection", "socket", "sock", "channel", "client", "response", "smtp"}
MANAGER_NAME = re.compile(r"^(objects|all_objects|_base_manager|_default_manager|[a-z_]*objects|[a-z_]*manager)$")


@lru_cache(maxsize=1)
def python_parser() -> Parser:
    return Parser(Language(tree_sitter_python.language()))


@dataclass
class ImportEntry:
    alias: str
    target: str
    is_star: bool
    line: int


@dataclass
class RawRef:
    src: str
    kind: str
    target: str
    line: int
    extra: dict = field(default_factory=dict)


@dataclass
class FieldInfo:
    name: str
    field_type: str
    related: str | None
    line: int
    options: dict = field(default_factory=dict)


@dataclass
class SymbolInfo:
    qualname: str
    name: str
    kind: str
    start_line: int
    end_line: int
    parent: str | None
    decorators: list[str] = field(default_factory=list)
    bases: list[str] = field(default_factory=list)
    signature_lines: tuple[int, int] = (0, 0)
    body_statements: list[tuple[int, int]] = field(default_factory=list)
    docstring_lines: tuple[int, int] | None = None
    is_async: bool = False
    tags: set[str] = field(default_factory=set)
    tx: list[dict] = field(default_factory=list)
    fields: list[FieldInfo] = field(default_factory=list)
    managers: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)
    skeleton_lines: list[int] = field(default_factory=list)
    task_name: str | None = None
    returns_pattern: str | None = None


@dataclass
class UrlPatternInfo:
    index: int
    start_line: int
    end_line: int
    function: str
    route: str | None
    name: str | None
    view_ref: str | None
    include_ref: str | None
    namespace: str | None
    is_regex: bool
    wrappers: list[str] = field(default_factory=list)
    symbol: str = ""


@dataclass
class ParsedPython:
    path: str
    module: str
    line_count: int
    imports: list[ImportEntry] = field(default_factory=list)
    symbols: list[SymbolInfo] = field(default_factory=list)
    url_patterns: list[UrlPatternInfo] = field(default_factory=list)
    refs: list[RawRef] = field(default_factory=list)
    header_lines: list[int] = field(default_factory=list)
    app_name: str | None = None
    signal_defs: list[str] = field(default_factory=list)
    module_variables: list[str] = field(default_factory=list)
    module_tags: set[str] = field(default_factory=set)
    migration: dict | None = None
    data_migration_functions: set[str] = field(default_factory=set)
    syntax_errors: int = 0


@dataclass
class WalkContext:
    src: str
    enclosing_class: str | None
    in_atomic: bool = False
    in_on_commit: bool = False
    symbol: SymbolInfo | None = None


def last_segment(dotted: str) -> str:
    return dotted.rsplit(".", 1)[-1]


class PythonExtractor:
    def __init__(self, path: str, source: bytes, module: str, is_test_file: bool, is_migration: bool):
        self.path = path
        self.source = source
        self.module = module
        self.is_test_file = is_test_file
        self.is_migration = is_migration
        self.lines = source.decode("utf-8", "replace").split("\n")
        self.result = ParsedPython(path=path, module=module, line_count=len(self.lines))
        self.import_aliases: dict[str, str] = {}
        self.covered_lines: set[int] = set()
        self.local_values: dict[tuple[str, str], Node] = {}
        self.local_names: dict[str, set[str]] = {}

    def text(self, node: Node | None) -> str:
        if node is None:
            return ""
        return self.source[node.start_byte : node.end_byte].decode("utf-8", "replace")

    def dotted(self, node: Node | None) -> str | None:
        if node is None:
            return None
        if node.type == "identifier":
            return self.text(node)
        if node.type == "attribute":
            base = self.dotted(node.child_by_field_name("object"))
            attribute = node.child_by_field_name("attribute")
            if base is None or attribute is None:
                return None
            return f"{base}.{self.text(attribute)}"
        if node.type == "call":
            function = node.child_by_field_name("function")
            if function is not None and function.type == "identifier" and self.text(function) == "super":
                return "super()"
        if node.type == "parenthesized_expression" and node.named_child_count == 1:
            return self.dotted(node.named_children[0])
        return None

    def string_value(self, node: Node | None) -> str | None:
        if node is None:
            return None
        if node.type == "string":
            pieces = []
            for child in node.children:
                if child.type == "string_content":
                    pieces.append(self.text(child))
                elif child.type == "interpolation":
                    pieces.append("{}")
                elif child.type == "escape_sequence":
                    pieces.append(self.text(child))
            return "".join(pieces)
        if node.type == "concatenated_string":
            parts = []
            for child in node.named_children:
                value = self.string_value(child)
                if value is not None:
                    parts.append(value)
            return "".join(parts)
        if node.type == "binary_operator":
            left = self.string_value(node.child_by_field_name("left"))
            operator = node.child_by_field_name("operator")
            operator_text = self.text(operator) if operator is not None else ""
            if left is not None and operator_text == "%":
                return re.sub(r"%[sdrif]|%\([^)]*\)[sdrif]", "{}", left)
            if left is not None and operator_text == "+":
                right = self.string_value(node.child_by_field_name("right"))
                return left + (right if right is not None else "{}")
        if node.type == "call":
            function = node.child_by_field_name("function")
            if function is not None and function.type == "attribute":
                attribute = function.child_by_field_name("attribute")
                if attribute is not None and self.text(attribute) == "format":
                    base = self.string_value(function.child_by_field_name("object"))
                    if base is not None:
                        return re.sub(r"\{[^}]*\}", "{}", base)
        return None

    def split_arguments(self, arguments: Node | None) -> tuple[list[Node], dict[str, Node]]:
        positional: list[Node] = []
        keywords: dict[str, Node] = {}
        if arguments is None:
            return positional, keywords
        for child in arguments.named_children:
            if child.type == "keyword_argument":
                name = child.child_by_field_name("name")
                value = child.child_by_field_name("value")
                if name is not None and value is not None:
                    keywords[self.text(name)] = value
            elif child.type not in ("comment", "list_splat", "dictionary_splat"):
                positional.append(child)
        return positional, keywords

    def line_of(self, node: Node) -> int:
        return node.start_point.row + 1

    def end_line_of(self, node: Node) -> int:
        end_row = node.end_point.row
        if node.end_point.column == 0 and end_row > node.start_point.row:
            end_row -= 1
        return end_row + 1

    def mark_covered(self, start_line: int, end_line: int) -> None:
        for number in range(start_line, end_line + 1):
            self.covered_lines.add(number)

    def run(self) -> ParsedPython:
        tree = python_parser().parse(self.source)
        root = tree.root_node
        self.result.syntax_errors = self.count_errors(root)
        self.collect_imports(root)
        module_context = WalkContext(src=self.module, enclosing_class=None)
        for statement in root.named_children:
            self.visit_module_statement(statement, module_context)
        self.compute_header_lines()
        if self.is_migration:
            self.finish_migration()
        return self.result

    def count_errors(self, root: Node) -> int:
        count = 0
        stack = [root]
        while stack:
            node = stack.pop()
            if node.type == "ERROR" or node.is_missing:
                count += 1
            if node.has_error:
                stack.extend(node.children)
        return count

    def resolve_relative(self, prefix_dots: int, module_part: str) -> str:
        package_parts = self.module.split(".")
        is_package_init = self.path.endswith("__init__.py")
        if not is_package_init:
            package_parts = package_parts[:-1]
        if prefix_dots > 1:
            package_parts = package_parts[: len(package_parts) - (prefix_dots - 1)]
        base = ".".join(part for part in package_parts if part)
        if module_part:
            return f"{base}.{module_part}" if base else module_part
        return base

    def collect_imports(self, root: Node) -> None:
        stack = [root]
        while stack:
            node = stack.pop()
            if node.type == "import_statement":
                self.handle_import(node)
            elif node.type == "import_from_statement":
                self.handle_import_from(node)
            elif node.type in ("function_definition", "class_definition", "lambda"):
                body = node.child_by_field_name("body")
                if body is not None:
                    stack.append(body)
            else:
                stack.extend(node.named_children)

    def handle_import(self, node: Node) -> None:
        line = self.line_of(node)
        for child in node.named_children:
            if child.type == "dotted_name":
                full = self.text(child)
                top = full.split(".")[0]
                self.add_import(top, top, False, line)
                if "." in full:
                    self.add_import(full, full, False, line)
            elif child.type == "aliased_import":
                name = self.text(child.child_by_field_name("name"))
                alias = self.text(child.child_by_field_name("alias"))
                self.add_import(alias, name, False, line)

    def handle_import_from(self, node: Node) -> None:
        line = self.line_of(node)
        module_node = node.child_by_field_name("module_name")
        module_text = self.text(module_node)
        if module_node is not None and module_node.type == "relative_import":
            dots = len(module_text) - len(module_text.lstrip("."))
            base_module = self.resolve_relative(dots, module_text.lstrip("."))
        else:
            base_module = module_text
        for child in node.named_children:
            if child == module_node:
                continue
            if child.type == "wildcard_import":
                self.add_import("*", base_module, True, line)
            elif child.type == "dotted_name":
                name = self.text(child)
                target = f"{base_module}.{name}" if base_module else name
                self.add_import(name.split(".")[-1], target, False, line)
            elif child.type == "aliased_import":
                name = self.text(child.child_by_field_name("name"))
                alias = self.text(child.child_by_field_name("alias"))
                target = f"{base_module}.{name}" if base_module else name
                self.add_import(alias, target, False, line)

    def add_import(self, alias: str, target: str, is_star: bool, line: int) -> None:
        self.result.imports.append(ImportEntry(alias=alias, target=target, is_star=is_star, line=line))
        if not is_star:
            self.import_aliases.setdefault(alias, target)
            self.result.refs.append(RawRef(src=self.module, kind="import", target=target, line=line))

    def imported_from(self, name: str, module_prefix: str) -> bool:
        target = self.import_aliases.get(name.split(".")[0])
        return target is not None and target.startswith(module_prefix)

    def visit_module_statement(self, node: Node, context: WalkContext) -> None:
        if node.type == "decorated_definition":
            definition = node.child_by_field_name("definition")
            decorators = [child for child in node.children if child.type == "decorator"]
            if definition is not None and definition.type == "function_definition":
                self.handle_function(definition, decorators, node, parent_class=None)
            elif definition is not None and definition.type == "class_definition":
                self.handle_class(definition, decorators, node, parent_class=None)
            return
        if node.type == "function_definition":
            self.handle_function(node, [], node, parent_class=None)
            return
        if node.type == "class_definition":
            self.handle_class(node, [], node, parent_class=None)
            return
        if node.type == "expression_statement":
            self.handle_module_expression(node, context)
            return
        if node.type in ("if_statement", "try_statement", "with_statement", "for_statement", "while_statement"):
            for block in self.blocks_of(node):
                for statement in block.named_children:
                    self.visit_module_statement(statement, context)
            self.walk_calls_in_conditions(node, context)
            return
        if node.type in ("import_statement", "import_from_statement", "future_import_statement", "comment"):
            return
        self.walk(node, context)

    def blocks_of(self, node: Node) -> list[Node]:
        blocks = []
        for child in node.named_children:
            if child.type == "block":
                blocks.append(child)
            elif child.type in ("else_clause", "elif_clause", "except_clause", "finally_clause", "except_group_clause"):
                for grandchild in child.named_children:
                    if grandchild.type == "block":
                        blocks.append(grandchild)
        return blocks

    def walk_calls_in_conditions(self, node: Node, context: WalkContext) -> None:
        condition = node.child_by_field_name("condition")
        if condition is not None:
            self.walk(condition, context)

    def handle_module_expression(self, node: Node, context: WalkContext) -> None:
        inner = node.named_children[0] if node.named_child_count else None
        if inner is None:
            return
        if inner.type == "assignment":
            left = inner.child_by_field_name("left")
            right = inner.child_by_field_name("right")
            left_text = self.text(left)
            if left_text == "urlpatterns" and right is not None:
                self.collect_url_list(right)
                return
            if left is not None and left.type == "identifier" and left_text not in self.result.module_variables:
                self.result.module_variables.append(left_text)
            if left_text == "app_name" and right is not None:
                self.result.app_name = self.string_value(right)
            if left_text == "ROOT_URLCONF" and right is not None:
                value = self.string_value(right)
                if value:
                    self.result.refs.append(RawRef(src=self.module, kind="root_urlconf", target=value, line=self.line_of(node)))
            if right is not None and right.type == "call" and left is not None and left.type == "identifier":
                callee = self.dotted(right.child_by_field_name("function")) or ""
                if last_segment(callee) == "Signal":
                    self.result.signal_defs.append(left_text)
                    self.result.module_tags.add("signal_definition")
            if right is not None:
                self.walk(right, context)
            return
        if inner.type == "augmented_assignment":
            left = inner.child_by_field_name("left")
            right = inner.child_by_field_name("right")
            if self.text(left) == "urlpatterns" and right is not None:
                self.collect_url_list(right)
                return
            if right is not None:
                self.walk(right, context)
            return
        if inner.type == "call":
            function_text = self.dotted(inner.child_by_field_name("function")) or ""
            if function_text in ("urlpatterns.append", "urlpatterns.extend"):
                arguments = inner.child_by_field_name("arguments")
                if arguments is not None:
                    for argument in arguments.named_children:
                        if argument.type == "list":
                            self.collect_url_list(argument)
                        elif argument.type == "call":
                            self.collect_url_element(argument)
                return
        self.walk(inner, context)

    def collect_url_list(self, node: Node) -> None:
        if node.type in ("list", "tuple"):
            for element in node.named_children:
                if element.type == "call":
                    self.collect_url_element(element)
        elif node.type == "binary_operator":
            self.collect_url_list(node.child_by_field_name("left"))
            self.collect_url_list(node.child_by_field_name("right"))
        elif node.type == "call":
            function_text = self.dotted(node.child_by_field_name("function")) or ""
            if last_segment(function_text) in URL_FUNCTIONS:
                self.collect_url_element(node)
            else:
                self.walk(node, WalkContext(src=self.module, enclosing_class=None))

    def unwrap_view(self, node: Node, wrappers: list[str]) -> tuple[str | None, str | None]:
        if node.type == "call":
            function_node = node.child_by_field_name("function")
            function_text = self.dotted(function_node)
            if function_text and last_segment(function_text) == "include":
                positional, keywords = self.split_arguments(node.child_by_field_name("arguments"))
                include_ref = None
                if positional:
                    first = positional[0]
                    include_ref = self.string_value(first) or self.dotted(first)
                    if include_ref is None and first.type == "tuple" and first.named_child_count:
                        include_ref = self.string_value(first.named_children[0]) or self.dotted(first.named_children[0])
                namespace = self.string_value(keywords.get("namespace"))
                return None, f"{include_ref or ''}|{namespace or ''}"
            if function_node is not None and function_node.type == "attribute":
                attribute = function_node.child_by_field_name("attribute")
                if attribute is not None and self.text(attribute) == "as_view":
                    return self.dotted(function_node.child_by_field_name("object")), None
            positional, _ = self.split_arguments(node.child_by_field_name("arguments"))
            if function_text:
                wrappers.append(last_segment(function_text))
            if positional:
                return self.unwrap_view(positional[0], wrappers)
            return None, None
        string_view = self.string_value(node)
        if string_view is not None:
            return string_view, None
        return self.dotted(node), None

    def collect_url_element(self, node: Node) -> None:
        function_text = self.dotted(node.child_by_field_name("function")) or ""
        function_name = last_segment(function_text)
        if function_name not in URL_FUNCTIONS:
            return
        positional, keywords = self.split_arguments(node.child_by_field_name("arguments"))
        route = self.string_value(positional[0]) if positional else None
        view_ref = None
        include_ref = None
        namespace = None
        wrappers: list[str] = []
        view_node = positional[1] if len(positional) > 1 else keywords.get("view")
        if view_node is not None:
            view_ref, include_blob = self.unwrap_view(view_node, wrappers)
            if include_blob is not None:
                include_ref, namespace = include_blob.split("|", 1)
                include_ref = include_ref or None
                namespace = namespace or None
        name = self.string_value(keywords.get("name"))
        start = self.line_of(node)
        end = self.end_line_of(node)
        info = UrlPatternInfo(
            index=len(self.result.url_patterns),
            start_line=start,
            end_line=end,
            function=function_name,
            route=route,
            name=name,
            view_ref=view_ref,
            include_ref=include_ref,
            namespace=namespace,
            is_regex=function_name in ("re_path", "url"),
            wrappers=wrappers,
        )
        key = name or route or f"#{info.index}"
        taken = {pattern.symbol for pattern in self.result.url_patterns}
        url_symbol = f"{self.module}.urlpatterns[{key}]"
        if url_symbol in taken:
            url_symbol = f"{self.module}.urlpatterns[{key}#{info.index}]"
        info.symbol = url_symbol
        self.result.url_patterns.append(info)
        self.mark_covered(start, end)
        if view_ref:
            self.result.refs.append(RawRef(src=url_symbol, kind="url_view", target=view_ref, line=start, extra={"wrappers": wrappers}))
        if include_ref:
            self.result.refs.append(RawRef(src=url_symbol, kind="url_include", target=include_ref, line=start))

    def decorator_parts(self, decorator: Node) -> tuple[str, Node | None]:
        expression = decorator.named_children[0] if decorator.named_child_count else None
        if expression is None:
            return "", None
        if expression.type == "call":
            return self.dotted(expression.child_by_field_name("function")) or self.text(expression), expression
        return self.dotted(expression) or self.text(expression), None

    def apply_decorators(self, symbol: SymbolInfo, decorators: list[Node], context: WalkContext) -> None:
        for decorator in decorators:
            name, call_node = self.decorator_parts(decorator)
            symbol.decorators.append(self.text(decorator).lstrip("@").strip())
            short = last_segment(name)
            line = self.line_of(decorator)
            positional: list[Node] = []
            keywords: dict[str, Node] = {}
            if call_node is not None:
                positional, keywords = self.split_arguments(call_node.child_by_field_name("arguments"))
            if short == "receiver":
                signals = []
                if positional:
                    first = positional[0]
                    if first.type in ("list", "tuple"):
                        for element in first.named_children:
                            signals.append(self.dotted(element) or self.text(element))
                    else:
                        signals.append(self.dotted(first) or self.text(first))
                sender = self.dotted(keywords.get("sender")) or self.string_value(keywords.get("sender")) or "*"
                symbol.tags.add("signal_receiver")
                self.result.refs.append(
                    RawRef(src=symbol.qualname, kind="receiver", target=sender, line=line, extra={"signals": signals, "via": "decorator"})
                )
            elif short in TASK_DECORATORS or name.endswith(".task"):
                symbol.tags.add("celery_task")
                symbol.task_name = self.string_value(keywords.get("name"))
            elif short == "atomic":
                symbol.tags.add("atomic_block")
                symbol.tx.append({"kind": "atomic_decorator", "line": line})
            elif short == "method_decorator" and positional:
                wrapped = self.dotted(positional[0]) or self.text(positional[0])
                if AUTH_DECORATOR.search(last_segment(wrapped)):
                    symbol.tags.add("auth_gate")
                if last_segment(wrapped) == "csrf_exempt":
                    symbol.tags.add("csrf_exempt")
            elif short == "csrf_exempt":
                symbol.tags.add("csrf_exempt")
            elif short in ("cache_page", "cache_control", "vary_on_cookie"):
                symbol.tags.add("cached_view")
            elif short == "register" and name.startswith("admin"):
                symbol.tags.add("admin")
                for model_node in positional:
                    model_ref = self.dotted(model_node)
                    if model_ref:
                        self.result.refs.append(
                            RawRef(src=symbol.qualname, kind="admin_register", target=model_ref, line=line, extra={"admin_class": symbol.qualname})
                        )
            elif name in ("pytest.fixture", "fixture"):
                symbol.tags.add("fixture")
            if short in VIEW_DECORATORS:
                symbol.tags.add("view")
            if AUTH_DECORATOR.search(short) and short != "method_decorator":
                symbol.tags.add("auth_gate")
            if name:
                self.result.refs.append(RawRef(src=symbol.qualname, kind="decorator", target=name, line=line))
            if call_node is not None:
                arguments = call_node.child_by_field_name("arguments")
                if arguments is not None:
                    self.walk(arguments, context)

    def function_signature_range(self, node: Node, outer: Node) -> tuple[int, int]:
        body = node.child_by_field_name("body")
        start = self.line_of(outer)
        if body is None:
            return start, self.end_line_of(node)
        body_line = self.line_of(body)
        signature_end = body_line - 1 if body_line > self.line_of(node) else body_line
        return start, max(signature_end, self.line_of(node))

    def record_parameters(self, src: str, parameters: Node | None) -> None:
        if parameters is None:
            return
        names = self.local_names.setdefault(src, set())
        for parameter in parameters.named_children:
            if parameter.type == "identifier":
                names.add(self.text(parameter))
                continue
            name_node = parameter.child_by_field_name("name")
            if name_node is None and parameter.named_child_count:
                name_node = parameter.named_children[0]
            if name_node is not None and name_node.type == "identifier":
                names.add(self.text(name_node))

    def record_local_targets(self, src: str, target: Node | None) -> None:
        if target is None:
            return
        names = self.local_names.setdefault(src, set())
        if target.type == "identifier":
            names.add(self.text(target))
            return
        if target.type in ("pattern_list", "tuple_pattern", "list_pattern", "tuple", "list"):
            for element in target.named_children:
                self.record_local_targets(src, element)

    def single_return_pattern(self, body: Node) -> str | None:
        statements = []
        for statement in body.named_children:
            if statement.type == "comment":
                continue
            if statement.type == "expression_statement" and statement.named_child_count and statement.named_children[0].type == "string":
                continue
            statements.append(statement)
        if len(statements) != 1 or statements[0].type != "return_statement":
            return None
        if statements[0].named_child_count == 0:
            return None
        value = self.string_value(statements[0].named_children[0])
        if value is None or "{}" not in value:
            return None
        return value

    def docstring_range(self, body: Node | None) -> tuple[int, int] | None:
        if body is None or body.named_child_count == 0:
            return None
        first = body.named_children[0]
        if first.type == "expression_statement" and first.named_child_count and first.named_children[0].type == "string":
            return self.line_of(first), self.end_line_of(first)
        return None

    def handle_function(self, node: Node, decorators: list[Node], outer: Node, parent_class: SymbolInfo | None) -> SymbolInfo:
        name = self.text(node.child_by_field_name("name"))
        if parent_class is not None:
            qualname = f"{parent_class.qualname}.{name}"
            is_test_method = "test_class" in parent_class.tags and name.startswith("test")
            kind = "test" if is_test_method else "method"
        else:
            qualname = f"{self.module}.{name}"
            kind = "test" if self.is_test_file and name.startswith("test") else "function"
        start = self.line_of(outer)
        end = self.end_line_of(node)
        body = node.child_by_field_name("body")
        symbol = SymbolInfo(
            qualname=qualname,
            name=name,
            kind=kind,
            start_line=start,
            end_line=end,
            parent=parent_class.qualname if parent_class else None,
            signature_lines=self.function_signature_range(node, outer),
            docstring_lines=self.docstring_range(body),
            is_async=self.text(node).startswith("async"),
        )
        if body is not None:
            for statement in body.named_children:
                symbol.body_statements.append((self.line_of(statement), self.end_line_of(statement)))
            symbol.returns_pattern = self.single_return_pattern(body)
        if self.is_test_file:
            symbol.tags.add("test" if kind == "test" else "test_support")
        context = WalkContext(src=qualname, enclosing_class=parent_class.qualname if parent_class else None, symbol=symbol)
        self.record_parameters(qualname, node.child_by_field_name("parameters"))
        self.apply_decorators(symbol, decorators, context)
        if "atomic_block" in symbol.tags:
            context.in_atomic = True
        if body is not None:
            self.walk(body, context)
        parameters = node.child_by_field_name("parameters")
        if parameters is not None:
            self.walk(parameters, context)
        self.result.symbols.append(symbol)
        self.mark_covered(start, end)
        return symbol

    def handle_class(self, node: Node, decorators: list[Node], outer: Node, parent_class: SymbolInfo | None) -> SymbolInfo:
        name = self.text(node.child_by_field_name("name"))
        qualname = f"{parent_class.qualname}.{name}" if parent_class else f"{self.module}.{name}"
        start = self.line_of(outer)
        end = self.end_line_of(node)
        body = node.child_by_field_name("body")
        symbol = SymbolInfo(
            qualname=qualname,
            name=name,
            kind="class",
            start_line=start,
            end_line=end,
            parent=parent_class.qualname if parent_class else None,
            signature_lines=self.function_signature_range(node, outer),
            docstring_lines=self.docstring_range(body),
        )
        superclasses = node.child_by_field_name("superclasses")
        if superclasses is not None:
            positional, _ = self.split_arguments(superclasses)
            for base_node in positional:
                base_text = self.dotted(base_node)
                if base_text is None and base_node.type == "subscript":
                    base_text = self.dotted(base_node.child_by_field_name("value"))
                if base_text:
                    symbol.bases.append(base_text)
                    self.result.refs.append(RawRef(src=qualname, kind="base", target=base_text, line=self.line_of(base_node)))
        self.tag_class_from_bases(symbol)
        context = WalkContext(src=qualname, enclosing_class=qualname, symbol=symbol)
        self.apply_decorators(symbol, decorators, context)

        skeleton = set(range(symbol.signature_lines[0], symbol.signature_lines[1] + 1))
        if symbol.docstring_lines:
            skeleton.update(range(symbol.docstring_lines[0], symbol.docstring_lines[1] + 1))
        if body is not None:
            for statement in body.named_children:
                self.handle_class_statement(statement, symbol, context, skeleton)
        if symbol.fields and symbol.kind == "class" and not symbol.tags.intersection({"form", "serializer", "factory", "admin"}):
            symbol.kind = "model"
            symbol.tags.add("model")
        if "model" in symbol.tags:
            symbol.kind = "model"
        symbol.skeleton_lines = sorted(skeleton)
        self.result.symbols.append(symbol)
        self.mark_covered(start, end)
        return symbol

    def tag_class_from_bases(self, symbol: SymbolInfo) -> None:
        for base in symbol.bases:
            short = last_segment(base)
            if short in MODEL_BASE_NAMES and (base.startswith("models.") or self.imported_from(base, "django") or short != "Model"):
                symbol.tags.add("model")
            if short.endswith("ModelAdmin") or short.endswith("Inline") or short == "AdminSite":
                symbol.tags.add("admin")
            if VIEW_BASE.search(short) or short.endswith("Mixin") and "View" in short:
                symbol.tags.add("view")
            if short.endswith("Serializer"):
                symbol.tags.add("serializer")
            if short.endswith("Form") or short.endswith("FormSet"):
                symbol.tags.add("form")
            if short.endswith("Factory"):
                symbol.tags.add("factory")
            if TEST_BASE.search(short) or (self.is_test_file and symbol.name.startswith("Test")):
                symbol.tags.add("test_class")
            if AUTH_MIXIN.search(short):
                symbol.tags.add("auth_gate")
            if short == "Task" and self.imported_from(base, "celery"):
                symbol.tags.add("celery_task")
            if short.endswith("Consumer"):
                symbol.tags.add("consumer")
        if self.is_test_file and symbol.name.startswith("Test"):
            symbol.tags.add("test_class")

    def is_model_field_call(self, callee: str) -> bool:
        short = last_segment(callee)
        if callee.startswith("models.") and (short.endswith("Field") or short in RELATION_FIELDS):
            return True
        if short in RELATION_FIELDS and not callee.startswith(("forms.", "serializers.")):
            return True
        if short.endswith("Field") and "." not in callee and self.imported_from(callee, "django.db.models"):
            return True
        if short in ("MoneyField", "TreeForeignKey", "GenericForeignKey"):
            return True
        return False

    def handle_class_statement(self, statement: Node, symbol: SymbolInfo, context: WalkContext, skeleton: set[int]) -> None:
        if statement.type == "decorated_definition":
            definition = statement.child_by_field_name("definition")
            decorators = [child for child in statement.children if child.type == "decorator"]
            if definition is not None and definition.type == "function_definition":
                method = self.handle_function(definition, decorators, statement, parent_class=symbol)
                skeleton.update(range(method.signature_lines[0], method.signature_lines[1] + 1))
            elif definition is not None and definition.type == "class_definition":
                nested = self.handle_class(definition, decorators, statement, parent_class=symbol)
                skeleton.update(range(nested.signature_lines[0], nested.signature_lines[1] + 1))
            return
        if statement.type == "function_definition":
            method = self.handle_function(statement, [], statement, parent_class=symbol)
            skeleton.update(range(method.signature_lines[0], method.signature_lines[1] + 1))
            return
        if statement.type == "class_definition":
            nested_name = self.text(statement.child_by_field_name("name"))
            if nested_name in ("Meta", "Config"):
                self.handle_meta_class(statement, symbol, context)
                skeleton.update(range(self.line_of(statement), self.end_line_of(statement) + 1))
                return
            nested = self.handle_class(statement, [], statement, parent_class=symbol)
            skeleton.update(range(nested.signature_lines[0], nested.signature_lines[1] + 1))
            return
        skeleton.update(range(self.line_of(statement), self.end_line_of(statement) + 1))
        if statement.type != "expression_statement" or statement.named_child_count == 0:
            self.walk(statement, context)
            return
        inner = statement.named_children[0]
        if inner.type != "assignment":
            self.walk(inner, context)
            return
        left = inner.child_by_field_name("left")
        right = inner.child_by_field_name("right")
        attribute_name = self.text(left)
        if right is None:
            return
        line = self.line_of(statement)
        if right.type == "call":
            callee = self.dotted(right.child_by_field_name("function")) or ""
            if self.is_model_field_call(callee):
                symbol.fields.append(self.field_info(attribute_name, callee, right, line))
            elif last_segment(callee).endswith("Manager") or callee.endswith("as_manager"):
                symbol.managers.append(attribute_name)
        if attribute_name == "template_name":
            template = self.string_value(right)
            if template:
                self.result.refs.append(RawRef(src=symbol.qualname, kind="template", target=template, line=line, extra={"via": "template_name"}))
        elif attribute_name == "model":
            model_ref = self.dotted(right)
            if model_ref:
                self.result.refs.append(RawRef(src=symbol.qualname, kind="meta_model", target=model_ref, line=line, extra={"via": "model_attr"}))
        elif attribute_name in ("permission_classes", "authentication_classes"):
            permission_text = self.text(right)
            if permission_text.strip("[]() ") and "AllowAny" not in permission_text:
                symbol.tags.add("auth_gate")
            if "AllowAny" in permission_text:
                symbol.tags.add("allow_any")
        elif attribute_name in ("serializer_class", "form_class", "filterset_class"):
            reference = self.dotted(right)
            if reference:
                self.result.refs.append(RawRef(src=symbol.qualname, kind="reference", target=reference, line=line))
        self.walk(right, context)

    def field_info(self, name: str, callee: str, call_node: Node, line: int) -> FieldInfo:
        positional, keywords = self.split_arguments(call_node.child_by_field_name("arguments"))
        related = None
        short = last_segment(callee)
        if short in RELATION_FIELDS or short in ("TreeForeignKey", "OneToOneField"):
            target_node = keywords.get("to") or (positional[0] if positional else None)
            if target_node is not None:
                related = self.string_value(target_node) or self.dotted(target_node)
        options = {}
        for option in ("null", "blank", "unique", "db_index", "default", "on_delete", "max_digits", "decimal_places", "related_name"):
            if option in keywords:
                options[option] = self.text(keywords[option])[:60]
        return FieldInfo(name=name, field_type=short, related=related, line=line, options=options)

    def handle_meta_class(self, node: Node, owner: SymbolInfo, context: WalkContext) -> None:
        body = node.child_by_field_name("body")
        if body is None:
            return
        for statement in body.named_children:
            if statement.type != "expression_statement" or statement.named_child_count == 0:
                continue
            inner = statement.named_children[0]
            if inner.type != "assignment":
                continue
            key = self.text(inner.child_by_field_name("left"))
            value = inner.child_by_field_name("right")
            if value is None:
                continue
            value_text = self.text(value)
            owner.meta[key] = value_text[:200]
            if key == "model":
                model_ref = self.dotted(value) or self.string_value(value)
                if model_ref:
                    self.result.refs.append(
                        RawRef(src=owner.qualname, kind="meta_model", target=model_ref, line=self.line_of(statement), extra={"via": "meta"})
                    )
            if key == "abstract" and value_text == "True":
                owner.tags.add("abstract")

    def walk(self, node: Node, context: WalkContext) -> None:
        stack: list[tuple[Node, WalkContext]] = [(node, context)]
        while stack:
            current, current_context = stack.pop()
            node_type = current.type
            if node_type == "call":
                extra_children = self.handle_call(current, current_context)
                for child, child_context in reversed(extra_children):
                    stack.append((child, child_context))
                continue
            if node_type == "with_statement":
                atomic_context = self.with_statement_context(current, current_context)
                for child in reversed(current.named_children):
                    stack.append((child, atomic_context))
                continue
            if node_type == "assignment" and current_context.symbol is not None:
                left = current.child_by_field_name("left")
                right = current.child_by_field_name("right")
                self.record_local_targets(current_context.src, left)
                if left is not None and right is not None and left.type == "identifier":
                    self.local_values[(current_context.src, self.text(left))] = right
            if node_type in ("for_statement", "for_in_clause") and current_context.symbol is not None:
                self.record_local_targets(current_context.src, current.child_by_field_name("left"))
            if node_type == "as_pattern" and current_context.symbol is not None:
                alias = current.child_by_field_name("alias")
                if alias is not None and alias.named_child_count:
                    self.record_local_targets(current_context.src, alias.named_children[0])
                elif alias is not None:
                    self.record_local_targets(current_context.src, alias)
            is_definition = node_type in ("function_definition", "class_definition", "decorated_definition")
            if is_definition and current != node and current_context.symbol is None:
                continue
            if is_definition and current != node:
                fresh_context = WalkContext(
                    src=current_context.src,
                    enclosing_class=current_context.enclosing_class,
                    symbol=current_context.symbol,
                )
                for child in reversed(current.named_children):
                    stack.append((child, fresh_context))
                continue
            if node_type == "attribute":
                self.maybe_tenant_reference(current, current_context)
            for child in reversed(current.named_children):
                stack.append((child, current_context))

    def maybe_tenant_reference(self, node: Node, context: WalkContext) -> None:
        if context.symbol is None:
            return
        text = self.text(node)
        if text in ("request.site", "request.tenant", "request.organization") or text.endswith("get_current_site"):
            context.symbol.tags.add("tenant_scoped")

    def with_statement_context(self, node: Node, context: WalkContext) -> WalkContext:
        for child in node.named_children:
            if child.type != "with_clause":
                continue
            for item in child.named_children:
                value = item.child_by_field_name("value") if item.type == "with_item" else None
                if value is not None and value.type == "as_pattern" and value.named_child_count:
                    value = value.named_children[0]
                if value is None:
                    continue
                call_node = value if value.type == "call" else None
                callee = self.dotted(call_node.child_by_field_name("function")) if call_node is not None else self.dotted(value)
                if callee and last_segment(callee) == "atomic":
                    if context.symbol is not None:
                        context.symbol.tags.add("atomic_block")
                        context.symbol.tx.append({"kind": "atomic_with", "line": self.line_of(node), "end_line": self.end_line_of(node)})
                    return WalkContext(
                        src=context.src,
                        enclosing_class=context.enclosing_class,
                        in_atomic=True,
                        in_on_commit=context.in_on_commit,
                        symbol=context.symbol,
                    )
        return context

    def query_chain(self, call_node: Node) -> tuple[list[str], list[str], Node]:
        methods: list[str] = []
        keyword_names: list[str] = []
        current = call_node
        while True:
            function_node = current.child_by_field_name("function")
            if function_node is not None and function_node.type == "attribute":
                attribute = function_node.child_by_field_name("attribute")
                methods.append(self.text(attribute))
            _, keywords = self.split_arguments(current.child_by_field_name("arguments"))
            for keyword in keywords:
                keyword_names.append(keyword)
            arguments = current.child_by_field_name("arguments")
            if arguments is not None:
                for argument in arguments.named_children:
                    if argument.type == "call":
                        inner_callee = self.dotted(argument.child_by_field_name("function")) or ""
                        if last_segment(inner_callee) == "Q":
                            _, q_keywords = self.split_arguments(argument.child_by_field_name("arguments"))
                            keyword_names.extend(q_keywords.keys())
            parent = current.parent
            if parent is not None and parent.type == "attribute" and parent.parent is not None and parent.parent.type == "call":
                current = parent.parent
                continue
            break
        return methods, keyword_names, current

    def detect_query(self, call_node: Node, context: WalkContext) -> bool:
        function_node = call_node.child_by_field_name("function")
        if function_node is None or function_node.type != "attribute":
            return False
        method_node = function_node.child_by_field_name("attribute")
        manager_node = function_node.child_by_field_name("object")
        if method_node is None or manager_node is None or manager_node.type != "attribute":
            return False
        method = self.text(method_node)
        manager = self.text(manager_node.child_by_field_name("attribute"))
        model_ref = self.dotted(manager_node.child_by_field_name("object"))
        if model_ref is None or method not in QUERYSET_METHODS or not MANAGER_NAME.match(manager):
            return False
        model_short = last_segment(model_ref)
        if not model_short[:1].isupper():
            return False
        methods, keyword_names, _ = self.query_chain(call_node)
        self.result.refs.append(
            RawRef(
                src=context.src,
                kind="query",
                target=model_ref,
                line=self.line_of(call_node),
                extra={"manager": manager, "methods": methods[:8], "kwargs": sorted(set(keyword_names))[:20], "in_atomic": context.in_atomic},
            )
        )
        if context.symbol is not None:
            context.symbol.tags.add("queries_db")
            if "select_for_update" in methods:
                context.symbol.tags.add("atomic_block")
                context.symbol.tx.append({"kind": "select_for_update", "line": self.line_of(call_node)})
            if "raw" in methods or "extra" in methods:
                context.symbol.tags.add("raw_sql")
        return True

    def cache_key(self, node: Node | None, context: WalkContext, depth: int = 0) -> str:
        if node is None:
            return "<none>"
        value = self.string_value(node)
        if value is not None:
            return value
        if node.type == "identifier" and depth < 3:
            assigned = self.local_values.get((context.src, self.text(node)))
            if assigned is not None and assigned != node:
                return self.cache_key(assigned, context, depth + 1)
        if node.type == "call":
            callee = self.dotted(node.child_by_field_name("function"))
            if callee:
                return f"<call:{callee}>"
        return f"<dynamic:{self.text(node)[:40]}>"

    def handle_call(self, node: Node, context: WalkContext) -> list[tuple[Node, WalkContext]]:
        function_node = node.child_by_field_name("function")
        arguments = node.child_by_field_name("arguments")
        positional, keywords = self.split_arguments(arguments)
        line = self.line_of(node)
        children: list[tuple[Node, WalkContext]] = []
        if function_node is not None and function_node.type not in ("identifier", "attribute"):
            children.append((function_node, context))
        elif function_node is not None and function_node.type == "attribute":
            receiver = function_node.child_by_field_name("object")
            while receiver is not None and receiver.type == "attribute":
                receiver = receiver.child_by_field_name("object")
            if receiver is not None and receiver.type != "identifier":
                children.append((receiver, context))

        self.detect_query(node, context)
        target = self.dotted(function_node)

        if function_node is not None and function_node.type == "attribute" and target is None:
            receiver = function_node.child_by_field_name("object")
            method = self.text(function_node.child_by_field_name("attribute"))
            if receiver is not None and receiver.type == "subscript" and self.text(receiver.child_by_field_name("value")) == "caches":
                if method in CACHE_OPERATIONS:
                    self.record_cache(context, method, positional, line)

        on_commit_lambda: Node | None = None
        if target:
            short = last_segment(target)
            prefix = target[: -(len(short) + 1)] if "." in target else ""
            if short in TASK_METHODS and prefix and last_segment(prefix) not in ("self", "cls", "super()"):
                if short not in ("s", "si", "signature", "map", "starmap", "chunks") or last_segment(prefix)[:1].islower():
                    self.result.refs.append(
                        RawRef(
                            src=context.src,
                            kind="task_call",
                            target=prefix,
                            line=line,
                            extra={"method": short, "in_atomic": context.in_atomic, "in_on_commit": context.in_on_commit},
                        )
                    )
                    if context.symbol is not None:
                        context.symbol.tags.add("enqueues_task")
            elif short == "send_task" and positional:
                task_name = self.string_value(positional[0])
                if task_name:
                    self.result.refs.append(
                        RawRef(src=context.src, kind="send_task_name", target=task_name, line=line, extra={"in_atomic": context.in_atomic, "in_on_commit": context.in_on_commit})
                    )
            elif short == "on_commit":
                if context.symbol is not None:
                    context.symbol.tags.add("atomic_block")
                    context.symbol.tx.append({"kind": "on_commit", "line": line})
                if positional:
                    first = positional[0]
                    if first.type == "lambda":
                        on_commit_lambda = first
                    else:
                        callback = self.dotted(first)
                        if first.type == "call":
                            inner_positional, _ = self.split_arguments(first.child_by_field_name("arguments"))
                            if inner_positional:
                                callback = self.dotted(inner_positional[0])
                        if callback:
                            self.result.refs.append(
                                RawRef(src=context.src, kind="reference", target=callback, line=line, extra={"via": "on_commit"})
                            )
            elif short == "connect" and positional and prefix:
                handler = self.dotted(positional[0])
                if handler:
                    sender = self.dotted(keywords.get("sender")) or self.string_value(keywords.get("sender")) or "*"
                    self.result.refs.append(
                        RawRef(
                            src=context.src,
                            kind="signal_connect",
                            target=handler,
                            line=line,
                            extra={"signal": prefix, "sender": sender},
                        )
                    )
            elif short in ("send", "send_robust") and prefix and last_segment(prefix) not in SIGNAL_SENDER_BLOCKLIST:
                self.result.refs.append(RawRef(src=context.src, kind="signal_send", target=prefix, line=line))
            if short in RENDER_FUNCTIONS:
                template = self.template_argument(positional, keywords)
                if template:
                    self.result.refs.append(RawRef(src=context.src, kind="template", target=template, line=line, extra={"via": short}))
                    if context.symbol is not None:
                        context.symbol.tags.add("renders_template")
            if short in EMAIL_FUNCTIONS:
                self.result.refs.append(RawRef(src=context.src, kind="email_send", target=target, line=line))
                if context.symbol is not None:
                    context.symbol.tags.add("sends_email")
            if short in CACHE_OPERATIONS and prefix and last_segment(prefix).lower().endswith("cache"):
                self.record_cache(context, short, positional, line)
            if short in REVERSE_FUNCTIONS and positional:
                url_name = self.string_value(positional[0])
                if url_name and "{}" not in url_name:
                    kind = "http_path" if url_name.startswith("/") else "url_name"
                    self.result.refs.append(RawRef(src=context.src, kind=kind, target=url_name, line=line))
            if short in TEST_CLIENT_METHODS and prefix and last_segment(prefix).endswith("client") and positional:
                path_value = self.string_value(positional[0])
                if path_value and path_value.startswith("/"):
                    self.result.refs.append(RawRef(src=context.src, kind="http_path", target=path_value, line=line, extra={"method": short}))
            if short == "register" and prefix and (prefix.endswith("site") or prefix.endswith("admin")) and positional:
                admin_class = self.dotted(positional[1]) if len(positional) > 1 else None
                models_node = positional[0]
                model_nodes = list(models_node.named_children) if models_node.type in ("list", "tuple") else [models_node]
                for model_node in model_nodes:
                    model_ref = self.dotted(model_node)
                    if model_ref:
                        self.result.refs.append(
                            RawRef(src=context.src, kind="admin_register", target=model_ref, line=line, extra={"admin_class": admin_class})
                        )
            elif short == "register" and prefix and len(positional) >= 2:
                viewset = self.dotted(positional[1])
                route = self.string_value(positional[0])
                if viewset:
                    self.result.refs.append(
                        RawRef(src=context.src, kind="router_register", target=viewset, line=line, extra={"route": route, "router": prefix})
                    )
            if short == "partial" and positional:
                wrapped = self.dotted(positional[0])
                if wrapped:
                    self.result.refs.append(RawRef(src=context.src, kind="reference", target=wrapped, line=line, extra={"via": "partial"}))
            if "target" in keywords:
                wrapped = self.dotted(keywords["target"])
                if wrapped:
                    self.result.refs.append(RawRef(src=context.src, kind="reference", target=wrapped, line=line, extra={"via": "target"}))
            if target.split(".")[0] not in SKIPPED_CALLS or "." in target:
                if short not in TASK_METHODS or not prefix:
                    call_extra = {}
                    if context.in_atomic:
                        call_extra["in_atomic"] = True
                    head = target.split(".")[0]
                    if head not in ("self", "cls") and head in self.local_names.get(context.src, set()):
                        call_extra["local"] = True
                    self.result.refs.append(RawRef(src=context.src, kind="call", target=target, line=line, extra=call_extra))

        if arguments is not None:
            for argument in arguments.named_children:
                if on_commit_lambda is not None and argument == on_commit_lambda:
                    lambda_context = WalkContext(
                        src=context.src,
                        enclosing_class=context.enclosing_class,
                        in_atomic=context.in_atomic,
                        in_on_commit=True,
                        symbol=context.symbol,
                    )
                    children.append((argument, lambda_context))
                else:
                    children.append((argument, context))
        return children

    def template_argument(self, positional: list[Node], keywords: dict[str, Node]) -> str | None:
        if "template_name" in keywords:
            value = self.string_value(keywords["template_name"])
            if value:
                return value
        for node in positional[:2]:
            value = self.string_value(node)
            if value and ("." in value or "/" in value):
                return value
            if node.type in ("list", "tuple"):
                for element in node.named_children:
                    element_value = self.string_value(element)
                    if element_value:
                        return element_value
        return None

    def record_cache(self, context: WalkContext, operation: str, positional: list[Node], line: int) -> None:
        key = self.cache_key(positional[0] if positional else None, context)
        self.result.refs.append(RawRef(src=context.src, kind="cache_op", target=key, line=line, extra={"op": operation}))
        if context.symbol is not None:
            if operation in CACHE_WRITE_OPERATIONS:
                context.symbol.tags.add("cache_write")
            else:
                context.symbol.tags.add("cache_read")

    def compute_header_lines(self) -> None:
        header = []
        for number, line in enumerate(self.lines, start=1):
            if number in self.covered_lines:
                continue
            if not line.strip():
                continue
            header.append(number)
        self.result.header_lines = header

    def finish_migration(self) -> None:
        operations = []
        run_python_functions: set[str] = set()
        tree = python_parser().parse(self.source)
        stack = [tree.root_node]
        while stack:
            node = stack.pop()
            if node.type == "call":
                callee = self.dotted(node.child_by_field_name("function")) or ""
                short = last_segment(callee)
                positional, keywords = self.split_arguments(node.child_by_field_name("arguments"))
                if callee.startswith("migrations.") or short in ("RunPython", "RunSQL", "AddField", "AlterField", "RemoveField", "CreateModel", "DeleteModel", "RenameField", "RenameModel", "AddIndex", "AddConstraint"):
                    operation = {"op": short, "line": self.line_of(node)}
                    model_name = self.string_value(keywords.get("model_name")) or self.string_value(keywords.get("name"))
                    if model_name is None and positional:
                        model_name = self.string_value(positional[0])
                    if model_name:
                        operation["model"] = model_name
                    field_name = self.string_value(keywords.get("name")) if "model_name" in keywords else None
                    if field_name:
                        operation["field"] = field_name
                    if short == "RunPython":
                        for argument in positional[:2]:
                            function_name = self.dotted(argument)
                            if function_name:
                                run_python_functions.add(function_name)
                                operation.setdefault("functions", []).append(function_name)
                    if short in ("AddField", "AlterField", "RemoveField", "CreateModel", "DeleteModel", "RenameField", "RenameModel", "RunPython", "RunSQL"):
                        operations.append(operation)
            stack.extend(node.named_children)
        parts = self.path.split("/")
        app = parts[-3] if len(parts) >= 3 else ""
        self.result.migration = {"app": app, "name": parts[-1][:-3], "operations": sorted(operations, key=lambda item: item["line"])}
        self.result.data_migration_functions = run_python_functions
        for symbol in self.result.symbols:
            if symbol.name in run_python_functions:
                symbol.tags.add("data_migration")


def parse_python(path: str, source: bytes, module: str, is_test_file: bool, is_migration: bool) -> ParsedPython:
    extractor = PythonExtractor(path, source, module, is_test_file, is_migration)
    return extractor.run()
