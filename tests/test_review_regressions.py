import json
import os
import subprocess

from code_review_agent.evals.golden import mine_cases
from code_review_agent.git.repo import GitRepo
from code_review_agent.guardrails.secrets import scrub
from code_review_agent.index.builder import repo_cache_dir, settings_for_repo
from code_review_agent.index.python_parser import parse_python
from code_review_agent.session import ReviewSession
from code_review_agent.tools.executor import ToolCall, ToolExecutor
from code_review_agent.tools.impl import build_registry
from code_review_agent.util import loads


def git(root, *args):
    environment = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@example.com", GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@example.com")
    subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, env=environment)


def write_files(root, files):
    for path, text in files.items():
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)


def branch_session(tmp_path, base_files, branch_files, deleted=()):
    root = tmp_path / "repo"
    root.mkdir()
    write_files(root, base_files)
    git(root, "init", "-q", "-b", "main")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "base")
    git(root, "checkout", "-q", "-b", "feature")
    write_files(root, branch_files)
    for path in deleted:
        (root / path).unlink()
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "change")
    return ReviewSession.open(str(root), base="main", use_llm=False)


def tags_for(session, symbol):
    rows = session.view.chunk_rows("c.symbol = ?", (symbol,))
    derived = session.view.derived_tags([row["chunk_id"] for row in rows])
    return set(session.view.all_tags(rows[0], derived))


APP_FILES = {
    "app/__init__.py": "",
    "app/models.py": "from django.db import models\n\n\nclass Order(models.Model):\n    total = models.DecimalField(max_digits=9, decimal_places=2)\n",
    "app/handlers.py": "def on_order_saved(sender, instance, **kwargs):\n    return instance\n",
    "app/apps.py": "from django.db.models.signals import post_save\n\nfrom .handlers import on_order_saved\nfrom .models import Order\n\npost_save.connect(on_order_saved, sender=Order)\n",
    "app/util.py": "def helper_x():\n    return 1\n",
    "app/use.py": "from . import util\n\n\ndef use_it():\n    return util.helper_x() + util.later_fn()\n",
}


def test_receiver_registered_in_unchanged_file_survives_handler_change(tmp_path):
    session = branch_session(tmp_path, APP_FILES, {"app/handlers.py": "def on_order_saved(sender, instance, **kwargs):\n    return instance.pk\n"})
    model = session.view.chunk_rows("c.symbol = 'app.models.Order'")[0]
    receivers = session.view.edges_to([model["chunk_id"]], ["receives"])
    assert len(receivers) == 1
    assert "signal_receiver" in tags_for(session, "app.handlers.on_order_saved")


def test_callers_of_deleted_symbol_are_flagged_and_added_symbols_resolve(tmp_path):
    session = branch_session(tmp_path, APP_FILES, {"app/util.py": "def later_fn():\n    return 2\n"})
    use_it = session.view.chunk_rows("c.symbol = 'app.use.use_it'")[0]
    edges = session.view.edges_from([use_it["chunk_id"]], ["calls"])
    by_name = {edge["dst_name"]: edge["confidence"] for edge in edges}
    assert by_name["app.util.helper_x"] == "missing"
    assert by_name["app.util.later_fn"] == "exact"
    assert "references_missing_symbol" in tags_for(session, "app.use.use_it")


def test_repaired_missing_edge_does_not_keep_missing_tag(tmp_path):
    files = dict(APP_FILES)
    files["app/use.py"] = "from . import util\n\n\ndef use_it():\n    return util.later_fn()\n"
    session = branch_session(tmp_path, files, {"app/util.py": "def helper_x():\n    return 1\n\n\ndef later_fn():\n    return 2\n"})
    assert "references_missing_symbol" not in tags_for(session, "app.use.use_it")


def test_url_names_stay_linked_when_patterns_are_inserted(tmp_path):
    base = {
        "site/__init__.py": "",
        "site/views.py": "def a(request):\n    return 1\n\n\ndef b(request):\n    return 2\n",
        "site/urls.py": "from django.urls import path\n\nfrom . import views\n\nurlpatterns = [\n    path('b/', views.b, name='b'),\n]\n",
        "site/links.py": "from django.urls import reverse\n\n\ndef link_b():\n    return reverse('b')\n",
    }
    changed = {"site/urls.py": "from django.urls import path\n\nfrom . import views\n\nurlpatterns = [\n    path('a/', views.a, name='a'),\n    path('b/', views.b, name='b'),\n]\n"}
    session = branch_session(tmp_path, base, changed)
    link = session.view.chunk_rows("c.symbol = 'site.links.link_b'")[0]
    target_ids = [edge["dst_chunk_id"] for edge in session.view.edges_from([link["chunk_id"]], ["links_to"])]
    targets = session.view.chunks_for_ids(target_ids)
    assert [loads(row["extra"])["name"] for row in targets] == ["b"]


def test_mode_only_change_is_treated_as_unchanged(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    write_files(root, {"tool.py": "def run():\n    return 1\n", "other.py": "def x():\n    return 2\n"})
    git(root, "init", "-q", "-b", "main")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "base")
    git(root, "checkout", "-q", "-b", "feature")
    os.chmod(root / "tool.py", 0o755)
    write_files(root, {"other.py": "def x():\n    return 3\n"})
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "chmod")
    session = ReviewSession.open(str(root), base="main", use_llm=False)
    rows = session.view.chunk_rows("c.symbol = 'tool.run'")
    assert len(rows) == 1 and rows[0]["version"] == "base" and not rows[0]["superseded"]


def test_grep_and_diff_never_expose_secret_files(tmp_path):
    base = {"app.py": "x = 1\n", "certs/server.pem": "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASC\n-----END PRIVATE KEY-----\n"}
    session = branch_session(tmp_path, base, {".env": "DB_PASSWORD=hunter2hunter2\n", "app.py": "x = 2\n"})
    executor = ToolExecutor(session, build_registry(), session.settings.tools)
    result = executor.run(ToolCall(id="g", name="grep", arguments={"pattern": ".", "path_glob": None, "version": "current", "max_matches": 200}))
    assert "hunter2" not in result.content and "MIIEvQIB" not in result.content
    diff = executor.run(ToolCall(id="d", name="get_diff", arguments={"path": None, "offset": None}))
    assert "hunter2" not in diff.content
    assert all("hunter2" not in line for hunk in session.diff.hunks() for line in hunk.lines)


def test_scrubber_covers_pgp_unquoted_short_and_mixed_quotes():
    pgp = "-----BEGIN PGP PRIVATE KEY BLOCK-----\nlQOYBF0abcdefghijklmnop\n=abcd\n-----END PGP PRIVATE KEY BLOCK-----"
    assert "lQOYBF0" not in scrub(pgp)
    assert "k8Zq2vN9xLm4" not in scrub("SECRET_KEY=k8Zq2vN9xLm4Pw7t")
    assert "k8Zq2vN9xLm4" not in scrub("password: k8Zq2vN9xLm4Pw7t")
    assert "Passw0rd!" not in scrub('DB_PASSWORD = "Passw0rd!"')
    assert "it's-s3cr3t-v4lue" not in scrub('API_TOKEN = "it\'s-s3cr3t-v4lue"')
    literal = 'MARKER = "-----BEGIN RSA PRIVATE KEY-----"\nprint("hello world")\nvalue = compute(1, 2)'
    assert scrub(literal).endswith('print("hello world")\nvalue = compute(1, 2)')
    assert scrub("SECRET_KEY = os.environ['SECRET_KEY']") == "SECRET_KEY = os.environ['SECRET_KEY']"


def test_transaction_flags_for_nested_callbacks_and_as_targets():
    source = b'''
from django.db import transaction
from .tasks import send_email


def safe():
    with transaction.atomic():
        def notify():
            send_email.delay(1)
        transaction.on_commit(notify)


def aliased():
    with transaction.atomic() as tx:
        send_email.delay(2)
'''
    parsed = parse_python("app/x.py", source, "app.x", False, False)
    calls = {ref.line: ref.extra for ref in parsed.refs if ref.kind == "task_call"}
    assert calls[9]["in_atomic"] is False
    assert calls[15]["in_atomic"] is True
    references = [ref.target for ref in parsed.refs if ref.kind == "reference"]
    assert "notify" in references


def test_calls_behind_attribute_receivers_are_walked():
    source = b'''
def work(pk):
    Order.objects.get(pk=pk).lines.update(x=1)
    get_thing(pk).attr.do_work()
'''
    parsed = parse_python("app/x.py", source, "app.x", False, False)
    kinds = {(ref.kind, ref.target) for ref in parsed.refs}
    assert ("query", "Order") in kinds
    assert ("call", "get_thing") in kinds


def test_tool_timeouts_do_not_starve_later_calls(tmp_path):
    import time

    from pydantic import BaseModel

    from code_review_agent.config import ToolSettings
    from code_review_agent.tools.registry import ToolRegistry, ToolSpec

    class NoArgs(BaseModel):
        pass

    registry = ToolRegistry()
    registry.register(ToolSpec("hang", "hangs", NoArgs, lambda executor, arguments: time.sleep(3), timeout_seconds=0.1))
    registry.register(ToolSpec("quick", "fast", NoArgs, lambda executor, arguments: {"ok": True}, timeout_seconds=0.5))
    executor = ToolExecutor(None, registry, ToolSettings())
    for index in range(6):
        assert not executor.run(ToolCall(id=f"h{index}", name="hang")).ok
    assert executor.run(ToolCall(id="q", name="quick")).ok


def test_golden_labels_insert_only_test_edits_and_renames(tmp_path):
    root = tmp_path / "history"
    root.mkdir()
    write_files(
        root,
        {
            "app/__init__.py": "",
            "app/pricing.py": "def total(lines):\n    return sum(lines)\n",
            "app/views.py": "from .pricing import total\n\n\ndef checkout(lines):\n    return total(lines)\n",
            "app/tests/test_pricing.py": "from app.pricing import total\n\n\ndef test_total():\n    result = total([1, 2])\n    assert result == 3\n\n\ndef test_other():\n    assert True\n",
            "app/tests/test_old_name.py": "def test_legacy():\n    assert True\n",
        },
    )
    git(root, "init", "-q", "-b", "main")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "initial")
    write_files(
        root,
        {
            "app/pricing.py": "def total(lines):\n    return max(sum(lines), 0)\n",
            "app/views.py": "from .pricing import total\n\n\ndef checkout(lines):\n    return total(lines or [])\n",
            "app/tests/test_pricing.py": "from app.pricing import total\n\n\ndef test_total():\n    result = total([1, 2])\n    assert result >= 0\n    assert result == 3\n\n\ndef test_other():\n    assert True\n",
        },
    )
    git(root, "mv", "app/tests/test_old_name.py", "app/tests/test_new_name.py")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "Fix negative totals")
    repo = GitRepo(root)
    settings = settings_for_repo(repo, use_llm=False)
    case = mine_cases(repo, repo_cache_dir(repo, settings), limit=1)[0]
    relevant = {chunk.symbol for chunk in case.relevant}
    assert "app.tests.test_pricing.test_total" in relevant
    assert "app.tests.test_pricing.test_other" not in relevant
    assert case.view_overrides["app/tests/test_new_name.py"] is None
    assert case.view_overrides["app/tests/test_old_name.py"] is not None


def test_omitted_listing_has_no_duplicates(demo_session):
    wave = demo_session.engine.first_wave(demo_session.diff, "state_regression", budget_tokens=1500)
    refs = [item.ref for item in wave.pack.omitted]
    assert len(refs) == len(set(refs))
    included = {item.ref for item in wave.pack.included}
    assert not included & set(refs)


def test_blob_diff_rewrite_keeps_body_lines():
    from code_review_agent.git.repo import rewrite_blob_diff_paths

    raw = "diff --git a/x b/y\nindex 1..2\n--- a/x\n+++ b/y\n@@ -1,2 +1,2 @@\n--- sql comment\n+++ counter\n"
    rewritten = rewrite_blob_diff_paths(raw, "db/q.sql").splitlines()
    assert rewritten[2] == "--- a/db/q.sql" and rewritten[3] == "+++ b/db/q.sql"
    assert rewritten[5] == "--- sql comment" and rewritten[6] == "+++ counter"


def test_json_dump_of_tool_results_is_valid(demo_session):
    executor = ToolExecutor(demo_session, build_registry(), demo_session.settings.tools)
    result = executor.run(ToolCall(id="s", name="signal_receivers", arguments={"model": "Refund"}))
    assert json.loads(result.content)["receivers"]
