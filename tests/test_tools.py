import json
import time

from pydantic import BaseModel

from code_review_agent.config import ToolSettings
from code_review_agent.tools.executor import ToolCall, ToolExecutor
from code_review_agent.tools.impl import build_registry
from code_review_agent.tools.registry import ToolRegistry, ToolSpec


def run(executor, tool_name, **arguments):
    return executor.run(ToolCall(id="t1", name=tool_name, arguments=arguments))


def test_path_guard_blocks_traversal_absolute_and_secrets(demo_session):
    executor = ToolExecutor(demo_session, build_registry(), demo_session.settings.tools)
    for bad in ("../outside.py", "/etc/passwd", ".env", "missing.py"):
        result = run(executor, "read_file", path=bad, start=None, end=None, version="current")
        assert not result.ok, bad
    good = run(executor, "read_file", path="shop/views.py", start=13, end=16, version="base")
    assert good.ok and "tenant=request.tenant" in good.content


def test_allowlist_and_argument_validation(demo_session):
    executor = ToolExecutor(demo_session, build_registry(), demo_session.settings.tools, allowlist=["find_symbol"])
    assert not run(executor, "grep", pattern="x", path_glob=None, version="current", max_matches=None).ok
    assert not run(executor, "find_symbol", name=5).ok
    assert run(executor, "find_symbol", name="Order", kind="model", version="current").ok


def test_timeouts_and_truncation():
    class NoArgs(BaseModel):
        pass

    registry = ToolRegistry()
    registry.register(ToolSpec("slow", "sleeps", NoArgs, lambda executor, arguments: time.sleep(2), timeout_seconds=0.2))
    registry.register(ToolSpec("huge", "big output", NoArgs, lambda executor, arguments: "x" * 5000, text_output=True, max_chars=1000))
    executor = ToolExecutor(None, registry, ToolSettings())
    slow = run(executor, "slow")
    assert not slow.ok and "timed out" in slow.error
    huge = run(executor, "huge")
    assert huge.ok and huge.truncated and len(huge.content) < 1200


def test_framework_tools_report_the_injected_race(demo_session):
    executor = ToolExecutor(demo_session, build_registry(), demo_session.settings.tools)
    sites = json.loads(run(executor, "task_call_sites", task="send_refund_email").content)["call_sites"]
    assert sites[0]["inside_atomic"] is True and sites[0]["deferred_with_on_commit"] is False
    receivers = json.loads(run(executor, "signal_receivers", model="Refund").content)["receivers"]
    assert receivers[0]["symbol"] == "shop.signals.refund_saved"
    routes = json.loads(run(executor, "route_for_view", symbol="order_detail", version="current").content)
    assert routes["routes"][0]["name"] == "shop:order_detail" and routes["access"][0]["auth_gate"]
    tests = json.loads(run(executor, "tests_for", symbol="shop.views.order_detail", version="current").content)["results"]
    assert tests[0]["symbol"].endswith("test_order_detail_requires_login")


def test_openai_tool_definitions_are_strict(demo_session):
    definitions = build_registry().openai_definitions()
    assert {definition["name"] for definition in definitions} >= {"get_diff", "callers", "semantic_search", "git_log"}
    for definition in definitions:
        parameters = definition["parameters"]
        assert definition["strict"] is True
        assert parameters["additionalProperties"] is False
        assert set(parameters["required"]) == set(parameters["properties"])
