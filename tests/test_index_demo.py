from code_review_agent.util import loads


def symbols_with_tag(session, tag):
    rows = session.store.query("SELECT chunk_id FROM chunk_tags WHERE view_key = ? AND tag = ?", (session.view.view_key, tag))
    chunk_ids = [row["chunk_id"] for row in rows]
    return {row["symbol"] for row in session.view.chunks_for_ids(chunk_ids)}


def test_overlay_keeps_both_versions_of_changed_files(demo_session):
    view = demo_session.view
    assert view.changed_paths() == {"shop/services/refunds.py", "shop/views.py"}
    before = view.chunk_rows("c.symbol = ?", ("shop.services.refunds.RefundService.issue_refund",), version="base")
    after = view.chunk_rows("c.symbol = ?", ("shop.services.refunds.RefundService.issue_refund",), version="head")
    assert len(before) == 1 and len(after) == 1
    assert before[0]["chunk_id"] == after[0]["chunk_id"]
    assert before[0]["content_hash"] != after[0]["content_hash"]
    current = view.chunk_rows("c.symbol = ?", ("shop.services.refunds.RefundService.issue_refund",))
    assert [row["version"] for row in current] == ["head"]


def test_graph_edges_capture_framework_facts(demo_session):
    view = demo_session.view
    task = view.chunk_rows("c.symbol = 'shop.tasks.send_refund_email'")[0]
    enqueues = view.edges_to([task["chunk_id"]], ["enqueues"])
    assert len(enqueues) == 1
    assert loads(enqueues[0]["extra"])["in_atomic"] is True
    assert loads(enqueues[0]["extra"])["in_on_commit"] is False
    base_enqueues = view.edges_to([task["chunk_id"]], ["enqueues"], version="base")
    assert loads(base_enqueues[0]["extra"])["in_on_commit"] is True

    refund_model = view.chunk_rows("c.symbol = 'shop.models.Refund'")[0]
    receivers = view.edges_to([refund_model["chunk_id"]], ["receives"])
    assert loads(receivers[0]["extra"])["via"] == "connect"

    routes = {route["name"]: route["route"] for route in view.all_url_routes()}
    assert routes["shop:order_detail"] == "shop/orders/[^/]+/"


def test_derived_risk_tags_point_at_injected_bugs(demo_session):
    assert "shop.views.order_detail" in symbols_with_tag(demo_session, "unscoped_query")
    assert "shop.services.refunds.RefundService.issue_refund" in symbols_with_tag(demo_session, "task_in_atomic_without_on_commit")
    assert "shop.views.OrderListView.get_queryset" in symbols_with_tag(demo_session, "tenant_scoped")


def test_cache_keys_resolve_through_helper(demo_session):
    edges = demo_session.view.edge_rows("kind = 'cache'")
    assert {edge["dst_name"] for edge in edges} == {"cache:order_total:{}"}


def test_boilerplate_and_incremental_rebuild_reuse_parses(demo_repo):
    from code_review_agent.git.repo import GitRepo
    from code_review_agent.index.builder import IndexBuilder, settings_for_repo

    repo = GitRepo(demo_repo)
    builder = IndexBuilder(repo, settings_for_repo(repo, use_llm=False))
    head = repo.rev_parse("HEAD")
    base = repo.merge_base("main", head)
    _, first = builder.build_overlay(base, head)
    _, second = builder.build_overlay(base, head, force=True)
    assert second.parsed_now == 0
