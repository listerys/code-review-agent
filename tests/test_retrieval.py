from code_review_agent.retrieval.fusion import reciprocal_rank_fusion
from code_review_agent.retrieval.models import Hit, RankedList, SearchFilters
from code_review_agent.retrieval.queries import changed_chunks


def test_rrf_rewards_agreement_and_weights():
    lexical = RankedList("lexical", 1.0, [Hit(1, "a", "lexical", 1), Hit(2, "b", "lexical", 2)])
    graph = RankedList("graph", 2.0, [Hit(2, "b", "graph", 1)])
    fused = reciprocal_rank_fusion([lexical, graph], 60)
    assert fused[0].row_id == 2
    assert fused[0].sources == ["lexical", "graph"]
    assert abs(fused[0].score - (1 / 62 + 2 / 61)) < 1e-9


def test_changed_chunks_pair_before_and_after(demo_session):
    changed = changed_chunks(demo_session.view, demo_session.diff)
    pairs = {(item.change, item.row["symbol"]) for item in changed}
    assert ("modified", "shop.views.order_detail") in pairs
    assert ("before", "shop.views.order_detail") in pairs
    assert ("modified", "shop.services.refunds.RefundService.calculate_refund") in pairs


def test_lexical_search_splits_identifiers(demo_session):
    ranked, _ = demo_session.engine.search("calculate refund", SearchFilters(kinds=["method"]), "lexical", 3)
    symbols = [row["symbol"] for row in demo_session.view.chunks_for_row_ids([item.row_id for item in ranked])]
    assert "shop.services.refunds.RefundService.calculate_refund" in symbols


def test_state_profile_retrieves_receivers_and_task(demo_session):
    wave = demo_session.engine.first_wave(demo_session.diff, "state_regression")
    facts = {item.symbol for item in wave.pack.included if item.role == "fact"}
    assert {"shop.signals.refund_saved", "shop.signals.order_saved"} <= facts
    included = {item.symbol for item in wave.pack.included}
    assert "shop.tasks.send_refund_email" in included
    pinned_roles = {item.role for item in wave.pack.included}
    assert {"changed", "before"} <= pinned_roles
    assert 'trust="untrusted"' in wave.pack.text


def test_security_profile_retrieves_sibling_tenant_scoped_queryset(demo_session):
    wave = demo_session.engine.first_wave(demo_session.diff, "security_integrity")
    retrieved = {item.symbol for item in wave.pack.included}
    assert "shop.views.OrderListView.get_queryset" in retrieved
    assert "shop.urls.urlpatterns[order_detail]" in retrieved


def test_pack_respects_budget_and_lists_omitted(demo_session):
    wave = demo_session.engine.first_wave(demo_session.diff, "impact_mapper", budget_tokens=2500)
    assert wave.pack.tokens <= 2500
    assert wave.pack.omitted
    assert "retrieved_not_shown" in wave.pack.text
