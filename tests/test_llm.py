from code_review_agent.index.enrich import Enricher, SummaryBatch
from code_review_agent.llm.client import LLMClient
from code_review_agent.retrieval.engine import RetrievalOptions
from code_review_agent.retrieval.models import SearchFilters

from .fake_openai import FakeOpenAI

SECRET = "sk-proj-abcdefghijklmnopqrstuvwxyz0123456789"


def make_client(plain_settings, fake, retries=4):
    settings = plain_settings.model_copy(deep=True)
    settings.openai_api_key = "sk-test-key-000000000000000000000"
    settings.models.max_retries = retries
    return LLMClient(settings, http_client=fake.http_client())


def test_egress_is_scrubbed_and_usage_is_costed(plain_settings):
    fake = FakeOpenAI()
    client = make_client(plain_settings, fake)
    prompt = f'<chunk id="abc" path="settings.py" symbol="settings" kind="config">\nKEY = "{SECRET}"\n</chunk>'
    result = client.structured(SummaryBatch, "summarize", prompt, purpose="test")
    assert result.items[0].id == "abc"
    sent = fake.requests[0]["body"]
    assert SECRET not in sent["input"] and "«REDACTED" in sent["input"]
    assert sent["reasoning"] == {"effort": "minimal"} and sent["store"] is False
    assert client.totals.calls == 1 and client.totals.cached_tokens == 10 and client.totals.cost_usd > 0


def test_fallback_model_on_server_error(plain_settings):
    fake = FakeOpenAI(fail_models={"gpt-5-mini"})
    client = make_client(plain_settings, fake, retries=0)
    result = client.structured(SummaryBatch, "summarize", '<chunk id="x" path="a.py" symbol="a" kind="function">\npass\n</chunk>', purpose="test")
    assert result.items[0].id == "x"
    models = [request["body"]["model"] for request in fake.requests]
    assert models == ["gpt-5-mini", "gpt-4.1-mini"]
    assert "reasoning" not in fake.requests[1]["body"]


def test_embeddings_scrub_and_keep_order(plain_settings):
    fake = FakeOpenAI()
    client = make_client(plain_settings, fake)
    vectors = client.embed(["refund amount", f"token {SECRET}"], purpose="test", dims=16)
    assert len(vectors) == 2 and len(vectors[0]) == 16
    assert SECRET not in str(fake.requests[0]["body"]["input"])


def test_enrichment_enables_dense_and_rerank(llm_session, fake_openai):
    enricher = Enricher(llm_session.store, llm_session.llm, llm_session.settings)
    assert enricher.summarize(llm_session.view) > 10
    assert enricher.embed(llm_session.view) > 10
    assert enricher.summarize(llm_session.view) == 0
    assert enricher.embed(llm_session.view) == 0
    engine = llm_session.engine
    assert engine.dense_available()
    ranked, lists = engine.search("restocking fee percent", SearchFilters(kinds=["method"]), "dense", 3, rerank_results=False)
    symbols = [row["symbol"] for row in llm_session.view.chunks_for_row_ids([item.row_id for item in ranked])]
    assert "shop.services.refunds.RefundService.restocking_fee" in symbols
    wave = engine.first_wave(llm_session.diff, "bug_hunter", options=RetrievalOptions())
    assert any(ranked_list.source == "dense" for ranked_list in wave.lists)
    assert any(item.relevance is not None for item in wave.ranked)
    schema_names = {request["body"].get("text", {}).get("format", {}).get("name") for request in fake_openai.requests}
    assert {"SummaryBatch", "GistBatch", "RerankResult"} <= schema_names
    assert llm_session.store.query_one("SELECT count(*) AS n FROM usage")["n"] > 0
