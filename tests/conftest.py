from __future__ import annotations

import pytest

from code_review_agent.config import load_settings
from code_review_agent.demo import create_demo_repo
from code_review_agent.llm.client import LLMClient
from code_review_agent.session import ReviewSession

from .fake_openai import FakeOpenAI


@pytest.fixture(autouse=True)
def isolated_environment(tmp_path_factory, monkeypatch):
    monkeypatch.setenv("CRA_CACHE_DIR", str(tmp_path_factory.mktemp("cache")))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path_factory.mktemp("config")))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)


@pytest.fixture
def demo_repo(tmp_path):
    return create_demo_repo(tmp_path / "tinyshop")


@pytest.fixture
def demo_session(demo_repo):
    return ReviewSession.open(str(demo_repo), base="main", use_llm=False)


@pytest.fixture
def fake_openai():
    return FakeOpenAI()


@pytest.fixture
def llm_session(demo_repo, fake_openai, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-real-key-0000000000")
    monkeypatch.setenv("CRA_EMBEDDING_DIMS", "64")
    session = ReviewSession.open(str(demo_repo), base="main", use_llm=True)
    session._llm = LLMClient(session.settings, session.store, http_client=fake_openai.http_client())
    session._llm_checked = True
    return session


@pytest.fixture
def plain_settings():
    return load_settings(None, use_llm=False)
