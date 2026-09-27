from __future__ import annotations

import os
import tomllib
from pathlib import Path

from pydantic import BaseModel, Field


class ModelPrice(BaseModel):
    input_per_million: float
    cached_input_per_million: float = 0.0
    output_per_million: float = 0.0


DEFAULT_PRICES = {
    "gpt-5": ModelPrice(input_per_million=1.25, cached_input_per_million=0.125, output_per_million=10.0),
    "gpt-5-mini": ModelPrice(input_per_million=0.25, cached_input_per_million=0.025, output_per_million=2.0),
    "gpt-4.1-mini": ModelPrice(input_per_million=0.40, cached_input_per_million=0.10, output_per_million=1.60),
    "text-embedding-3-large": ModelPrice(input_per_million=0.13),
    "text-embedding-3-small": ModelPrice(input_per_million=0.02),
}


class IndexSettings(BaseModel):
    exclude: list[str] = Field(default_factory=list)
    max_file_bytes: int = 4_000_000
    max_chunk_tokens: int = 1500
    target_chunk_tokens: int = 600
    small_template_tokens: int = 600
    boilerplate_family_min: int = 20
    parse_workers: int = 0


class DjangoSettings(BaseModel):
    tenant_models: list[str] = Field(default_factory=list)
    tenant_fields: list[str] = Field(default_factory=list)


class MoneySettings(BaseModel):
    identifier_pattern: str = (
        r"(amount|price|prices|total|subtotal|balance|fee|fees|refund|refunds|tax|taxes|cost|charge|"
        r"payment|payments|invoice|invoices|credit|credits|discount|payout|cents|currency|money)"
    )


class ModelSettings(BaseModel):
    mini_model: str = "gpt-5-mini"
    mini_reasoning_effort: str = "minimal"
    fallback_mini_model: str = "gpt-4.1-mini"
    embedding_model: str = "text-embedding-3-large"
    embedding_dims: int = 1024
    llm_concurrency: int = 8
    max_retries: int = 4
    timeout_seconds: float = 90.0
    summary_batch_size: int = 12
    embedding_batch_size: int = 256
    review_model: str = "gpt-5"
    review_reasoning_effort: str = "medium"


class RetrievalSettings(BaseModel):
    rrf_k: int = 60
    graph_weight: float = 2.0
    lexical_weight: float = 1.0
    dense_weight: float = 1.0
    per_retriever_limit: int = 50
    rerank_candidates: int = 40
    rerank_keep: int = 10
    dense_overfetch: int = 4
    max_hunk_identifiers: int = 25


class ToolSettings(BaseModel):
    default_timeout_seconds: float = 15.0
    max_output_chars: int = 12_000
    max_items: int = 40
    read_file_max_lines: int = 250


class AgentSettings(BaseModel):
    specialist_max_turns: int = 12
    verifier_max_turns: int = 16
    verifier_batch_size: int = 10
    max_review_cost_usd: float = 2.0
    diff_prompt_tokens: int = 6000


class Settings(BaseModel):
    cache_root: Path
    openai_api_key: str | None = None
    openai_base_url: str | None = None
    index: IndexSettings = Field(default_factory=IndexSettings)
    django: DjangoSettings = Field(default_factory=DjangoSettings)
    money: MoneySettings = Field(default_factory=MoneySettings)
    models: ModelSettings = Field(default_factory=ModelSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    tools: ToolSettings = Field(default_factory=ToolSettings)
    agent: AgentSettings = Field(default_factory=AgentSettings)
    prices: dict[str, ModelPrice] = Field(default_factory=lambda: dict(DEFAULT_PRICES))

    @property
    def llm_enabled(self) -> bool:
        return bool(self.openai_api_key)


def default_cache_root() -> Path:
    override = os.environ.get("CRA_CACHE_DIR")
    if override:
        return Path(override).expanduser()
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        return Path(xdg) / "code-review-agent"
    return Path.home() / ".cache" / "code-review-agent"


def user_config_path() -> Path:
    xdg = os.environ.get("XDG_CONFIG_HOME")
    if xdg:
        return Path(xdg) / "code-review-agent" / "config.toml"
    return Path.home() / ".config" / "code-review-agent" / "config.toml"


def merge_section(target: dict, section: dict) -> None:
    for key, value in section.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            merge_section(target[key], value)
        else:
            target[key] = value


def load_settings(repo_config_text: str | None = None, use_llm: bool = True) -> Settings:
    data: dict = {}
    config_file = user_config_path()
    if config_file.exists():
        merge_section(data, tomllib.loads(config_file.read_text()))
    if repo_config_text:
        repo_data = tomllib.loads(repo_config_text)
        allowed = {key: value for key, value in repo_data.items() if key in ("index", "django", "money")}
        merge_section(data, allowed)

    data["cache_root"] = default_cache_root()
    if use_llm:
        data["openai_api_key"] = os.environ.get("OPENAI_API_KEY")
        data["openai_base_url"] = os.environ.get("OPENAI_BASE_URL")

    models = data.setdefault("models", {})
    if os.environ.get("CRA_MINI_MODEL"):
        models["mini_model"] = os.environ["CRA_MINI_MODEL"]
    if os.environ.get("CRA_REVIEW_MODEL"):
        models["review_model"] = os.environ["CRA_REVIEW_MODEL"]
    if os.environ.get("CRA_EMBEDDING_MODEL"):
        models["embedding_model"] = os.environ["CRA_EMBEDDING_MODEL"]
    if os.environ.get("CRA_EMBEDDING_DIMS"):
        models["embedding_dims"] = int(os.environ["CRA_EMBEDDING_DIMS"])

    return Settings.model_validate(data)
