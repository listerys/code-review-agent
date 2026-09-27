from __future__ import annotations

import io
import json
import threading
from dataclasses import dataclass, field
from typing import TypeVar

import openai
from openai import OpenAI
from openai.lib._parsing._responses import type_to_text_format_param
from pydantic import BaseModel

from ..config import ModelPrice, Settings
from ..guardrails.secrets import scrub
from ..util import truncate_to_tokens

SchemaType = TypeVar("SchemaType", bound=BaseModel)

REASONING_MODEL_PREFIXES = ("gpt-5", "o1", "o3", "o4")
EMBEDDING_MAX_INPUT_TOKENS = 8000


class LLMUnavailable(RuntimeError):
    pass


@dataclass
class UsageTotals:
    calls: int = 0
    input_tokens: int = 0
    cached_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    by_purpose: dict[str, float] = field(default_factory=dict)


def price_for(model: str, prices: dict[str, ModelPrice]) -> ModelPrice | None:
    if model in prices:
        return prices[model]
    best_match = None
    for name, price in prices.items():
        if model.startswith(name) and (best_match is None or len(name) > len(best_match[0])):
            best_match = (name, price)
    return best_match[1] if best_match else None


def is_reasoning_model(model: str) -> bool:
    return model.startswith(REASONING_MODEL_PREFIXES)


def strict_text_format(schema: type[BaseModel]) -> dict:
    return type_to_text_format_param(schema)


def scrub_input_item(item: dict) -> dict:
    if item.get("type") == "function_call_output":
        return dict(item, output=scrub(item["output"]))
    if item.get("role") == "user" and isinstance(item.get("content"), str):
        return dict(item, content=scrub(item["content"]))
    return item


class LLMClient:
    def __init__(self, settings: Settings, store=None, http_client=None):
        if not settings.openai_api_key:
            raise LLMUnavailable("OPENAI_API_KEY is not set")
        self.settings = settings
        self.store = store
        self.lock = threading.Lock()
        self.totals = UsageTotals()
        arguments = {
            "api_key": settings.openai_api_key,
            "max_retries": settings.models.max_retries,
            "timeout": settings.models.timeout_seconds,
        }
        if settings.openai_base_url:
            arguments["base_url"] = settings.openai_base_url
        if http_client is not None:
            arguments["http_client"] = http_client
        self.client = OpenAI(**arguments)

    def record(self, purpose: str, model: str, input_tokens: int, cached_tokens: int, output_tokens: int) -> None:
        price = price_for(model, self.settings.prices)
        cost = 0.0
        if price is not None:
            uncached = max(input_tokens - cached_tokens, 0)
            cost = (
                uncached * price.input_per_million
                + cached_tokens * price.cached_input_per_million
                + output_tokens * price.output_per_million
            ) / 1_000_000
        with self.lock:
            self.totals.calls += 1
            self.totals.input_tokens += input_tokens
            self.totals.cached_tokens += cached_tokens
            self.totals.output_tokens += output_tokens
            self.totals.cost_usd += cost
            self.totals.by_purpose[purpose] = self.totals.by_purpose.get(purpose, 0.0) + cost
        if self.store is not None:
            self.store.record_usage(purpose, model, input_tokens, cached_tokens, output_tokens, cost)

    def embed(self, texts: list[str], purpose: str, model: str | None = None, dims: int | None = None) -> list[list[float]]:
        model = model or self.settings.models.embedding_model
        dims = dims or self.settings.models.embedding_dims
        prepared = []
        for text in texts:
            cleaned = scrub(text) or " "
            prepared.append(truncate_to_tokens(cleaned, EMBEDDING_MAX_INPUT_TOKENS))
        response = self.client.embeddings.create(model=model, input=prepared, dimensions=dims)
        vectors: list[list[float]] = [[] for _ in prepared]
        for item in response.data:
            vectors[item.index] = list(item.embedding)
        usage = response.usage
        self.record(purpose, response.model or model, usage.prompt_tokens if usage else 0, 0, 0)
        return vectors

    def request_arguments(self, model: str, instructions: str, prompt: str, reasoning_effort: str | None, max_output_tokens: int | None) -> dict:
        arguments = {
            "model": model,
            "instructions": scrub(instructions),
            "input": scrub(prompt),
            "store": False,
        }
        if is_reasoning_model(model):
            arguments["reasoning"] = {"effort": reasoning_effort or self.settings.models.mini_reasoning_effort}
        if max_output_tokens:
            arguments["max_output_tokens"] = max_output_tokens
        return arguments

    def structured(
        self,
        schema: type[SchemaType],
        instructions: str,
        prompt: str,
        purpose: str,
        model: str | None = None,
        reasoning_effort: str | None = None,
        max_output_tokens: int | None = None,
    ) -> SchemaType:
        primary = model or self.settings.models.mini_model
        fallback = self.settings.models.fallback_mini_model
        try:
            return self.structured_once(schema, primary, instructions, prompt, purpose, reasoning_effort, max_output_tokens)
        except (openai.BadRequestError, openai.AuthenticationError, openai.PermissionDeniedError):
            raise
        except (openai.APIError, ValueError):
            if not fallback or fallback == primary:
                raise
            return self.structured_once(schema, fallback, instructions, prompt, purpose, reasoning_effort, max_output_tokens)

    def structured_once(
        self,
        schema: type[SchemaType],
        model: str,
        instructions: str,
        prompt: str,
        purpose: str,
        reasoning_effort: str | None,
        max_output_tokens: int | None,
    ) -> SchemaType:
        arguments = self.request_arguments(model, instructions, prompt, reasoning_effort, max_output_tokens)
        response = self.client.responses.parse(text_format=schema, **arguments)
        usage = response.usage
        cached = 0
        if usage is not None and usage.input_tokens_details is not None:
            cached = usage.input_tokens_details.cached_tokens or 0
        self.record(
            purpose,
            response.model or model,
            usage.input_tokens if usage else 0,
            cached,
            usage.output_tokens if usage else 0,
        )
        parsed = response.output_parsed
        if parsed is None:
            raise ValueError("model returned no structured output")
        return parsed

    def agent_turn(self, instructions: str, items: list[dict], tools: list[dict], schema: type[BaseModel], purpose: str, allow_tools: bool = True):
        model = self.settings.models.review_model
        arguments = {
            "model": model,
            "instructions": scrub(instructions),
            "input": [scrub_input_item(item) for item in items],
            "tools": tools,
            "tool_choice": "auto" if allow_tools else "none",
            "text": {"format": strict_text_format(schema)},
            "store": False,
        }
        if is_reasoning_model(model):
            arguments["reasoning"] = {"effort": self.settings.models.review_reasoning_effort}
            arguments["include"] = ["reasoning.encrypted_content"]
        response = self.client.responses.create(**arguments)
        usage = response.usage
        cached = 0
        if usage is not None and usage.input_tokens_details is not None:
            cached = usage.input_tokens_details.cached_tokens or 0
        self.record(purpose, response.model or model, usage.input_tokens if usage else 0, cached, usage.output_tokens if usage else 0)
        return response

    def structured_request_body(self, schema: type[BaseModel], instructions: str, prompt: str, model: str | None = None) -> dict:
        model = model or self.settings.models.mini_model
        body = self.request_arguments(model, instructions, prompt, None, None)
        body["text"] = {"format": strict_text_format(schema)}
        return body

    def embedding_request_body(self, text: str, model: str | None = None, dims: int | None = None) -> dict:
        cleaned = truncate_to_tokens(scrub(text) or " ", EMBEDDING_MAX_INPUT_TOKENS)
        return {
            "model": model or self.settings.models.embedding_model,
            "input": cleaned,
            "dimensions": dims or self.settings.models.embedding_dims,
        }

    def submit_batch(self, endpoint: str, requests: list[tuple[str, dict]]) -> str:
        lines = []
        for custom_id, body in requests:
            lines.append(json.dumps({"custom_id": custom_id, "method": "POST", "url": endpoint, "body": body}))
        payload = io.BytesIO(("\n".join(lines) + "\n").encode())
        uploaded = self.client.files.create(file=("batch.jsonl", payload), purpose="batch")
        batch = self.client.batches.create(completion_window="24h", endpoint=endpoint, input_file_id=uploaded.id)
        return batch.id

    def batch_status(self, batch_id: str) -> tuple[str, str | None]:
        batch = self.client.batches.retrieve(batch_id)
        return batch.status, batch.output_file_id

    def batch_results(self, output_file_id: str) -> list[dict]:
        content = self.client.files.content(output_file_id).text
        results = []
        for line in content.splitlines():
            if line.strip():
                results.append(json.loads(line))
        return results


def optional_client(settings: Settings, store=None) -> LLMClient | None:
    if not settings.llm_enabled:
        return None
    return LLMClient(settings, store)
