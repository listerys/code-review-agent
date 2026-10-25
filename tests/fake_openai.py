from __future__ import annotations

import hashlib
import itertools
import json
import math
import re

import httpx2

WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def hashed_embedding(text: str, dims: int) -> list[float]:
    vector = [0.0] * dims
    for word in WORD.findall(text.lower()):
        for piece in word.split("_"):
            if len(piece) < 3:
                continue
            bucket = int(hashlib.md5(piece.encode()).hexdigest(), 16) % dims
            vector[bucket] += 1.0
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


class FakeOpenAI:
    def __init__(self, fail_models: set[str] | None = None):
        self.requests: list[dict] = []
        self.fail_models = fail_models or set()
        self.agent_policy = None
        self.turn_numbers = itertools.count(1)

    def usage(self, input_tokens: int = 100, output_tokens: int = 20) -> dict:
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "input_tokens_details": {"cached_tokens": 10},
            "output_tokens_details": {"reasoning_tokens": 0},
        }

    def message(self, payload: dict) -> dict:
        return {
            "type": "message",
            "id": "msg_test",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": json.dumps(payload), "annotations": []}],
        }

    def agent_output(self, body: dict) -> list[dict]:
        turn = next(self.turn_numbers)
        decision = self.agent_policy(body)
        output = [{"type": "reasoning", "id": f"rs_{turn}", "summary": [], "encrypted_content": f"encrypted-{turn}"}]
        if "calls" not in decision:
            return output + [self.message(decision["final"])]
        for index, (name, arguments) in enumerate(decision["calls"]):
            output.append(
                {
                    "type": "function_call",
                    "id": f"fc_{turn}_{index}",
                    "call_id": f"call_{turn}_{index}",
                    "name": name,
                    "arguments": json.dumps(arguments),
                    "status": "completed",
                }
            )
        return output

    def response(self, model: str, payload: dict, output: list[dict] | None = None) -> dict:
        return {
            "id": "resp_test",
            "object": "response",
            "created_at": 1,
            "status": "completed",
            "model": model,
            "output": output if output is not None else [self.message(payload)],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            "usage": self.usage(),
        }

    def structured_payload(self, schema_name: str, prompt: str) -> dict:
        if schema_name == "SummaryBatch":
            items = []
            for match in re.finditer(r'<chunk id="([^"]+)" path="([^"]+)" symbol="([^"]+)"', prompt):
                items.append({"id": match.group(1), "summary": f"Summary of {match.group(3)} in {match.group(2)}"})
            return {"items": items}
        if schema_name == "GistBatch":
            items = []
            for match in re.finditer(r'<hunk id="([^"]+)" path="([^"]+)">\n(.*?)</hunk>', prompt, re.S):
                words = " ".join(WORD.findall(match.group(3))[:30])
                items.append({"id": match.group(1), "gist": f"change in {match.group(2)} touching {words}"})
            return {"items": items}
        if schema_name == "RerankResult":
            ids = re.findall(r'<candidate id="([^"]+)">', prompt)
            ranking = []
            for position, identifier in enumerate(reversed(ids)):
                ranking.append({"id": identifier, "relevance": 3 if position < 3 else 1})
            return {"ranking": ranking}
        return {}

    def handler(self, request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        self.requests.append({"path": request.url.path, "body": body})
        model = body.get("model", "")
        if model in self.fail_models:
            return httpx2.Response(500, json={"error": {"message": "boom", "type": "server_error"}})
        if request.url.path.endswith("/embeddings"):
            inputs = body["input"] if isinstance(body["input"], list) else [body["input"]]
            dims = body.get("dimensions", 64)
            data = []
            for index, text in enumerate(inputs):
                data.append({"object": "embedding", "index": index, "embedding": hashed_embedding(text, dims)})
            return httpx2.Response(
                200,
                json={"object": "list", "data": data, "model": model, "usage": {"prompt_tokens": 10 * len(inputs), "total_tokens": 10 * len(inputs)}},
            )
        if request.url.path.endswith("/responses") and "tools" in body:
            return httpx2.Response(200, json=self.response(f"{model}-2026-01-01", {}, self.agent_output(body)))
        if request.url.path.endswith("/responses"):
            schema_name = body["text"]["format"]["name"]
            payload = self.structured_payload(schema_name, body["input"])
            return httpx2.Response(200, json=self.response(f"{model}-2026-01-01", payload))
        return httpx2.Response(404, json={"error": {"message": "not found"}})

    def http_client(self) -> httpx2.Client:
        return httpx2.Client(transport=httpx2.MockTransport(self.handler))
