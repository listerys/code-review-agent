from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

from pydantic import BaseModel, ValidationError

from ..llm.client import LLMClient
from ..tools.executor import ToolCall, ToolExecutor

OutputType = TypeVar("OutputType", bound=BaseModel)


class AgentError(RuntimeError):
    pass


@dataclass
class LoopResult(Generic[OutputType]):
    output: OutputType
    turns: int
    tool_calls: int
    stopped: str | None


def final_text(response) -> str:
    messages = [item for item in response.output if item.type == "message"]
    if not messages:
        raise AgentError(f"model returned no answer (status {response.status})")
    parts = []
    for content in messages[-1].content:
        if content.type == "refusal":
            raise AgentError(f"model refused: {content.refusal}")
        if content.type == "output_text":
            parts.append(content.text)
    return "".join(parts)


class AgentLoop:
    def __init__(
        self,
        llm: LLMClient,
        executor: ToolExecutor,
        tools: list[dict],
        over_budget: Callable[[], bool] = lambda: False,
        progress: Callable[[str], None] | None = None,
    ):
        self.llm = llm
        self.executor = executor
        self.tools = tools
        self.over_budget = over_budget
        self.progress = progress

    def run(self, name: str, instructions: str, prompt: str, schema: type[OutputType], max_turns: int) -> LoopResult[OutputType]:
        items: list[dict] = [{"role": "user", "content": prompt}]
        tool_calls = 0
        for turn in range(1, max_turns + 1):
            stopped = None
            if self.over_budget():
                stopped = "budget"
            elif turn == max_turns:
                stopped = "max_turns"
            response = self.llm.agent_turn(instructions, items, self.tools, schema, f"agent.{name}", allow_tools=stopped is None)
            calls = [item for item in response.output if item.type == "function_call"]
            if not calls:
                try:
                    output = schema.model_validate_json(final_text(response))
                except ValidationError as error:
                    raise AgentError(f"model answer did not match {schema.__name__}: {error.errors(include_url=False)[:3]}") from error
                return LoopResult(output=output, turns=turn, tool_calls=tool_calls, stopped=stopped)
            if stopped is not None:
                raise AgentError("model kept calling tools after tools were disabled")
            items.extend(item.to_dict(mode="json") for item in response.output)
            for call in calls:
                items.append({"type": "function_call_output", "call_id": call.call_id, "output": self.call_tool(name, call)})
            tool_calls += len(calls)
        raise AgentError("agent loop ended without an answer")

    def call_tool(self, name: str, call) -> str:
        try:
            arguments = json.loads(call.arguments or "{}")
        except json.JSONDecodeError as error:
            return f"error: arguments are not valid JSON ({error})"
        if self.progress is not None:
            self.progress(f"{name}: {call.name}({', '.join(f'{key}={value}' for key, value in arguments.items() if value is not None)[:120]})")
        result = self.executor.run(ToolCall(id=call.call_id, name=call.name, arguments=arguments))
        return result.content
