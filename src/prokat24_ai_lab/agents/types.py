from collections.abc import Callable
from typing import TypedDict


class ToolCall(TypedDict):
    tool_name: str
    arguments: dict[str, str | int | bool]


class ToolExecution(TypedDict):
    tool_call: ToolCall
    output: str


class AgentStep(TypedDict):
    final_answer: str | None
    tool_call: ToolCall | None


class AgentRunResult(TypedDict):
    final_answer: str
    tool_calls: list[ToolCall]


ToolExecutor = Callable[[ToolCall], str]
AgentRunner = Callable[[str], AgentRunResult]
AgentStepGenerator = Callable[[str, list[ToolExecution]], AgentStep]
