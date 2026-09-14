from collections.abc import Callable
from typing import Literal, TypedDict

ToolArgumentValue = str | int | bool
ToolArguments = dict[str, ToolArgumentValue]


class ToolProperty(TypedDict):
    type: Literal["string", "integer", "boolean"]
    description: str


class ToolParameters(TypedDict):
    type: Literal["object"]
    properties: dict[str, ToolProperty]
    required: list[str]


class ToolDefinition(TypedDict):
    name: str
    description: str
    parameters: ToolParameters


class ToolCall(TypedDict):
    tool_name: str
    arguments: ToolArguments


class ToolExecution(TypedDict):
    tool_call: ToolCall
    output: str


ToolHandler = Callable[[ToolArguments], str]


class RegisteredTool(TypedDict):
    definition: ToolDefinition
    handler: ToolHandler


ToolRegistry = dict[str, RegisteredTool]


class AgentStep(TypedDict):
    final_answer: str | None
    tool_call: ToolCall | None


class AgentRunResult(TypedDict):
    final_answer: str
    tool_calls: list[ToolCall]


ToolExecutor = Callable[[ToolCall], str]
ToolCallGenerator = Callable[[str], ToolCall | None]
AgentRunner = Callable[[str], AgentRunResult]
AgentStepGenerator = Callable[[str, list[ToolExecution]], AgentStep]
