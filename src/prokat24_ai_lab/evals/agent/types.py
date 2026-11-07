from typing import TypedDict

from prokat24_ai_lab.agents.types import ToolCall, ToolExecution


class AgentOutcomeEvalResult(TypedDict):
    id: str
    trace_id: str | None
    user_input: str
    final_answer: str
    missing_fragments: list[str]
    unexpected_fragments: list[str]
    tool_calls: list[ToolCall]
    missing_tool_names: list[str]
    tool_executions: list[ToolExecution]
    passed: bool


class AgentCriterionResult(TypedDict):
    name: str
    passed: bool
    reason: str


class AgentCriterionEvalResult(TypedDict):
    user_input: str
    final_answer: str
    passed: bool
    failed_criteria: list[str]
    criteria: list[AgentCriterionResult]
