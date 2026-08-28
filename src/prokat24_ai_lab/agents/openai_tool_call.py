import json
from collections.abc import Sequence
from typing import Protocol, cast

from openai.types.responses import FunctionToolParam, ToolChoiceOptions

from prokat24_ai_lab.agents.tools import TOOLS
from prokat24_ai_lab.evals.tool_use import ToolCall, ToolCallGenerator


class OutputItemProtocol(Protocol):
    @property
    def type(self) -> str: ...


class ResponseProtocol(Protocol):
    @property
    def output(self) -> Sequence[OutputItemProtocol]: ...


class FunctionCallProtocol(OutputItemProtocol, Protocol):
    @property
    def name(self) -> str: ...

    @property
    def arguments(self) -> str: ...


class ResponsesProtocol(Protocol):
    def create(
        self,
        *,
        model: str,
        input: str,
        tools: list[FunctionToolParam],
        tool_choice: ToolChoiceOptions,
    ) -> ResponseProtocol: ...


class ToolCallClientProtocol(Protocol):
    @property
    def responses(self) -> ResponsesProtocol: ...


OPENAI_TOOL_CALL_MODEL = "gpt-5-mini"


def create_openai_tool_call_generator(
    client: ToolCallClientProtocol,
    model: str = OPENAI_TOOL_CALL_MODEL,
) -> ToolCallGenerator:
    def generate_tool_call(user_input: str) -> ToolCall | None:
        response = client.responses.create(
            model=model,
            input=user_input,
            tools=TOOLS,
            tool_choice="auto",
        )

        for output_item in response.output:
            if output_item.type == "function_call":
                function_call = cast(FunctionCallProtocol, output_item)
                tool_name = function_call.name
                arguments = json.loads(function_call.arguments)

                return ToolCall(
                    tool_name=tool_name,
                    arguments=arguments,
                )

        return None

    return generate_tool_call
