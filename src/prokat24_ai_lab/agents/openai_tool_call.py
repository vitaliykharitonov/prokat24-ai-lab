import json
from collections.abc import Sequence
from typing import Protocol, cast

from openai.types.responses import FunctionToolParam, ToolChoiceOptions

from prokat24_ai_lab.agents.types import ToolCall, ToolCallGenerator


class OutputItemProtocol(Protocol):
    @property
    def type(self) -> str: ...


class InputTokensDetailsProtocol(Protocol):
    @property
    def cached_tokens(self) -> int: ...


class OutputTokensDetailsProtocol(Protocol):
    @property
    def reasoning_tokens(self) -> int: ...


class ResponseUsageProtocol(Protocol):
    @property
    def input_tokens(self) -> int: ...

    @property
    def input_tokens_details(self) -> InputTokensDetailsProtocol: ...

    @property
    def output_tokens(self) -> int: ...

    @property
    def output_tokens_details(self) -> OutputTokensDetailsProtocol: ...

    @property
    def total_tokens(self) -> int: ...


class ResponseProtocol(Protocol):
    @property
    def output(self) -> Sequence[OutputItemProtocol]: ...

    @property
    def output_text(self) -> str: ...

    @property
    def usage(self) -> ResponseUsageProtocol | None: ...


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
        instructions: str,
        tools: list[FunctionToolParam],
        tool_choice: ToolChoiceOptions,
    ) -> ResponseProtocol: ...


class ToolCallClientProtocol(Protocol):
    @property
    def responses(self) -> ResponsesProtocol: ...


OPENAI_TOOL_CALL_MODEL = "gpt-5-mini"

TOOL_CALL_INSTRUCTIONS = (
    "For requests about delivery, product search or product information, "
    "and rental prices, always call the appropriate available tool. "
    "Do not answer these requests from your own knowledge. "
    "Call only the tool that matches the user's request. "
    "Do not call tools for greetings or general questions about your capabilities."
)


def create_openai_tool_call_generator(
    client: ToolCallClientProtocol,
    tools: list[FunctionToolParam],
    model: str = OPENAI_TOOL_CALL_MODEL,
) -> ToolCallGenerator:
    def generate_tool_call(user_input: str) -> ToolCall | None:
        response = client.responses.create(
            model=model,
            input=user_input,
            instructions=TOOL_CALL_INSTRUCTIONS,
            tools=tools,
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
