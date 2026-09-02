from openai.types.responses import FunctionToolParam, ToolChoiceOptions

from prokat24_ai_lab.agents.openai_tool_call import (
    TOOL_CALL_INSTRUCTIONS,
    ResponseProtocol,
    create_openai_tool_call_generator,
)
from prokat24_ai_lab.agents.tool_schemas import TOOLS


class FakeOutputItem:
    type = "function_call"
    name = "get_delivery_info"
    arguments = '{"location": "Київ"}'


class FakeMessageOutputItem:
    type = "message"


class FakeResponse:
    def __init__(self) -> None:
        self.output = [FakeOutputItem()]


class FakeResponseWithoutToolCall:
    def __init__(self) -> None:
        self.output = [FakeMessageOutputItem()]


class FakeResponses:
    def __init__(self, response: ResponseProtocol) -> None:
        self.response = response
        self.model: str | None = None
        self.input: str | None = None
        self.instructions: str | None = None
        self.tools: list[FunctionToolParam] | None = None
        self.tool_choice: ToolChoiceOptions | None = None

    def create(
        self,
        *,
        model: str,
        input: str,
        instructions: str,
        tools: list[FunctionToolParam],
        tool_choice: ToolChoiceOptions,
    ) -> ResponseProtocol:
        self.model = model
        self.tools = tools
        self.input = input
        self.instructions = instructions
        self.tool_choice = tool_choice
        return self.response


class FakeClient:
    def __init__(self, response: ResponseProtocol) -> None:
        self.responses = FakeResponses(response)


def test_openai_tool_call_generator_returns_tool_call() -> None:
    fake_client = FakeClient(FakeResponse())

    generate_tool_call = create_openai_tool_call_generator(fake_client)

    result = generate_tool_call("Скільки коштує доставка по Києву?")

    assert fake_client.responses.model == "gpt-5-mini"
    assert fake_client.responses.input == "Скільки коштує доставка по Києву?"
    assert fake_client.responses.instructions == TOOL_CALL_INSTRUCTIONS
    assert fake_client.responses.tools == TOOLS

    assert result == {
        "tool_name": "get_delivery_info",
        "arguments": {"location": "Київ"},
    }


def test_openai_tool_call_generator_returns_none_when_tool_call_is_missing() -> None:
    fake_client = FakeClient(FakeResponseWithoutToolCall())

    generate_tool_call = create_openai_tool_call_generator(fake_client)

    result = generate_tool_call("Скільки коштує доставка по Києву?")

    assert result is None
