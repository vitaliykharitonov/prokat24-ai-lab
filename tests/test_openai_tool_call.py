from openai.types.responses import FunctionToolParam, ToolChoiceOptions

from prokat24_ai_lab.agents.openai_tool_call import (
    TOOL_CALL_INSTRUCTIONS,
    ResponseProtocol,
    create_openai_tool_call_generator,
)


class FakeOutputItem:
    type = "function_call"
    name = "search_products"
    arguments = '{"query": "намет"}'


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

    generate_tool_call = create_openai_tool_call_generator(fake_client, tools=[])

    result = generate_tool_call("Знайди намет")

    assert fake_client.responses.model == "gpt-5-mini"
    assert fake_client.responses.input == "Знайди намет"
    assert fake_client.responses.instructions == TOOL_CALL_INSTRUCTIONS
    assert fake_client.responses.tools == []
    assert fake_client.responses.tool_choice == "auto"

    assert result == {
        "tool_name": "search_products",
        "arguments": {"query": "намет"},
    }


def test_openai_tool_call_generator_returns_none_when_tool_call_is_missing() -> None:
    fake_client = FakeClient(FakeResponseWithoutToolCall())

    generate_tool_call = create_openai_tool_call_generator(fake_client, tools=[])

    result = generate_tool_call("Привіт!")

    assert result is None
