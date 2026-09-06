from openai.types.responses import FunctionToolParam, ToolChoiceOptions

from prokat24_ai_lab.agents.openai_agent_step import (
    AGENT_STEP_INSTRUCTIONS,
    build_agent_step_input,
    create_openai_agent_step_generator,
)
from prokat24_ai_lab.agents.openai_tool_call import ResponseProtocol
from prokat24_ai_lab.agents.tool_schemas import TOOLS
from prokat24_ai_lab.agents.types import ToolCall, ToolExecution


class FakeOutputItem:
    type = "function_call"
    name = "search_products"
    arguments = '{"query": "2-місний намет"}'


class FakeResponse:
    def __init__(self) -> None:
        self.output = [FakeOutputItem()]
        self.output_text = ""


class FakeResponseWithoutCall:
    def __init__(self) -> None:
        self.output = []
        self.output_text = "Привіт! Чим можу допомогти?"


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


def test_build_agent_step_input_includes_user_input_without_tool_executions() -> None:
    user_input = "Привіт!"
    tool_executions = []

    result = build_agent_step_input(user_input, tool_executions)

    assert "Привіт!" in result


def test_build_agent_step_input_includes_tool_execution_history() -> None:
    user_input = "Скільки буде коштувати 2-місний намет на 3 дні?"
    tool_executions = [
        ToolExecution(
            tool_call=ToolCall(
                tool_name="search_products",
                arguments={"query": "2-місний намет"},
            ),
            output="Намет 2-місний\nURL: https://example.com/tent/",
        )
    ]

    result = build_agent_step_input(user_input, tool_executions)

    assert "Скільки буде коштувати 2-місний намет на 3 дні?" in result
    assert "search_products" in result
    assert '"query": "2-місний намет"' in result
    assert "Намет 2-місний" in result
    assert "https://example.com/tent/" in result


def test_openai_agent_step_generator_returns_tool_call() -> None:
    fake_client = FakeClient(FakeResponse())

    generate_agent_step = create_openai_agent_step_generator(fake_client)

    result = generate_agent_step("Скільки буде коштувати 2-місний намет на 3 дні?", [])

    assert result == {
        "final_answer": None,
        "tool_call": {
            "tool_name": "search_products",
            "arguments": {"query": "2-місний намет"},
        },
    }
    assert fake_client.responses.model == "gpt-5-mini"
    assert fake_client.responses.input == (
        "User request:\nСкільки буде коштувати 2-місний намет на 3 дні?"
    )
    assert fake_client.responses.instructions == AGENT_STEP_INSTRUCTIONS
    assert fake_client.responses.tools == TOOLS
    assert fake_client.responses.tool_choice == "auto"


def test_openai_agent_step_generator_returns_final_answer() -> None:
    fake_client = FakeClient(FakeResponseWithoutCall())

    generate_agent_step = create_openai_agent_step_generator(fake_client)

    result = generate_agent_step("Скільки буде коштувати 2-місний намет на 3 дні?", [])

    assert result == {
        "final_answer": "Привіт! Чим можу допомогти?",
        "tool_call": None,
    }
