import pytest

from prokat24_ai_lab.agents.agent_loop import run_agent
from prokat24_ai_lab.agents.types import AgentStep, ToolCall, ToolExecution


def test_run_agent_returns_final_answer_without_tool_calls() -> None:
    user_input = "Привіт!"

    def fake_agent_step_generator(
        user_input: str, _tool_executions: list[ToolExecution]
    ) -> AgentStep:
        return AgentStep(
            final_answer=user_input,
            tool_call=None,
        )

    def fake_tool_executor(_tool_call: ToolCall) -> str:
        raise AssertionError("Tool executor should not be called")

    result = run_agent(
        user_input,
        fake_agent_step_generator,
        fake_tool_executor,
    )

    assert result["final_answer"] == "Привіт!"
    assert result["tool_calls"] == []


def test_run_agent_executes_tool_and_returns_final_answer() -> None:
    user_input = "Знайди намет"

    def fake_agent_step_generator(
        user_input: str, tool_executions: list[ToolExecution]
    ) -> AgentStep:
        if not tool_executions:
            return AgentStep(
                final_answer=None,
                tool_call=ToolCall(
                    tool_name="search_products",
                    arguments={"query": "намет"},
                ),
            )

        return AgentStep(
            final_answer="Знайдено: Намет 2-місний",
            tool_call=None,
        )

    def fake_tool_executor(tool_call: ToolCall) -> str:
        return "Намет 2-місний"

    result = run_agent(
        user_input,
        fake_agent_step_generator,
        fake_tool_executor,
    )

    assert result["final_answer"] == "Знайдено: Намет 2-місний"
    assert result["tool_calls"] == [
        {
            "tool_name": "search_products",
            "arguments": {"query": "намет"},
        }
    ]


def test_run_agent_executes_multiple_tools_before_final_answer() -> None:
    user_input = "Скільки буде коштувати 2-місний намет на 3 дні?"

    def fake_agent_step_generator(
        user_input: str, tool_executions: list[ToolExecution]
    ) -> AgentStep:
        if not tool_executions:
            return AgentStep(
                final_answer=None,
                tool_call=ToolCall(
                    tool_name="search_products",
                    arguments={"query": "2-місний намет"},
                ),
            )

        if len(tool_executions) == 1:
            return AgentStep(
                final_answer=None,
                tool_call=ToolCall(
                    tool_name="get_product_price",
                    arguments={
                        "product_url": "https://example.com/tent/",
                        "days": 3,
                    },
                ),
            )

        return AgentStep(
            final_answer="Оренда 2-місного намету на 3 дні коштує 900 грн",
            tool_call=None,
        )

    def fake_tool_executor(tool_call: ToolCall) -> str:
        if tool_call["tool_name"] == "search_products":
            return "https://example.com/tent/"

        if tool_call["tool_name"] == "get_product_price":
            return "Оренда 2-місного намету на 3 дні коштує 900 грн"

        raise AssertionError(f"Unexpected tool: {tool_call['tool_name']}")

    result = run_agent(
        user_input,
        fake_agent_step_generator,
        fake_tool_executor,
    )

    assert result["final_answer"] == "Оренда 2-місного намету на 3 дні коштує 900 грн"
    assert result["tool_calls"] == [
        {
            "tool_name": "search_products",
            "arguments": {"query": "2-місний намет"},
        },
        {
            "tool_name": "get_product_price",
            "arguments": {
                "product_url": "https://example.com/tent/",
                "days": 3,
            },
        },
    ]


def test_run_agent_raises_when_max_steps_exceeded() -> None:
    user_input = "Знайди намет"

    def fake_agent_step_generator(
        user_input: str, tool_executions: list[ToolExecution]
    ) -> AgentStep:
        return AgentStep(
            final_answer=None,
            tool_call=ToolCall(
                tool_name="search_products",
                arguments={"query": "намет"},
            ),
        )

    def fake_tool_executor(tool_call: ToolCall) -> str:
        return "Намет 2-місний"

    with pytest.raises(ValueError, match="Agent exceeded maximum number of steps"):
        run_agent(
            user_input,
            fake_agent_step_generator,
            fake_tool_executor,
            max_steps=3,
        )
