from prokat24_ai_lab.agents.types import AgentStep, ToolCall, ToolExecution
from prokat24_ai_lab.observability.agent_step_instrumentation import (
    AgentStepExecutionRecord,
    create_instrumented_agent_step_generator,
)
from prokat24_ai_lab.observability.context import (
    current_sequence_number,
    current_trace_id,
)
from prokat24_ai_lab.observability.tool_instrumentation import (
    ToolExecutionRecord,
    create_instrumented_tool_executor,
)


def test_observability_records_share_trace_and_sequence() -> None:
    agent_step_records: list[AgentStepExecutionRecord] = []
    tool_records: list[ToolExecutionRecord] = []

    def fake_generator(
        _user_input: str,
        tool_executions: list[ToolExecution],
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
            final_answer="Знайдено намет.",
            tool_call=None,
        )

    def fake_executor(_tool_call: ToolCall) -> str:
        return "Намет 2-місний"

    instrumented_generator = create_instrumented_agent_step_generator(
        generator=fake_generator,
        on_record=agent_step_records.append,
    )

    instrumented_executor = create_instrumented_tool_executor(
        executor=fake_executor,
        on_record=tool_records.append,
    )

    trace_token = current_trace_id.set("test-trace")
    sequence_token = current_sequence_number.set(0)

    try:
        first_step = instrumented_generator(
            "Знайди намет",
            [],
        )

        tool_call = first_step["tool_call"]
        assert tool_call is not None

        tool_output = instrumented_executor(tool_call)

        tool_executions = [
            ToolExecution(
                tool_call=tool_call,
                output=tool_output,
            )
        ]

        instrumented_generator(
            "Знайди намет",
            tool_executions,
        )

    finally:
        current_sequence_number.reset(sequence_token)
        current_trace_id.reset(trace_token)

    assert len(agent_step_records) == 2
    assert len(tool_records) == 1

    assert agent_step_records[0]["trace_id"] == "test-trace"
    assert agent_step_records[0]["sequence_number"] == 1
    assert agent_step_records[0]["decision"] == "tool_call"

    assert tool_records[0]["trace_id"] == "test-trace"
    assert tool_records[0]["sequence_number"] == 2
    assert agent_step_records[0]["step_number"] == 1
    assert agent_step_records[1]["step_number"] == 2
    assert tool_records[0]["tool_name"] == "search_products"

    assert agent_step_records[1]["trace_id"] == "test-trace"
    assert agent_step_records[1]["sequence_number"] == 3
    assert agent_step_records[1]["decision"] == "final_answer"
