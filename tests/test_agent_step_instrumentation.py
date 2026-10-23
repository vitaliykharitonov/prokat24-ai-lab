import pytest

from prokat24_ai_lab.agents.types import AgentStep, ToolCall, ToolExecution
from prokat24_ai_lab.observability.agent_step_instrumentation import (
    AgentStepExecutionRecord,
    create_instrumented_agent_step_generator,
)
from prokat24_ai_lab.observability.context import current_sequence_number, current_trace_id


def test_instrumented_agent_step_generator_records_tool_call_decision() -> None:
    records: list[AgentStepExecutionRecord] = []

    def fake_generator(_user_input: str, _tool_executions: list[ToolExecution]) -> AgentStep:
        return AgentStep(
            final_answer=None,
            tool_call=ToolCall(
                tool_name="search_products",
                arguments={"query": "намет"},
            ),
        )

    def on_record(record: AgentStepExecutionRecord) -> None:
        records.append(record)

    instrumented_generator = create_instrumented_agent_step_generator(fake_generator, on_record)

    sequence_token = current_sequence_number.set(0)

    try:
        result = instrumented_generator(
            "Знайди намет",
            [],
        )

    finally:
        current_sequence_number.reset(sequence_token)

    assert result["final_answer"] is None
    assert result["tool_call"] == ToolCall(
        tool_name="search_products",
        arguments={"query": "намет"},
    )
    assert len(records) == 1
    assert records[0]["trace_id"] is None
    assert records[0]["sequence_number"] == 1
    assert records[0]["duration_ms"] >= 0
    assert records[0]["status"] == "success"
    assert records[0]["decision"] == "tool_call"
    assert records[0]["error_type"] is None
    assert records[0]["error_message"] is None


def test_instrumented_agent_step_generator_records_final_answer_decision() -> None:
    records: list[AgentStepExecutionRecord] = []

    def fake_generator(_user_input: str, _tool_executions: list[ToolExecution]) -> AgentStep:
        return AgentStep(
            final_answer="Ось знайдений намет.",
            tool_call=None,
        )

    def on_record(record: AgentStepExecutionRecord) -> None:
        records.append(record)

    instrumented_generator = create_instrumented_agent_step_generator(fake_generator, on_record)

    sequence_token = current_sequence_number.set(0)

    try:
        result = instrumented_generator(
            "Знайди намет",
            [],
        )

    finally:
        current_sequence_number.reset(sequence_token)

    assert result["final_answer"] == "Ось знайдений намет."
    assert result["tool_call"] is None
    assert len(records) == 1
    assert records[0]["trace_id"] is None
    assert records[0]["sequence_number"] == 1
    assert records[0]["duration_ms"] >= 0
    assert records[0]["status"] == "success"
    assert records[0]["decision"] == "final_answer"
    assert records[0]["error_type"] is None
    assert records[0]["error_message"] is None


def test_instrumented_agent_step_generator_records_error_and_reraises() -> None:
    records: list[AgentStepExecutionRecord] = []

    def fake_generator(_user_input: str, _tool_executions: list[ToolExecution]) -> AgentStep:
        raise ValueError("Model failed")

    def on_record(record: AgentStepExecutionRecord) -> None:
        records.append(record)

    instrumented_generator = create_instrumented_agent_step_generator(fake_generator, on_record)

    sequence_token = current_sequence_number.set(0)

    try:
        with pytest.raises(ValueError, match="Model failed"):
            instrumented_generator(
                "Знайди намет",
                [],
            )

    finally:
        current_sequence_number.reset(sequence_token)

    assert len(records) == 1
    assert records[0]["trace_id"] is None
    assert records[0]["sequence_number"] == 1
    assert records[0]["duration_ms"] >= 0
    assert records[0]["status"] == "error"
    assert records[0]["decision"] is None
    assert records[0]["error_type"] == "ValueError"
    assert records[0]["error_message"] == "Model failed"


def test_instrumented_agent_step_generator_uses_current_trace_id() -> None:
    records: list[AgentStepExecutionRecord] = []

    def fake_generator(_user_input: str, _tool_executions: list[ToolExecution]) -> AgentStep:
        return AgentStep(
            final_answer="Ось знайдений намет.",
            tool_call=None,
        )

    def on_record(record: AgentStepExecutionRecord) -> None:
        records.append(record)

    instrumented_generator = create_instrumented_agent_step_generator(fake_generator, on_record)

    token = current_trace_id.set("test-trace")

    sequence_token = current_sequence_number.set(0)

    try:
        instrumented_generator(
            "Знайди намет",
            [],
        )

    finally:
        current_sequence_number.reset(sequence_token)
        current_trace_id.reset(token)

    assert len(records) == 1
    assert records[0]["trace_id"] == "test-trace"
