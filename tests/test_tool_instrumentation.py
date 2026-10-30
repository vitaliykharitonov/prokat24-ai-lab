import pytest

from prokat24_ai_lab.agents.types import ToolCall
from prokat24_ai_lab.observability.context import current_sequence_number, current_trace_id
from prokat24_ai_lab.observability.tool_instrumentation import (
    ToolExecutionRecord,
    create_instrumented_tool_executor,
)


def test_instrumented_tool_executor_records_latency_and_returns_output() -> None:
    records: list[ToolExecutionRecord] = []

    def fake_executor(_tool_call: ToolCall) -> str:
        return "tool output"

    def on_record(record: ToolExecutionRecord) -> None:
        records.append(record)

    instrumented_executor = create_instrumented_tool_executor(fake_executor, on_record)

    sequence_token = current_sequence_number.set(0)

    try:
        result = instrumented_executor(
            ToolCall(
                tool_name="search_products",
                arguments={"query": "намет"},
            )
        )

    finally:
        current_sequence_number.reset(sequence_token)

    assert result == "tool output"
    assert len(records) == 1
    assert records[0]["trace_id"] is None
    assert records[0]["sequence_number"] == 1
    assert records[0]["tool_name"] == "search_products"
    assert records[0]["duration_ms"] >= 0
    assert records[0]["status"] == "success"
    assert records[0]["error_type"] is None
    assert records[0]["error_message"] is None


def test_instrumented_tool_executor_records_error_and_reraises() -> None:
    records: list[ToolExecutionRecord] = []

    def fake_executor(_tool_call: ToolCall) -> str:
        raise ValueError("Product not found")

    def on_record(record: ToolExecutionRecord) -> None:
        records.append(record)

    instrumented_executor = create_instrumented_tool_executor(fake_executor, on_record)

    sequence_token = current_sequence_number.set(0)

    try:
        with pytest.raises(ValueError, match="Product not found"):
            instrumented_executor(
                ToolCall(
                    tool_name="search_products",
                    arguments={"query": "намет"},
                )
            )

    finally:
        current_sequence_number.reset(sequence_token)

    assert len(records) == 1
    assert records[0]["trace_id"] is None
    assert records[0]["sequence_number"] == 1
    assert records[0]["tool_name"] == "search_products"
    assert records[0]["status"] == "error"
    assert records[0]["error_type"] == "ValueError"
    assert records[0]["error_message"] == "Product not found"
    assert records[0]["duration_ms"] >= 0


def test_instrumented_tool_executor_uses_current_trace_id() -> None:
    records: list[ToolExecutionRecord] = []

    def fake_executor(_tool_call: ToolCall) -> str:
        return "tool output"

    def on_record(record: ToolExecutionRecord) -> None:
        records.append(record)

    instrumented_executor = create_instrumented_tool_executor(fake_executor, on_record)

    token = current_trace_id.set("test-trace")
    sequence_token = current_sequence_number.set(0)

    try:
        instrumented_executor(
            ToolCall(
                tool_name="search_products",
                arguments={"query": "намет"},
            )
        )

    finally:
        current_sequence_number.reset(sequence_token)
        current_trace_id.reset(token)

    assert len(records) == 1
    assert records[0]["trace_id"] == "test-trace"
    assert records[0]["sequence_number"] == 1
