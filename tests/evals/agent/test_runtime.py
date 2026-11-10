import logging
from typing import cast
from unittest.mock import MagicMock

from opentelemetry.trace import INVALID_SPAN_CONTEXT, SpanContext, TraceFlags, Tracer
from pytest import MonkeyPatch

from prokat24_ai_lab.agents.types import (
    AgentStep,
    ToolCall,
    ToolExecution,
)
from prokat24_ai_lab.evals.agent import runtime
from prokat24_ai_lab.observability.context import current_eval_case_id


def test_create_agent_eval_runtime_runs_instrumented_agent(
    monkeypatch: MonkeyPatch,
) -> None:
    trace_id = 0x1234567890ABCDEF1234567890ABCDEF

    span_context = SpanContext(
        trace_id=trace_id,
        span_id=0x1234567890ABCDEF,
        is_remote=False,
        trace_flags=TraceFlags(1),
    )

    span = MagicMock()
    span.get_span_context.return_value = span_context

    span_context_manager = MagicMock()
    span_context_manager.__enter__.return_value = span

    tracer_mock = MagicMock()
    tracer_mock.start_as_current_span.return_value = span_context_manager
    tracer = cast(Tracer, tracer_mock)

    def generate_agent_step(
        user_input: str,
        tool_executions: list[ToolExecution],
    ) -> AgentStep:
        assert user_input == "Find a tent"

        if not tool_executions:
            return AgentStep(
                final_answer=None,
                tool_call=ToolCall(
                    tool_name="search_products",
                    arguments={"query": "tent"},
                ),
            )

        return AgentStep(
            final_answer="Tent found.",
            tool_call=None,
        )

    create_agent_step_generator_mock = MagicMock(
        return_value=generate_agent_step,
    )
    monkeypatch.setattr(
        runtime,
        "create_openai_agent_step_generator",
        create_agent_step_generator_mock,
    )

    def execute_tool_call(tool_call: ToolCall) -> str:
        assert tool_call["tool_name"] == "search_products"
        return "Tent search result"

    create_tool_executor_mock = MagicMock(
        return_value=execute_tool_call,
    )
    monkeypatch.setattr(
        runtime,
        "create_tool_executor",
        create_tool_executor_mock,
    )

    logger = logging.getLogger("test.agent_eval_runtime")
    logger.handlers = [logging.NullHandler()]
    logger.propagate = False

    agent_runtime = runtime.create_agent_eval_runtime(
        MagicMock(),
        products=[],
        pages=[],
        tracer=tracer,
        logger=logger,
        max_steps=5,
    )

    case_token = current_eval_case_id.set("case_1")

    try:
        result = agent_runtime["agent_runner"]("Find a tent")
    finally:
        current_eval_case_id.reset(case_token)

    expected_trace_id = format(trace_id, "032x")

    assert result["final_answer"] == "Tent found."

    assert result["tool_calls"] == [
        {
            "tool_name": "search_products",
            "arguments": {"query": "tent"},
        }
    ]

    assert result["tool_executions"] == [
        {
            "tool_call": {
                "tool_name": "search_products",
                "arguments": {"query": "tent"},
            },
            "output": "Tent search result",
        }
    ]

    assert agent_runtime["eval_trace_ids"] == {
        "case_1": expected_trace_id,
    }

    assert len(agent_runtime["agent_step_records"]) == 2
    assert len(agent_runtime["tool_execution_records"]) == 1
    assert agent_runtime["model_usage_records"] == []

    assert [record["sequence_number"] for record in agent_runtime["agent_step_records"]] == [1, 3]

    assert [record["sequence_number"] for record in agent_runtime["tool_execution_records"]] == [2]

    assert all(
        record["trace_id"] == expected_trace_id for record in agent_runtime["agent_step_records"]
    )

    assert agent_runtime["tool_execution_records"][0]["trace_id"] == expected_trace_id


def test_get_trace_id_returns_none_for_invalid_span_context() -> None:
    assert runtime.get_trace_id(INVALID_SPAN_CONTEXT) is None


def test_get_trace_id_returns_formatted_id_for_valid_span_context() -> None:
    span_context = SpanContext(
        trace_id=int("1234567890abcdef1234567890abcdef", 16),
        span_id=1,
        is_remote=False,
        trace_flags=TraceFlags(0),
        trace_state=None,
    )

    assert runtime.get_trace_id(span_context) == "1234567890abcdef1234567890abcdef"


def test_get_correlation_id_reuses_trace_id() -> None:
    assert runtime.get_correlation_id("trace-123") == "trace-123"


def test_get_correlation_id_generates_id_when_trace_id_is_missing() -> None:
    correlation_id = runtime.get_correlation_id(None)

    assert len(correlation_id) == 32
    int(correlation_id, 16)
