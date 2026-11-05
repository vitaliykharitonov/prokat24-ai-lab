import json
import logging

from prokat24_ai_lab.observability.agent_step_instrumentation import AgentStepExecutionRecord
from prokat24_ai_lab.observability.openai_usage import OpenAIModelUsageRecord
from prokat24_ai_lab.observability.structured_logging import (
    ObservabilityLogEvent,
    build_agent_step_log_event,
    build_model_usage_log_event,
    build_observability_log_event,
    build_tool_execution_log_event,
    log_observability_event,
    serialize_observability_log_event,
)
from prokat24_ai_lab.observability.tool_instrumentation import ToolExecutionRecord


def test_build_observability_log_event() -> None:
    event = "agent_step"
    data = {
        "trace_id": "trace-123",
        "step_number": 2,
        "status": "success",
    }

    log_event = build_observability_log_event(event, data)

    assert log_event == {
        "event": "agent_step",
        "data": {
            "trace_id": "trace-123",
            "step_number": 2,
            "status": "success",
        },
    }


def test_serialize_observability_log_event_returns_valid_json() -> None:
    log_event = ObservabilityLogEvent(
        event="tool_execution",
        data={
            "trace_id": "trace-456",
            "tool_name": "search_products",
            "duration_ms": 3.25,
            "status": "success",
        },
    )

    serialized = serialize_observability_log_event(log_event)
    parsed = json.loads(serialized)

    assert parsed["event"] == "tool_execution"
    assert parsed["data"]["trace_id"] == "trace-456"
    assert parsed["data"]["tool_name"] == "search_products"
    assert parsed["data"]["duration_ms"] == 3.25
    assert parsed["data"]["status"] == "success"


def test_log_observability_event_writes_json_to_logger(caplog) -> None:
    log_event = ObservabilityLogEvent(
        event="agent_step",
        data={
            "trace_id": "trace-789",
            "step_number": 2,
            "status": "success",
        },
    )

    logger = logging.getLogger("test.observability")

    caplog.set_level(logging.INFO, logger=logger.name)

    log_observability_event(logger, log_event)

    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "INFO"

    message = caplog.records[0].getMessage()
    parsed = json.loads(message)

    assert parsed["event"] == "agent_step"
    assert parsed["data"]["trace_id"] == "trace-789"
    assert parsed["data"]["step_number"] == 2
    assert parsed["data"]["status"] == "success"


def test_build_agent_step_log_event() -> None:
    record = AgentStepExecutionRecord(
        trace_id="trace-123",
        sequence_number=3,
        step_number=2,
        started_at="2026-10-25T18:00:00.000+00:00",
        status="success",
        duration_ms=1250.5,
        decision="final_answer",
        error_type=None,
        error_message=None,
    )

    log_event = build_agent_step_log_event(record)

    assert log_event["event"] == "agent_step"
    assert log_event["data"] == {
        "trace_id": "trace-123",
        "sequence_number": 3,
        "step_number": 2,
        "started_at": "2026-10-25T18:00:00.000+00:00",
        "status": "success",
        "duration_ms": 1250.5,
        "decision": "final_answer",
        "error_type": None,
        "error_message": None,
    }


def test_build_tool_execution_log_event() -> None:
    record = ToolExecutionRecord(
        trace_id="trace-456",
        sequence_number=2,
        started_at="2026-10-25T18:00:01.000+00:00",
        tool_name="search_products",
        duration_ms=3.75,
        status="success",
        error_type=None,
        error_message=None,
    )

    log_event = build_tool_execution_log_event(record)

    assert log_event["event"] == "tool_execution"
    assert log_event["data"] == {
        "trace_id": "trace-456",
        "sequence_number": 2,
        "started_at": "2026-10-25T18:00:01.000+00:00",
        "tool_name": "search_products",
        "duration_ms": 3.75,
        "status": "success",
        "error_type": None,
        "error_message": None,
    }


def test_build_model_usage_log_event() -> None:
    record = OpenAIModelUsageRecord(
        trace_id="trace-789",
        step_number=2,
        model="gpt-5-mini",
        input_tokens=1200,
        cached_input_tokens=400,
        output_tokens=300,
        reasoning_tokens=100,
        total_tokens=1500,
    )

    log_event = build_model_usage_log_event(record)

    assert log_event["event"] == "model_usage"
    assert log_event["data"] == {
        "trace_id": "trace-789",
        "step_number": 2,
        "model": "gpt-5-mini",
        "input_tokens": 1200,
        "cached_input_tokens": 400,
        "output_tokens": 300,
        "reasoning_tokens": 100,
        "total_tokens": 1500,
    }
