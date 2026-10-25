import json
import logging
from typing import TypedDict

from prokat24_ai_lab.observability.agent_step_instrumentation import AgentStepExecutionRecord
from prokat24_ai_lab.observability.openai_usage import OpenAIModelUsageRecord
from prokat24_ai_lab.observability.tool_instrumentation import ToolExecutionRecord


class ObservabilityLogEvent(TypedDict):
    event: str
    data: dict[str, object]


def build_observability_log_event(
    event: str,
    data: dict[str, object],
) -> ObservabilityLogEvent:
    return ObservabilityLogEvent(
        event=event,
        data=data,
    )


def serialize_observability_log_event(
    log_event: ObservabilityLogEvent,
) -> str:
    return json.dumps(log_event)


def log_observability_event(
    logger: logging.Logger,
    log_event: ObservabilityLogEvent,
) -> None:
    logger.info("%s", serialize_observability_log_event(log_event))


def build_agent_step_log_event(
    record: AgentStepExecutionRecord,
) -> ObservabilityLogEvent:
    return ObservabilityLogEvent(
        event="agent_step",
        data={
            "trace_id": record["trace_id"],
            "sequence_number": record["sequence_number"],
            "step_number": record["step_number"],
            "started_at": record["started_at"],
            "status": record["status"],
            "duration_ms": record["duration_ms"],
            "decision": record["decision"],
            "error_type": record["error_type"],
            "error_message": record["error_message"],
        },
    )


def build_tool_execution_log_event(
    record: ToolExecutionRecord,
) -> ObservabilityLogEvent:
    return ObservabilityLogEvent(
        event="tool_execution",
        data={
            "trace_id": record["trace_id"],
            "sequence_number": record["sequence_number"],
            "started_at": record["started_at"],
            "tool_name": record["tool_name"],
            "duration_ms": record["duration_ms"],
            "status": record["status"],
            "error_type": record["error_type"],
            "error_message": record["error_message"],
        },
    )


def build_model_usage_log_event(
    record: OpenAIModelUsageRecord,
) -> ObservabilityLogEvent:
    return ObservabilityLogEvent(
        event="model_usage",
        data={
            "trace_id": record["trace_id"],
            "step_number": record["step_number"],
            "model": record["model"],
            "input_tokens": record["input_tokens"],
            "cached_input_tokens": record["cached_input_tokens"],
            "output_tokens": record["output_tokens"],
            "reasoning_tokens": record["reasoning_tokens"],
            "total_tokens": record["total_tokens"],
        },
    )
