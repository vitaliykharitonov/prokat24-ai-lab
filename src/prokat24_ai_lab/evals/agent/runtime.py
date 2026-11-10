import logging
from typing import TypedDict
from uuid import uuid4

from opentelemetry.trace import SpanContext, Tracer

from prokat24_ai_lab.agents.agent_loop import run_agent
from prokat24_ai_lab.agents.openai_agent_step import create_openai_agent_step_generator
from prokat24_ai_lab.agents.openai_tool_call import ToolCallClientProtocol
from prokat24_ai_lab.agents.openai_tool_schema import create_openai_tools
from prokat24_ai_lab.agents.tool_executor import create_tool_executor
from prokat24_ai_lab.agents.tool_registry import create_tool_registry
from prokat24_ai_lab.agents.types import AgentRunner, AgentRunResult
from prokat24_ai_lab.knowledge.base import KnowledgePage
from prokat24_ai_lab.observability.agent_step_instrumentation import (
    AgentStepExecutionRecord,
    create_instrumented_agent_step_generator,
)
from prokat24_ai_lab.observability.context import (
    current_eval_case_id,
    current_sequence_number,
    current_trace_id,
)
from prokat24_ai_lab.observability.openai_usage import OpenAIModelUsageRecord
from prokat24_ai_lab.observability.otel_instrumentation import (
    create_traced_agent_step_generator,
    create_traced_tool_executor,
)
from prokat24_ai_lab.observability.structured_logging import (
    build_agent_step_log_event,
    build_model_usage_log_event,
    build_tool_execution_log_event,
    log_observability_event,
)
from prokat24_ai_lab.observability.tool_instrumentation import (
    ToolExecutionRecord,
    create_instrumented_tool_executor,
)
from prokat24_ai_lab.products.catalog import Product


class AgentEvalRuntime(TypedDict):
    agent_runner: AgentRunner
    agent_step_records: list[AgentStepExecutionRecord]
    tool_execution_records: list[ToolExecutionRecord]
    model_usage_records: list[OpenAIModelUsageRecord]
    eval_trace_ids: dict[str, str]


def get_trace_id(span_context: SpanContext) -> str | None:
    if not span_context.is_valid:
        return None

    return format(span_context.trace_id, "032x")


def get_correlation_id(trace_id: str | None) -> str:
    if trace_id is not None:
        return trace_id

    return uuid4().hex


def create_agent_eval_runtime(
    client: ToolCallClientProtocol,
    products: list[Product],
    pages: list[KnowledgePage],
    *,
    tracer: Tracer,
    logger: logging.Logger,
    max_steps: int,
) -> AgentEvalRuntime:
    registry = create_tool_registry(products, pages)
    openai_tools = create_openai_tools(registry)

    agent_step_records: list[AgentStepExecutionRecord] = []
    tool_execution_records: list[ToolExecutionRecord] = []
    model_usage_records: list[OpenAIModelUsageRecord] = []
    eval_trace_ids: dict[str, str] = {}

    def on_agent_step_record(record: AgentStepExecutionRecord) -> None:
        agent_step_records.append(record)

        log_event = build_agent_step_log_event(record)
        log_observability_event(logger, log_event)

    def on_model_usage_record(record: OpenAIModelUsageRecord) -> None:
        model_usage_records.append(record)

        log_event = build_model_usage_log_event(record)
        log_observability_event(logger, log_event)

    def on_tool_execution_record(record: ToolExecutionRecord) -> None:
        tool_execution_records.append(record)

        log_event = build_tool_execution_log_event(record)
        log_observability_event(logger, log_event)

    base_agent_step_generator = create_openai_agent_step_generator(
        client,
        openai_tools,
        on_usage=on_model_usage_record,
    )

    traced_agent_step_generator = create_traced_agent_step_generator(
        generator=base_agent_step_generator,
        tracer=tracer,
    )

    instrumented_agent_step_generator = create_instrumented_agent_step_generator(
        generator=traced_agent_step_generator,
        on_record=on_agent_step_record,
    )

    base_tool_executor = create_tool_executor(registry)

    traced_tool_executor = create_traced_tool_executor(
        executor=base_tool_executor,
        tracer=tracer,
    )

    instrumented_executor = create_instrumented_tool_executor(
        executor=traced_tool_executor,
        on_record=on_tool_execution_record,
    )

    def agent_runner(user_input: str) -> AgentRunResult:
        case_id = current_eval_case_id.get()
        sequence_token = current_sequence_number.set(0)

        try:
            with tracer.start_as_current_span("agent.run") as span:
                span.set_attribute("openinference.span.kind", "AGENT")

                trace_id = get_trace_id(span.get_span_context())
                correlation_id = get_correlation_id(trace_id)

                trace_token = current_trace_id.set(correlation_id)

                try:
                    if case_id is not None:
                        span.set_attribute("eval.case_id", case_id)

                        if trace_id is not None:
                            eval_trace_ids[case_id] = trace_id

                    span.set_attribute("input.value", user_input)
                    span.set_attribute("input.mime_type", "text/plain")

                    agent_result = run_agent(
                        user_input,
                        instrumented_agent_step_generator,
                        instrumented_executor,
                        max_steps=max_steps,
                    )

                    span.set_attribute("output.value", agent_result["final_answer"])
                    span.set_attribute("output.mime_type", "text/plain")

                    return agent_result

                finally:
                    current_trace_id.reset(trace_token)

        finally:
            current_sequence_number.reset(sequence_token)

    return AgentEvalRuntime(
        agent_runner=agent_runner,
        agent_step_records=agent_step_records,
        tool_execution_records=tool_execution_records,
        model_usage_records=model_usage_records,
        eval_trace_ids=eval_trace_ids,
    )
