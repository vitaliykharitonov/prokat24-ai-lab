from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from prokat24_ai_lab.agents.types import AgentStep, ToolCall, ToolExecution
from prokat24_ai_lab.observability.otel_instrumentation import (
    create_traced_agent_step_generator,
    create_traced_tool_executor,
)


def test_traced_tool_executor_creates_tool_span() -> None:
    def fake_executor(_tool_call: ToolCall) -> str:
        return "tool result"

    exporter = InMemorySpanExporter()

    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))

    tracer = tracer_provider.get_tracer(__name__)

    traced_executor = create_traced_tool_executor(
        executor=fake_executor,
        tracer=tracer,
    )

    tool_call = ToolCall(
        tool_name="search_products",
        arguments={"query": "намет"},
    )

    result = traced_executor(tool_call)

    spans = exporter.get_finished_spans()

    assert result == "tool result"
    assert len(spans) == 1

    assert spans[0].name == "tool.search_products"
    assert spans[0].attributes["openinference.span.kind"] == "TOOL"
    assert spans[0].attributes["tool.name"] == "search_products"


def test_traced_agent_step_generator_creates_llm_span() -> None:
    def fake_generator(
        _user_input: str,
        _tool_executions: list[ToolExecution],
    ) -> AgentStep:
        return AgentStep(
            final_answer="Привіт!",
            tool_call=None,
        )

    exporter = InMemorySpanExporter()

    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))

    tracer = tracer_provider.get_tracer(__name__)

    traced_generator = create_traced_agent_step_generator(
        generator=fake_generator,
        tracer=tracer,
    )

    result = traced_generator(
        "Привіт",
        [],
    )

    spans = exporter.get_finished_spans()

    assert result == AgentStep(
        final_answer="Привіт!",
        tool_call=None,
    )

    assert len(spans) == 1
    assert spans[0].name == "llm.agent_step"
    assert spans[0].attributes["prokat24.step_number"] == 1
    assert spans[0].attributes["prokat24.decision"] == "final_answer"
