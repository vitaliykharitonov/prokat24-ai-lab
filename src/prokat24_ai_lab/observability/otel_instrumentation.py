from opentelemetry.trace import Tracer

from prokat24_ai_lab.agents.types import (
    AgentStep,
    AgentStepGenerator,
    ToolCall,
    ToolExecution,
    ToolExecutor,
)


def create_traced_tool_executor(
    executor: ToolExecutor,
    tracer: Tracer,
) -> ToolExecutor:
    def traced_executor(tool_call: ToolCall) -> str:
        tool_name = tool_call["tool_name"]

        with tracer.start_as_current_span(f"tool.{tool_name}") as span:
            span.set_attribute("openinference.span.kind", "TOOL")
            span.set_attribute("tool.name", tool_name)

            return executor(tool_call)

    return traced_executor


def create_traced_agent_step_generator(
    generator: AgentStepGenerator,
    tracer: Tracer,
) -> AgentStepGenerator:
    def traced_step_generator(user_input: str, tool_executions: list[ToolExecution]) -> AgentStep:
        with tracer.start_as_current_span("llm.agent_step") as span:
            step_number = len(tool_executions) + 1
            span.set_attribute("openinference.span.kind", "CHAIN")
            span.set_attribute("prokat24.step_number", step_number)

            step = generator(user_input, tool_executions)

            decision = "tool_call" if step["tool_call"] is not None else "final_answer"
            span.set_attribute("prokat24.decision", decision)

            return step

    return traced_step_generator
