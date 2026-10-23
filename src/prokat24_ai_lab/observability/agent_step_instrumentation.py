import datetime
from collections.abc import Callable
from time import perf_counter
from typing import Literal, TypedDict

from prokat24_ai_lab.agents.types import AgentStep, AgentStepGenerator, ToolExecution
from prokat24_ai_lab.observability.context import current_trace_id, next_sequence_number


class AgentStepExecutionRecord(TypedDict):
    trace_id: str | None
    sequence_number: int
    started_at: str
    status: Literal["success", "error"]
    duration_ms: float
    decision: Literal["tool_call", "final_answer"] | None
    error_type: str | None
    error_message: str | None


def create_instrumented_agent_step_generator(
    generator: AgentStepGenerator,
    on_record: Callable[[AgentStepExecutionRecord], None],
) -> AgentStepGenerator:
    def instrumented_generator(user_input: str, tool_executions: list[ToolExecution]) -> AgentStep:
        started_at = datetime.datetime.now(datetime.UTC).isoformat(timespec="milliseconds")
        start = perf_counter()

        status: Literal["success", "error"] = "success"
        decision: Literal["tool_call", "final_answer"] | None = None
        error_type: str | None = None
        error_message: str | None = None

        try:
            step = generator(user_input, tool_executions)
            decision = "tool_call" if step["tool_call"] is not None else "final_answer"

        except Exception as error:
            status = "error"
            error_type = type(error).__name__
            error_message = str(error)

            raise

        finally:
            duration_ms = (perf_counter() - start) * 1000

            record = AgentStepExecutionRecord(
                trace_id=current_trace_id.get(),
                sequence_number=next_sequence_number(),
                started_at=started_at,
                duration_ms=duration_ms,
                status=status,
                decision=decision,
                error_type=error_type,
                error_message=error_message,
            )
            on_record(record)

        return step

    return instrumented_generator
