from collections.abc import Callable
from time import perf_counter
from typing import Literal, TypedDict

from prokat24_ai_lab.agents.types import ToolCall, ToolExecutor


class ToolExecutionRecord(TypedDict):
    tool_name: str
    duration_ms: float
    status: Literal["success", "error"]
    error_type: str | None
    error_message: str | None


def create_instrumented_tool_executor(
    executor: ToolExecutor,
    on_record: Callable[[ToolExecutionRecord], None],
) -> ToolExecutor:
    def instrumented_executor(tool_call: ToolCall) -> str:
        start = perf_counter()

        status: Literal["success", "error"] = "success"
        error_type: str | None = None
        error_message: str | None = None

        try:
            output = executor(tool_call)

        except Exception as error:
            status = "error"
            error_type = type(error).__name__
            error_message = str(error)

            raise

        finally:
            duration_ms = (perf_counter() - start) * 1000

            record = ToolExecutionRecord(
                tool_name=tool_call["tool_name"],
                duration_ms=duration_ms,
                status=status,
                error_type=error_type,
                error_message=error_message,
            )
            on_record(record)

        return output

    return instrumented_executor
