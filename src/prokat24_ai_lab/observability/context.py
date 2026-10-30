from contextvars import ContextVar

current_trace_id: ContextVar[str | None] = ContextVar(
    "current_trace_id",
    default=None,
)

current_sequence_number: ContextVar[int] = ContextVar(
    "current_sequence_number",
    default=0,
)

current_eval_case_id: ContextVar[str | None] = ContextVar(
    "current_eval_case_id",
    default=None,
)


def next_sequence_number() -> int:
    number = current_sequence_number.get() + 1
    current_sequence_number.set(number)

    return number
