from typing import TypedDict


class OpenAIModelUsageRecord(TypedDict):
    trace_id: str | None
    step_number: int
    model: str
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    total_tokens: int
