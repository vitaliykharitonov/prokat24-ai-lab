from typing import TypedDict

from prokat24_ai_lab.observability.openai_usage import OpenAIModelUsageRecord


class OpenAIModelPricing(TypedDict):
    input_usd_per_million: float
    cached_input_usd_per_million: float
    output_usd_per_million: float


class OpenAIModelCostRecord(TypedDict):
    trace_id: str | None
    step_number: int
    model: str
    input_cost_usd: float
    cached_input_cost_usd: float
    output_cost_usd: float
    total_cost_usd: float


class OpenAITraceCostRecord(TypedDict):
    trace_id: str | None
    model_call_count: int
    input_cost_usd: float
    cached_input_cost_usd: float
    output_cost_usd: float
    total_cost_usd: float


OPENAI_MODEL_PRICING: dict[str, OpenAIModelPricing] = {
    "gpt-5-mini": {
        "input_usd_per_million": 0.25,
        "cached_input_usd_per_million": 0.025,
        "output_usd_per_million": 2.00,
    }
}


def estimate_openai_model_cost(
    usage: OpenAIModelUsageRecord,
) -> OpenAIModelCostRecord:
    uncached_input_tokens = usage["input_tokens"] - usage["cached_input_tokens"]

    pricing = OPENAI_MODEL_PRICING[usage["model"]]

    input_cost_usd = uncached_input_tokens / 1_000_000 * pricing["input_usd_per_million"]
    cached_input_cost_usd = (
        usage["cached_input_tokens"] / 1_000_000 * pricing["cached_input_usd_per_million"]
    )
    output_cost_usd = usage["output_tokens"] / 1_000_000 * pricing["output_usd_per_million"]

    total_cost_usd = input_cost_usd + cached_input_cost_usd + output_cost_usd

    return OpenAIModelCostRecord(
        trace_id=usage["trace_id"],
        step_number=usage["step_number"],
        model=usage["model"],
        input_cost_usd=input_cost_usd,
        cached_input_cost_usd=cached_input_cost_usd,
        output_cost_usd=output_cost_usd,
        total_cost_usd=total_cost_usd,
    )


def aggregate_openai_costs_by_trace(
    records: list[OpenAIModelCostRecord],
) -> list[OpenAITraceCostRecord]:
    grouped: dict[str | None, OpenAITraceCostRecord] = {}

    for record in records:
        trace_id = record["trace_id"]

        if trace_id not in grouped:
            grouped[trace_id] = OpenAITraceCostRecord(
                trace_id=trace_id,
                model_call_count=0,
                input_cost_usd=0.0,
                cached_input_cost_usd=0.0,
                output_cost_usd=0.0,
                total_cost_usd=0.0,
            )

        grouped[trace_id]["model_call_count"] += 1
        grouped[trace_id]["input_cost_usd"] += record["input_cost_usd"]
        grouped[trace_id]["cached_input_cost_usd"] += record["cached_input_cost_usd"]
        grouped[trace_id]["output_cost_usd"] += record["output_cost_usd"]
        grouped[trace_id]["total_cost_usd"] += record["total_cost_usd"]

    return list(grouped.values())
