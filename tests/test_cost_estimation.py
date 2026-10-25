import pytest

from prokat24_ai_lab.observability.cost_estimation import (
    OpenAIModelCostRecord,
    aggregate_openai_costs_by_trace,
    estimate_openai_model_cost,
)
from prokat24_ai_lab.observability.openai_usage import OpenAIModelUsageRecord


def test_estimate_openai_model_cost_with_cached_input() -> None:
    usage = OpenAIModelUsageRecord(
        trace_id="test-trace",
        step_number=2,
        model="gpt-5-mini",
        input_tokens=1_000_000,
        cached_input_tokens=400_000,
        output_tokens=100_000,
        reasoning_tokens=50_000,
        total_tokens=1_100_000,
    )

    cost = estimate_openai_model_cost(usage)

    assert cost["trace_id"] == "test-trace"
    assert cost["step_number"] == 2
    assert cost["model"] == "gpt-5-mini"

    assert cost["input_cost_usd"] == pytest.approx(0.15)
    assert cost["cached_input_cost_usd"] == pytest.approx(0.01)
    assert cost["output_cost_usd"] == pytest.approx(0.20)
    assert cost["total_cost_usd"] == pytest.approx(0.36)


def test_aggregate_openai_costs_by_trace() -> None:
    records = [
        OpenAIModelCostRecord(
            trace_id="trace-1",
            step_number=1,
            model="gpt-5-mini",
            input_cost_usd=0.10,
            cached_input_cost_usd=0.01,
            output_cost_usd=0.20,
            total_cost_usd=0.31,
        ),
        OpenAIModelCostRecord(
            trace_id="trace-1",
            step_number=2,
            model="gpt-5-mini",
            input_cost_usd=0.20,
            cached_input_cost_usd=0.02,
            output_cost_usd=0.30,
            total_cost_usd=0.52,
        ),
        OpenAIModelCostRecord(
            trace_id="trace-2",
            step_number=1,
            model="gpt-5-mini",
            input_cost_usd=0.05,
            cached_input_cost_usd=0.00,
            output_cost_usd=0.10,
            total_cost_usd=0.15,
        ),
    ]

    result = aggregate_openai_costs_by_trace(records)

    assert result[0]["trace_id"] == "trace-1"
    assert result[0]["model_call_count"] == 2
    assert result[0]["input_cost_usd"] == pytest.approx(0.30)
    assert result[0]["cached_input_cost_usd"] == pytest.approx(0.03)
    assert result[0]["output_cost_usd"] == pytest.approx(0.50)
    assert result[0]["total_cost_usd"] == pytest.approx(0.83)

    assert result[1]["trace_id"] == "trace-2"
    assert result[1]["model_call_count"] == 1
    assert result[1]["input_cost_usd"] == pytest.approx(0.05)
    assert result[1]["cached_input_cost_usd"] == pytest.approx(0.00)
    assert result[1]["output_cost_usd"] == pytest.approx(0.10)
    assert result[1]["total_cost_usd"] == pytest.approx(0.15)
