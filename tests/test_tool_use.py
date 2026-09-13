import json
from pathlib import Path

import pytest

from prokat24_ai_lab.evals.tool_use import (
    ToolCall,
    ToolSequenceEvalCase,
    ToolSequenceEvalResult,
    ToolUseEvalCase,
    ToolUseEvalResult,
    evaluate_tool_sequence_case,
    evaluate_tool_use_case,
    load_tool_use_eval_cases,
    run_repeated_tool_use_evals,
    run_tool_sequence_evals,
    run_tool_use_evals,
    summarize_tool_sequence_eval_results,
    summarize_tool_use_eval_results,
)


def test_load_tool_use_eval_cases(tmp_path: Path) -> None:
    cases: list[ToolUseEvalCase] = [
        {
            "user_input": "Скільки коштує доставка по Києву?",
            "expected_tool_call": {
                "tool_name": "get_delivery_info",
                "arguments": {"location": "Київ"},
            },
        },
    ]

    input_file = tmp_path / "tool_use_eval_cases.json"

    with open(input_file, "w", encoding="utf-8") as file:
        json.dump(cases, file, ensure_ascii=False, indent=2)

    loaded_data = load_tool_use_eval_cases(input_file)

    assert loaded_data == cases


def test_load_tool_sequence_eval_cases_loads_cases(tmp_path: Path) -> None:
    cases: list[ToolSequenceEvalCase] = [
        {
            "user_input": "Знайди намет і скажи ціну оренди на 3 дні",
            "expected_tool_calls": [
                {
                    "tool_name": "search_products",
                    "arguments": {"query": "намет"},
                },
                {
                    "tool_name": "get_product_price",
                    "arguments": {
                        "product_url": "https://example.com/tent/",
                        "days": 3,
                    },
                },
            ],
        },
    ]

    input_file = tmp_path / "tool_use_eval_cases.json"

    with open(input_file, "w", encoding="utf-8") as file:
        json.dump(cases, file, ensure_ascii=False, indent=2)

    loaded_data = load_tool_use_eval_cases(input_file)

    assert loaded_data == cases


def test_evaluate_tool_use_case_passes_for_matching_tool_call() -> None:
    case = {
        "user_input": "Скільки коштує доставка по Києву?",
        "expected_tool_call": {
            "tool_name": "get_delivery_info",
            "arguments": {"location": "Київ"},
        },
    }
    actual_tool_call = {
        "tool_name": "get_delivery_info",
        "arguments": {"location": "Київ"},
    }

    result = evaluate_tool_use_case(case, actual_tool_call)

    assert result["user_input"] == case["user_input"]
    assert result["expected_tool_call"] == case["expected_tool_call"]
    assert result["actual_tool_call"] == actual_tool_call
    assert result["tool_name_match"] is True
    assert result["arguments_match"] is True
    assert result["passed"] is True


def test_evaluate_tool_use_case_fails_for_wrong_tool_call() -> None:
    case = {
        "user_input": "Скільки коштує доставка по Києву?",
        "expected_tool_call": {
            "tool_name": "get_delivery_info",
            "arguments": {"location": "Київ"},
        },
    }
    actual_tool_call = {
        "tool_name": "get_delivery_info",
        "arguments": {"location": "Львів"},
    }

    result = evaluate_tool_use_case(case, actual_tool_call)

    assert result["tool_name_match"] is True
    assert result["arguments_match"] is False
    assert result["passed"] is False


def test_evaluate_tool_use_case_fails_for_wrong_tool_name() -> None:
    case = {
        "user_input": "Скільки коштує доставка по Києву?",
        "expected_tool_call": {
            "tool_name": "get_delivery_info",
            "arguments": {"location": "Київ"},
        },
    }
    actual_tool_call = {
        "tool_name": "search_products",
        "arguments": {"location": "Київ"},
    }

    result = evaluate_tool_use_case(case, actual_tool_call)

    assert result["tool_name_match"] is False
    assert result["arguments_match"] is False
    assert result["passed"] is False


def test_evaluate_tool_use_case_fails_when_tool_call_is_missing() -> None:
    case = {
        "user_input": "Скільки коштує доставка по Києву?",
        "expected_tool_call": {
            "tool_name": "get_delivery_info",
            "arguments": {"location": "Київ"},
        },
    }
    actual_tool_call = None

    result = evaluate_tool_use_case(case, actual_tool_call)

    assert result["tool_name_match"] is False
    assert result["arguments_match"] is False
    assert result["passed"] is False


def test_evaluate_tool_use_case_passes_when_tool_call_is_not_expected() -> None:
    case = {
        "user_input": "Привіт!",
        "expected_tool_call": None,
    }
    actual_tool_call = None

    result = evaluate_tool_use_case(case, actual_tool_call)

    assert result["tool_name_match"] is True
    assert result["arguments_match"] is True
    assert result["passed"] is True


def test_evaluate_tool_use_case_fails_when_unexpected_tool_call_is_made() -> None:
    case = {
        "user_input": "Привіт!",
        "expected_tool_call": None,
    }
    actual_tool_call = {
        "tool_name": "get_delivery_info",
        "arguments": {"location": "Київ"},
    }

    result = evaluate_tool_use_case(case, actual_tool_call)

    assert result["tool_name_match"] is False
    assert result["arguments_match"] is False
    assert result["passed"] is False


def test_evaluate_tool_sequence_case_passes_for_matching_sequence() -> None:
    case = ToolSequenceEvalCase(
        user_input="Знайди намет і скажи ціну оренди на 3 дні",
        expected_tool_calls=[
            ToolCall(
                tool_name="search_products",
                arguments={"query": "намет"},
            ),
            ToolCall(
                tool_name="get_product_price",
                arguments={
                    "product_url": "https://example.com/tent/",
                    "days": 3,
                },
            ),
        ],
    )
    actual_tool_calls = [
        ToolCall(
            tool_name="search_products",
            arguments={"query": "намет"},
        ),
        ToolCall(
            tool_name="get_product_price",
            arguments={
                "product_url": "https://example.com/tent/",
                "days": 3,
            },
        ),
    ]
    result = evaluate_tool_sequence_case(case, actual_tool_calls)

    assert result["passed"] is True


def test_evaluate_tool_sequence_case_fails_for_wrong_tool_order() -> None:
    case = ToolSequenceEvalCase(
        user_input="Знайди намет і скажи ціну оренди на 3 дні",
        expected_tool_calls=[
            ToolCall(
                tool_name="search_products",
                arguments={"query": "намет"},
            ),
            ToolCall(
                tool_name="get_product_price",
                arguments={
                    "product_url": "https://example.com/tent/",
                    "days": 3,
                },
            ),
        ],
    )
    actual_tool_calls = [
        ToolCall(
            tool_name="get_product_price",
            arguments={
                "product_url": "https://example.com/tent/",
                "days": 3,
            },
        ),
        ToolCall(
            tool_name="search_products",
            arguments={"query": "намет"},
        ),
    ]
    result = evaluate_tool_sequence_case(case, actual_tool_calls)

    assert result["tool_count_match"] is True
    assert result["tool_names_match"] is False
    assert result["arguments_match"] is False
    assert result["passed"] is False


def test_evaluate_tool_sequence_case_detects_missing_tool_call() -> None:
    case = ToolSequenceEvalCase(
        user_input="Знайди намет і скажи ціну оренди на 3 дні",
        expected_tool_calls=[
            ToolCall(
                tool_name="search_products",
                arguments={"query": "намет"},
            ),
            ToolCall(
                tool_name="get_product_price",
                arguments={
                    "product_url": "https://example.com/tent/",
                    "days": 3,
                },
            ),
        ],
    )
    actual_tool_calls = [
        ToolCall(
            tool_name="search_products",
            arguments={"query": "намет"},
        ),
    ]

    result = evaluate_tool_sequence_case(case, actual_tool_calls)

    assert result["tool_count_match"] is False
    assert result["passed"] is False


def test_evaluate_tool_sequence_case_detects_wrong_arguments() -> None:
    case = ToolSequenceEvalCase(
        user_input="Знайди намет і скажи ціну оренди на 3 дні",
        expected_tool_calls=[
            ToolCall(
                tool_name="search_products",
                arguments={"query": "намет"},
            ),
            ToolCall(
                tool_name="get_product_price",
                arguments={
                    "product_url": "https://example.com/tent/",
                    "days": 3,
                },
            ),
        ],
    )
    actual_tool_calls = [
        ToolCall(
            tool_name="search_products",
            arguments={"query": "намет"},
        ),
        ToolCall(
            tool_name="get_product_price",
            arguments={
                "product_url": "https://example.com/tent/",
                "days": 5,
            },
        ),
    ]
    result = evaluate_tool_sequence_case(case, actual_tool_calls)

    assert result["tool_count_match"] is True
    assert result["tool_names_match"] is True
    assert result["arguments_match"] is False
    assert result["passed"] is False


def test_run_tool_use_evals() -> None:
    cases = [
        {
            "user_input": "Скільки коштує доставка по Києву?",
            "expected_tool_call": {
                "tool_name": "get_delivery_info",
                "arguments": {"location": "Київ"},
            },
        },
        {
            "user_input": "Знайди намет",
            "expected_tool_call": {
                "tool_name": "search_products",
                "arguments": {"query": "намет"},
            },
        },
    ]
    correct_tool_call = {
        "tool_name": "get_delivery_info",
        "arguments": {"location": "Київ"},
    }
    wrong_arguments_tool_call = {
        "tool_name": "search_products",
        "arguments": {"location": "Київ"},
    }

    def fake_generate_tool_call(user_input: str) -> ToolCall | None:
        if user_input == "Скільки коштує доставка по Києву?":
            return correct_tool_call

        if user_input == "Знайди намет":
            return wrong_arguments_tool_call

        return None

    results = run_tool_use_evals(cases, fake_generate_tool_call)

    assert len(results) == 2
    assert results[0]["passed"] is True
    assert results[1]["passed"] is False


def test_run_tool_sequence_evals_returns_results_for_all_cases() -> None:
    cases = [
        ToolSequenceEvalCase(
            user_input="Знайди намет і скажи ціну оренди на 3 дні",
            expected_tool_calls=[
                ToolCall(
                    tool_name="search_products",
                    arguments={"query": "намет"},
                ),
                ToolCall(
                    tool_name="get_product_price",
                    arguments={
                        "product_url": "https://example.com/tent/",
                        "days": 3,
                    },
                ),
            ],
        ),
        ToolSequenceEvalCase(
            user_input="Знайди спальник і скажи ціну оренди на 2 дні",
            expected_tool_calls=[
                ToolCall(
                    tool_name="search_products",
                    arguments={"query": "спальник"},
                ),
                ToolCall(
                    tool_name="get_product_price",
                    arguments={
                        "product_url": "https://example.com/sleeping-bag/",
                        "days": 2,
                    },
                ),
            ],
        ),
    ]

    def generate_tool_sequence(user_input: str) -> list[ToolCall]:
        if "намет" in user_input:
            return [
                ToolCall(
                    tool_name="search_products",
                    arguments={"query": "намет"},
                ),
                ToolCall(
                    tool_name="get_product_price",
                    arguments={
                        "product_url": "https://example.com/tent/",
                        "days": 3,
                    },
                ),
            ]

        return [
            ToolCall(
                tool_name="search_products",
                arguments={"query": "спальник"},
            ),
            ToolCall(
                tool_name="get_product_price",
                arguments={
                    "product_url": "https://example.com/sleeping-bag/",
                    "days": 2,
                },
            ),
        ]

    results = run_tool_sequence_evals(cases, generate_tool_sequence)

    assert len(results) == 2
    assert results[0]["passed"] is True
    assert results[1]["passed"] is True


def test_run_repeated_tool_use_evals() -> None:
    cases = [
        {
            "user_input": "Скільки коштує доставка по Києву?",
            "expected_tool_call": {
                "tool_name": "get_delivery_info",
                "arguments": {"location": "Київ"},
            },
        },
    ]
    correct_tool_call = {
        "tool_name": "get_delivery_info",
        "arguments": {"location": "Київ"},
    }

    def fake_generate_tool_call(_user_input: str) -> ToolCall | None:
        return correct_tool_call

    results = run_repeated_tool_use_evals(cases, fake_generate_tool_call, runs=3)

    assert len(results) == 1
    assert results[0]["user_input"] == cases[0]["user_input"]
    assert len(results[0]["results"]) == 3
    assert all(result["passed"] for result in results[0]["results"])
    assert results[0]["expected_tool_call"] == cases[0]["expected_tool_call"]


def test_summarize_tool_use_eval_results() -> None:
    results: list[ToolUseEvalResult] = [
        {
            "user_input": "Question 1",
            "expected_tool_call": {
                "tool_name": "get_delivery_info",
                "arguments": {"location": "Київ"},
            },
            "actual_tool_call": {
                "tool_name": "get_delivery_info",
                "arguments": {"location": "Київ"},
            },
            "tool_call_decision_match": True,
            "tool_name_match": True,
            "arguments_match": True,
            "passed": True,
        },
        {
            "user_input": "Question 2",
            "expected_tool_call": {
                "tool_name": "search_products",
                "arguments": {"query": "намет"},
            },
            "actual_tool_call": {
                "tool_name": "search_products",
                "arguments": {"query": "лижi"},
            },
            "tool_call_decision_match": True,
            "tool_name_match": True,
            "arguments_match": False,
            "passed": False,
        },
        {
            "user_input": "Question 3",
            "expected_tool_call": {
                "tool_name": "get_product_price",
                "arguments": {
                    "product": "намет",
                    "days": 3,
                },
            },
            "actual_tool_call": {
                "tool_name": "search_products",
                "arguments": {
                    "query": "намет",
                },
            },
            "tool_call_decision_match": True,
            "tool_name_match": False,
            "arguments_match": False,
            "passed": False,
        },
    ]

    summary = summarize_tool_use_eval_results(results)

    assert summary["total_cases"] == 3
    assert summary["tool_call_cases"] == 3
    assert summary["passed_cases"] == 1
    assert summary["pass_rate"] == pytest.approx(1 / 3)

    assert summary["tool_name_matches"] == 2
    assert summary["tool_name_match_rate"] == pytest.approx(2 / 3)

    assert summary["arguments_matches"] == 1
    assert summary["arguments_match_rate"] == pytest.approx(1 / 3)

    assert summary["tool_call_decision_matches"] == 3
    assert summary["tool_call_decision_match_rate"] == pytest.approx(1.0)


def test_summarize_tool_use_eval_results_handles_empty_results() -> None:
    results: list[ToolUseEvalResult] = []

    summary = summarize_tool_use_eval_results(results)

    assert summary["total_cases"] == 0
    assert summary["tool_call_cases"] == 0
    assert summary["passed_cases"] == 0
    assert summary["pass_rate"] == 0.0

    assert summary["tool_name_matches"] == 0
    assert summary["tool_name_match_rate"] == 0.0

    assert summary["arguments_matches"] == 0
    assert summary["arguments_match_rate"] == 0.0

    assert summary["tool_call_decision_matches"] == 0
    assert summary["tool_call_decision_match_rate"] == 0.0


def test_summarize_tool_sequence_eval_results() -> None:
    results: list[ToolSequenceEvalResult] = [
        {
            "user_input": "case 1",
            "expected_tool_calls": [],
            "actual_tool_calls": [],
            "tool_count_match": True,
            "tool_names_match": True,
            "arguments_match": True,
            "passed": True,
        },
        {
            "user_input": "case 2",
            "expected_tool_calls": [],
            "actual_tool_calls": [],
            "tool_count_match": True,
            "tool_names_match": True,
            "arguments_match": False,
            "passed": False,
        },
        {
            "user_input": "case 3",
            "expected_tool_calls": [],
            "actual_tool_calls": [],
            "tool_count_match": False,
            "tool_names_match": False,
            "arguments_match": False,
            "passed": False,
        },
    ]

    summary = summarize_tool_sequence_eval_results(results)

    assert summary["total_cases"] == 3
    assert summary["passed_cases"] == 1
    assert summary["pass_rate"] == pytest.approx(1 / 3)
    assert summary["tool_count_matches"] == 2
    assert summary["tool_count_match_rate"] == pytest.approx(2 / 3)
    assert summary["tool_name_matches"] == 2
    assert summary["tool_name_match_rate"] == pytest.approx(2 / 3)
    assert summary["arguments_matches"] == 1
    assert summary["arguments_match_rate"] == pytest.approx(1 / 3)


def test_summarize_tool_sequence_eval_results_returns_zero_summary_for_empty_results() -> None:
    results: list[ToolSequenceEvalResult] = []

    summary = summarize_tool_sequence_eval_results(results)

    assert summary["total_cases"] == 0
    assert summary["passed_cases"] == 0
    assert summary["pass_rate"] == 0.0
    assert summary["tool_count_matches"] == 0
    assert summary["tool_count_match_rate"] == 0.0
    assert summary["tool_name_matches"] == 0
    assert summary["tool_name_match_rate"] == 0.0
    assert summary["arguments_matches"] == 0
    assert summary["arguments_match_rate"] == 0.0
