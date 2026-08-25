import json
from pathlib import Path

from prokat24_ai_lab.evals.tool_use import (
    ToolUseEvalCase,
    evaluate_tool_use_case,
    load_tool_use_eval_cases,
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
    assert result["tool_name_matches"] is True
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

    assert result["tool_name_matches"] is True
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

    assert result["tool_name_matches"] is False
    assert result["arguments_match"] is True
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

    assert result["tool_name_matches"] is False
    assert result["arguments_match"] is False
    assert result["passed"] is False
