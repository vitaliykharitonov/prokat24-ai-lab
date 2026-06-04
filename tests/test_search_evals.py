import json
from pathlib import Path

from prokat24_ai_lab.search_evals import (
    SearchEvalCase,
    calculate_accuracy,
    evaluate_search_case,
    load_search_eval_cases,
    run_search_evals,
)


def test_load_search_eval_cases_reads_json_file(tmp_path: Path) -> None:
    output_file = tmp_path / "search_eval_cases.json"
    cases: list[SearchEvalCase] = [
        {
            "query": "доставка київ",
            "expected_url": "https://prokat24.com.ua/ua/dostavka/",
        }
    ]

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(cases, file, ensure_ascii=False, indent=2)

    loaded_cases = load_search_eval_cases(output_file)

    assert loaded_cases == cases


def test_evaluate_search_case_is_correct() -> None:
    delivery_text = "Доставка можлива. По Києву — від 200 грн. Доставка великогабаритних товарів."
    pages = [
        {
            "url": "https://prokat24.com.ua/ua/dostavka/",
            "title": "Доставка прокатного спорядження — Прокат24",
            "text": delivery_text,
            "text_length": len(delivery_text),
        },
        {
            "url": "https://prokat24.com.ua/ua/o-nas/",
            "title": "Про нас — Прокат24",
            "text": "Прокат спорядження та сервіс.",
            "text_length": len("Прокат спорядження та сервіс."),
        },
    ]

    case = {
        "query": "доставка київ",
        "expected_url": "https://prokat24.com.ua/ua/dostavka/",
    }

    evaluated_search_case = evaluate_search_case(case, pages)

    assert evaluated_search_case["passed"] is True
    assert evaluated_search_case["actual_url"] == case["expected_url"]
    assert evaluated_search_case["query"] == case["query"]
    assert evaluated_search_case["expected_url"] == case["expected_url"]


def test_evaluate_search_case_fails_when_no_results_found() -> None:
    pages = [
        {
            "url": "https://prokat24.com.ua/ua/o-nas/",
            "title": "Про нас — Прокат24",
            "text": "Прокат спорядження та сервіс.",
            "text_length": len("Прокат спорядження та сервіс."),
        },
    ]

    case = {
        "query": "абракадабра",
        "expected_url": "https://prokat24.com.ua/ua/dostavka/",
    }

    result = evaluate_search_case(case, pages)

    assert result["passed"] is False
    assert result["actual_url"] is None
    assert result["query"] == case["query"]
    assert result["expected_url"] == case["expected_url"]


def test_run_search_evals_is_correct() -> None:
    delivery_text = "Доставка можлива. По Києву — від 200 грн. Доставка великогабаритних товарів."
    pages = [
        {
            "url": "https://prokat24.com.ua/ua/dostavka/",
            "title": "Доставка прокатного спорядження — Прокат24",
            "text": delivery_text,
            "text_length": len(delivery_text),
        },
        {
            "url": "https://prokat24.com.ua/ua/o-nas/",
            "title": "Про нас — Прокат24",
            "text": "Прокат спорядження та сервіс.",
            "text_length": len("Прокат спорядження та сервіс."),
        },
    ]

    cases = [
        {
            "query": "доставка київ",
            "expected_url": "https://prokat24.com.ua/ua/dostavka/",
        },
        {
            "query": "прокат сервіс",
            "expected_url": "https://prokat24.com.ua/ua/o-nas/",
        },
    ]

    results = run_search_evals(cases, pages)

    assert len(results) == len(cases)
    assert results[0]["passed"] is True
    assert results[1]["passed"] is True
    assert results[0]["actual_url"] == cases[0]["expected_url"]
    assert results[1]["actual_url"] == cases[1]["expected_url"]


def test_calculate_accuracy_returns_passed_ratio() -> None:
    search_eval_results = [
        {
            "query": "доставка київ",
            "expected_url": "https://prokat24.com.ua/ua/dostavka/",
            "actual_url": "https://prokat24.com.ua/ua/dostavka/",
            "passed": True,
        },
        {
            "query": "прокат сервіс",
            "expected_url": "https://prokat24.com.ua/ua/o-nas/",
            "actual_url": "https://prokat24.com.ua/ua/o-nas/",
            "passed": True,
        },
        {
            "query": "застава паспорт",
            "expected_url": "https://prokat24.com.ua/ua/pravila-prokata/",
            "actual_url": "https://prokat24.com.ua/ua/o-nas/",
            "passed": False,
        },
    ]

    result = calculate_accuracy(search_eval_results)

    assert result == 2 / 3


def test_calculate_accuracy_returns_zero_for_empty_results() -> None:
    search_eval_results = []

    result = calculate_accuracy(search_eval_results)

    assert result == 0.0
