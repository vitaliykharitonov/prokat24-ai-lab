import json
from pathlib import Path

from prokat24_ai_lab.search_evals import (
    SearchEvalCase,
    build_search_eval_report,
    calculate_accuracy,
    count_passed_results,
    evaluate_search_case,
    load_search_eval_cases,
    run_search_evals,
    save_search_eval_report,
)

TEST_PAGES = [
    {
        "url": "https://prokat24.com.ua/ua/dostavka/",
        "title": "Доставка прокатного спорядження — Прокат24",
        "text": "Доставка можлива. По Києву — від 200 грн. Доставка великогабаритних товарів.",
        "text_length": len(
            "Доставка можлива. По Києву — від 200 грн. Доставка великогабаритних товарів."
        ),
    },
    {
        "url": "https://prokat24.com.ua/ua/o-nas/",
        "title": "Про нас — Прокат24",
        "text": "Прокат спорядження та сервіс.",
        "text_length": len("Прокат спорядження та сервіс."),
    },
]

TEST_SEARCH_EVAL_RESULTS = [
    {
        "query": "доставка київ",
        "expected_url": "https://prokat24.com.ua/ua/dostavka/",
        "actual_url": "https://prokat24.com.ua/ua/dostavka/",
        "actual_title": "Доставка прокатного спорядження — Прокат24",
        "actual_score": 2.0,
        "passed": True,
        "top_results": [
            {
                "url": "https://prokat24.com.ua/ua/dostavka/",
                "title": "Доставка прокатного спорядження — Прокат24",
                "snippet": "Доставка можлива. По Києву — від 200 грн. Доставка товарів.",
                "score": 2.0,
            },
        ],
    },
    {
        "query": "прокат сервіс",
        "expected_url": "https://prokat24.com.ua/ua/o-nas/",
        "actual_url": "https://prokat24.com.ua/ua/o-nas/",
        "actual_title": "Про нас — Прокат24",
        "actual_score": 2.0,
        "passed": True,
        "top_results": [
            {
                "url": "https://prokat24.com.ua/ua/o-nas/",
                "title": "Про нас — Прокат24",
                "snippet": "Прокат спорядження та сервіс.",
                "score": 2.0,
            },
        ],
    },
    {
        "query": "застава паспорт",
        "expected_url": "https://prokat24.com.ua/ua/pravila-prokata/",
        "actual_url": "https://prokat24.com.ua/ua/o-nas/",
        "actual_title": "Про нас — Прокат24",
        "actual_score": 1.0,
        "passed": False,
        "top_results": [
            {
                "url": "https://prokat24.com.ua/ua/o-nas/",
                "title": "Про нас — Прокат24",
                "snippet": "Прокат спорядження та сервіс.",
                "score": 1.0,
            },
        ],
    },
]

TEST_CREATED_AT = "2026-06-04T17:45:08"

TEST_SEARCH_EVAL_REPORT = {
    "created_at": TEST_CREATED_AT,
    "total_cases": 3,
    "passed_cases": 2,
    "accuracy": 2 / 3,
    "top_3_accuracy": 2 / 3,
    "results": TEST_SEARCH_EVAL_RESULTS,
}


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
    case = {
        "query": "скільки коштує привезти",
        "expected_url": "https://prokat24.com.ua/ua/dostavka/",
    }

    evaluated_search_case = evaluate_search_case(case, TEST_PAGES)
    top_results = evaluated_search_case["top_results"]

    assert evaluated_search_case["passed"] is True
    assert evaluated_search_case["actual_url"] == case["expected_url"]
    assert evaluated_search_case["query"] == case["query"]
    assert "доставка" in evaluated_search_case["query_words"]
    assert "привезти" in evaluated_search_case["query_words"]
    assert evaluated_search_case["expected_url"] == case["expected_url"]
    assert evaluated_search_case["actual_title"] == "Доставка прокатного спорядження — Прокат24"
    assert evaluated_search_case["actual_score"] == 3.0
    assert len(top_results) == 1
    assert top_results[0]["url"] == "https://prokat24.com.ua/ua/dostavka/"
    assert top_results[0]["title"] == "Доставка прокатного спорядження — Прокат24"
    assert top_results[0]["score"] == 3.0
    assert "Доставка" in top_results[0]["snippet"]


def test_evaluate_search_case_fails_when_no_results_found() -> None:
    case = {
        "query": "абракадабра",
        "expected_url": "https://prokat24.com.ua/ua/dostavka/",
    }

    result = evaluate_search_case(case, TEST_PAGES)

    assert result["passed"] is False
    assert result["actual_url"] is None
    assert result["query"] == case["query"]
    assert result["expected_url"] == case["expected_url"]
    assert result["actual_title"] is None
    assert result["actual_score"] is None
    assert result["top_results"] == []


def test_run_search_evals_is_correct() -> None:
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

    results = run_search_evals(cases, TEST_PAGES)

    assert len(results) == len(cases)
    assert results[0]["passed"] is True
    assert results[1]["passed"] is True
    assert results[0]["actual_url"] == cases[0]["expected_url"]
    assert results[1]["actual_url"] == cases[1]["expected_url"]


def test_calculate_accuracy_returns_passed_ratio() -> None:
    result = calculate_accuracy(TEST_SEARCH_EVAL_RESULTS)

    assert result == 2 / 3


def test_calculate_accuracy_returns_zero_for_empty_results() -> None:
    empty_search_eval_results = []

    result = calculate_accuracy(empty_search_eval_results)

    assert result == 0.0


def test_count_passed_results_returns_number_of_passed_results() -> None:
    result = count_passed_results(TEST_SEARCH_EVAL_RESULTS)

    assert result == 2


def test_build_search_eval_report_returns_summary() -> None:
    report = build_search_eval_report(TEST_SEARCH_EVAL_RESULTS, TEST_CREATED_AT)

    assert report == TEST_SEARCH_EVAL_REPORT


def test_save_search_eval_report_writes_json_file(tmp_path: Path) -> None:
    output_file = tmp_path / "search_eval_report.json"

    save_search_eval_report(TEST_SEARCH_EVAL_REPORT, output_file)

    assert output_file.exists()

    with open(output_file, encoding="utf-8") as file:
        loaded_report = json.load(file)

    assert loaded_report == TEST_SEARCH_EVAL_REPORT
