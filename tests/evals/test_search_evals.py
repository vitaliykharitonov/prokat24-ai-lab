import json
from pathlib import Path

from prokat24_ai_lab.evals.retrieval_cases import RetrievalEvalCase
from prokat24_ai_lab.evals.search_evals import (
    build_search_eval_report,
    calculate_accuracy,
    count_passed_results,
    evaluate_search_case,
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
        "id": "retrieval_test_001",
        "category": "delivery",
        "difficulty": "easy",
        "split": "dev",
        "query": "доставка київ",
        "query_words": ["доставка", "київ"],
        "expected_document_id": "https://prokat24.com.ua/ua/dostavka/",
        "actual_document_id": "https://prokat24.com.ua/ua/dostavka/",
        "actual_title": "Доставка прокатного спорядження — Прокат24",
        "actual_score": 2.0,
        "expected_rank": 1,
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
        "id": "retrieval_test_002",
        "category": "company_services",
        "difficulty": "easy",
        "split": "dev",
        "query": "прокат сервіс",
        "query_words": ["прокат", "сервіс"],
        "expected_document_id": "https://prokat24.com.ua/ua/o-nas/",
        "actual_document_id": "https://prokat24.com.ua/ua/o-nas/",
        "actual_title": "Про нас — Прокат24",
        "actual_score": 2.0,
        "expected_rank": 1,
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
        "id": "retrieval_test_003",
        "category": "deposit",
        "difficulty": "medium",
        "split": "dev",
        "query": "застава паспорт",
        "query_words": ["застава", "паспорт"],
        "expected_document_id": "https://prokat24.com.ua/ua/pravila-prokata/",
        "actual_document_id": "https://prokat24.com.ua/ua/o-nas/",
        "actual_title": "Про нас — Прокат24",
        "actual_score": 1.0,
        "expected_rank": None,
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

TEST_SEARCH_EVAL_REPORT = {
    "report_type": "search_eval_report",
    "schema_version": 1,
    "created_at": "2026-01-01T00:00:00+00:00",
    "git_commit": "abc1234",
    "git_dirty": False,
    "total_cases": 3,
    "passed_cases": 2,
    "accuracy": 2 / 3,
    "top_3_accuracy": 2 / 3,
    "results": TEST_SEARCH_EVAL_RESULTS,
}


def test_evaluate_search_case_is_correct() -> None:
    case: RetrievalEvalCase = {
        "id": "retrieval_test_001",
        "query": "скільки коштує привезти",
        "expected_document_id": "https://prokat24.com.ua/ua/dostavka/",
        "category": "delivery",
        "difficulty": "easy",
        "split": "dev",
    }

    evaluated_search_case = evaluate_search_case(case, TEST_PAGES)
    top_results = evaluated_search_case["top_results"]

    assert evaluated_search_case["id"] == case["id"]
    assert evaluated_search_case["category"] == case["category"]
    assert evaluated_search_case["difficulty"] == case["difficulty"]
    assert evaluated_search_case["split"] == case["split"]
    assert evaluated_search_case["passed"] is True
    assert evaluated_search_case["actual_document_id"] == case["expected_document_id"]
    assert evaluated_search_case["query"] == case["query"]
    assert "доставка" in evaluated_search_case["query_words"]
    assert "привезти" in evaluated_search_case["query_words"]
    assert evaluated_search_case["expected_document_id"] == case["expected_document_id"]
    assert evaluated_search_case["actual_title"] == ("Доставка прокатного спорядження — Прокат24")
    assert evaluated_search_case["actual_score"] == 3.0
    assert evaluated_search_case["expected_rank"] == 1
    assert len(top_results) == 1
    assert top_results[0]["url"] == "https://prokat24.com.ua/ua/dostavka/"
    assert top_results[0]["title"] == "Доставка прокатного спорядження — Прокат24"
    assert top_results[0]["score"] == 3.0
    assert "Доставка" in top_results[0]["snippet"]


def test_evaluate_search_case_fails_when_no_results_found() -> None:
    case: RetrievalEvalCase = {
        "id": "retrieval_test_002",
        "query": "абракадабра",
        "expected_document_id": "https://prokat24.com.ua/ua/dostavka/",
        "category": "delivery",
        "difficulty": "hard",
        "split": "dev",
    }

    result = evaluate_search_case(case, TEST_PAGES)

    assert result["passed"] is False
    assert result["actual_document_id"] is None
    assert result["query"] == case["query"]
    assert result["expected_document_id"] == case["expected_document_id"]
    assert result["actual_title"] is None
    assert result["actual_score"] is None
    assert result["expected_rank"] is None
    assert result["top_results"] == []


def test_run_search_evals_is_correct() -> None:
    cases: list[RetrievalEvalCase] = [
        {
            "id": "retrieval_test_001",
            "query": "доставка київ",
            "expected_document_id": "https://prokat24.com.ua/ua/dostavka/",
            "category": "delivery",
            "difficulty": "easy",
            "split": "dev",
        },
        {
            "id": "retrieval_test_002",
            "query": "прокат сервіс",
            "expected_document_id": "https://prokat24.com.ua/ua/o-nas/",
            "category": "company_services",
            "difficulty": "easy",
            "split": "dev",
        },
    ]

    results = run_search_evals(cases, TEST_PAGES)

    assert len(results) == len(cases)
    assert results[0]["passed"] is True
    assert results[1]["passed"] is True
    assert results[0]["actual_document_id"] == cases[0]["expected_document_id"]
    assert results[1]["actual_document_id"] == cases[1]["expected_document_id"]


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
    report = build_search_eval_report(
        TEST_SEARCH_EVAL_RESULTS,
        created_at="2026-01-01T00:00:00+00:00",
        git_commit="abc1234",
        git_dirty=False,
    )

    assert report == TEST_SEARCH_EVAL_REPORT


def test_save_search_eval_report_writes_json_file(tmp_path: Path) -> None:
    output_file = tmp_path / "search_eval_report.json"

    save_search_eval_report(TEST_SEARCH_EVAL_REPORT, output_file)

    assert output_file.exists()

    with open(output_file, encoding="utf-8") as file:
        loaded_report = json.load(file)

    assert loaded_report == TEST_SEARCH_EVAL_REPORT
