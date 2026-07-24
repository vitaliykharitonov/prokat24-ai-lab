import json
from pathlib import Path

from prokat24_ai_lab.evals.answer_evals import (
    build_answer_eval_report,
    calculate_answer_accuracy,
    count_passed_answer_results,
    evaluate_answer_case,
    load_answer_eval_cases,
    run_answer_evals,
    save_answer_eval_report,
)

TEST_PAGES = [
    {
        "url": "https://prokat24.com.ua/ua/dostavka/",
        "title": "Доставка прокатного спорядження — Прокат24",
        "text": "Доставка можлива. По Києву — від 200 грн.",
        "text_length": len("Доставка можлива. По Києву — від 200 грн."),
    },
]

TEST_RESULTS = [
    {
        "question": "що залишити в заставу",
        "expected_answer_type": "extractive_snippet",
        "actual_answer_type": "extractive_snippet",
        "expected_source_url": "https://prokat24.com.ua/ua/pravila-prokata/",
        "actual_source_url": "https://prokat24.com.ua/ua/pravila-prokata/",
        "expected_answer_contains": ["Грошова застава", "паспорт"],
        "missing_answer_fragments": [],
        "passed": True,
        "answer": "Грошова застава. Закордонний паспорт.",
    },
    {
        "question": "скільки коштує доставка по Києву",
        "expected_answer_type": "extractive_snippet",
        "actual_answer_type": "extractive_snippet",
        "expected_source_url": "https://prokat24.com.ua/ua/dostavka/",
        "actual_source_url": "https://prokat24.com.ua/ua/dostavka/",
        "expected_answer_contains": ["По Києву", "від 200 грн"],
        "missing_answer_fragments": [],
        "passed": True,
        "answer": "По Києву — від 200 грн.",
    },
    {
        "question": "де вас знайти",
        "expected_answer_type": "extractive_snippet",
        "actual_answer_type": "extractive_snippet",
        "expected_source_url": "https://prokat24.com.ua/ua/contact-us/",
        "actual_source_url": "https://prokat24.com.ua/ua/contact-us/",
        "expected_answer_contains": ["вул. Табірна", "вул. Васильківська"],
        "missing_answer_fragments": ["вул. Васильківська"],
        "passed": False,
        "answer": "вул. Табірна, 42",
    },
]

TEST_CREATED_AT = "2026-01-01T00:00:00+00:00"


def test_load_answer_eval_cases_reads_json_file(tmp_path: Path) -> None:
    output_file = tmp_path / "answer_eval_cases.json"
    cases = [
        {
            "question": "що залишити в заставу",
            "expected_answer_type": "extractive_snippet",
            "expected_source_url": "https://prokat24.com.ua/ua/pravila-prokata/",
            "expected_answer_contains": ["Грошова застава", "паспорт"],
        }
    ]

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(cases, file, ensure_ascii=False, indent=2)

    loaded_cases = load_answer_eval_cases(output_file)

    assert loaded_cases == cases


def test_evaluate_answer_case_passes_when_answer_matches_expectations() -> None:
    pages = TEST_PAGES

    case = {
        "question": "скільки коштує доставка по Києву",
        "expected_answer_type": "extractive_snippet",
        "expected_source_url": "https://prokat24.com.ua/ua/dostavka/",
        "expected_answer_contains": ["По Києву", "від 200 грн"],
    }

    result = evaluate_answer_case(case, pages)

    assert result["passed"] is True
    assert result["actual_source_url"] == case["expected_source_url"]
    assert result["actual_answer_type"] == case["expected_answer_type"]
    assert result["missing_answer_fragments"] == []
    assert "По Києву" in result["answer"]
    assert "від 200 грн" in result["answer"]


def test_evaluate_answer_case_reports_missing_answer_fragments() -> None:
    pages = TEST_PAGES

    case = {
        "question": "скільки коштує доставка по Києву",
        "expected_answer_type": "extractive_snippet",
        "expected_source_url": "https://prokat24.com.ua/ua/dostavka/",
        "expected_answer_contains": ["По Києву", "від 200 грн", "безкоштовно"],
    }

    result = evaluate_answer_case(case, pages)

    assert result["passed"] is False
    assert result["missing_answer_fragments"] == ["безкоштовно"]


def test_evaluate_answer_case_matches_answer_fragments_case_insensitively() -> None:
    pages = TEST_PAGES

    case = {
        "question": "скільки коштує доставка по Києву",
        "expected_answer_type": "extractive_snippet",
        "expected_source_url": "https://prokat24.com.ua/ua/dostavka/",
        "expected_answer_contains": ["по Києву"],
    }

    result = evaluate_answer_case(case, pages)

    assert "По Києву" in result["answer"]
    assert "по Києву" in result["expected_answer_contains"]
    assert result["passed"] is True
    assert result["missing_answer_fragments"] == []


def test_run_answer_evals_returns_result_for_each_case() -> None:
    pages = TEST_PAGES

    cases = [
        {
            "question": "скільки коштує доставка по Києву",
            "expected_answer_type": "extractive_snippet",
            "expected_source_url": "https://prokat24.com.ua/ua/dostavka/",
            "expected_answer_contains": ["По Києву", "від 200 грн"],
        },
        {
            "question": "абракадабра невідомеслово",
            "expected_answer_type": "not_found",
            "expected_source_url": None,
            "expected_answer_contains": ["Не знайшов відповідь"],
        },
    ]

    result = run_answer_evals(cases, pages)

    assert len(result) == 2
    assert result[0]["passed"] is True
    assert result[1]["passed"] is True
    assert result[0]["question"] == cases[0]["question"]
    assert result[1]["question"] == cases[1]["question"]


def test_count_passed_answer_results_counts_passed_results() -> None:
    results = TEST_RESULTS

    count = count_passed_answer_results(results)

    assert count == 2


def test_calculate_answer_accuracy_returns_passed_ratio() -> None:
    results = TEST_RESULTS

    accuracy = calculate_answer_accuracy(results)

    assert accuracy == 2 / 3


def test_calculate_answer_accuracy_returns_zero_for_empty_results() -> None:
    results = []

    accuracy = calculate_answer_accuracy(results)

    assert accuracy == 0.0


def test_build_answer_eval_report_returns_summary_and_results() -> None:
    report = build_answer_eval_report(
        TEST_RESULTS,
        TEST_CREATED_AT,
        git_commit="abc1234",
        git_dirty=False,
    )

    assert report["report_type"] == "answer_eval_report"
    assert report["schema_version"] == 1
    assert report["created_at"] == TEST_CREATED_AT
    assert report["git_commit"] == "abc1234"
    assert report["git_dirty"] is False
    assert report["total_cases"] == 3
    assert report["passed_cases"] == 2
    assert report["accuracy"] == 2 / 3
    assert report["results"] == TEST_RESULTS


def test_save_answer_eval_report_writes_json_file(tmp_path: Path) -> None:
    report = build_answer_eval_report(TEST_RESULTS, TEST_CREATED_AT)

    output_file = tmp_path / "reports" / "answer_eval_report.json"

    save_answer_eval_report(report, output_file)

    with open(output_file, encoding="utf-8") as file:
        loaded_report = json.load(file)

    assert loaded_report == report
