import json
from pathlib import Path

from prokat24_ai_lab.answer_evals import (
    evaluate_answer_case,
    load_answer_eval_cases,
)


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
    pages = [
        {
            "url": "https://prokat24.com.ua/ua/dostavka/",
            "title": "Доставка прокатного спорядження — Прокат24",
            "text": "Доставка можлива. По Києву — від 200 грн.",
            "text_length": len("Доставка можлива. По Києву — від 200 грн."),
        },
    ]

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
    pages = [
        {
            "url": "https://prokat24.com.ua/ua/dostavka/",
            "title": "Доставка прокатного спорядження — Прокат24",
            "text": "Доставка можлива. По Києву — від 200 грн.",
            "text_length": len("Доставка можлива. По Києву — від 200 грн."),
        },
    ]

    case = {
        "question": "скільки коштує доставка по Києву",
        "expected_answer_type": "extractive_snippet",
        "expected_source_url": "https://prokat24.com.ua/ua/dostavka/",
        "expected_answer_contains": ["По Києву", "від 200 грн", "безкоштовно"],
    }

    result = evaluate_answer_case(case, pages)

    assert result["passed"] is False
    assert result["missing_answer_fragments"] == ["безкоштовно"]
