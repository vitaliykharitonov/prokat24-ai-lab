import json
from pathlib import Path

from prokat24_ai_lab.evals.rag_answer_evals import (
    RagAnswerEvalCase,
    RagAnswerEvalReport,
    RagAnswerEvalResult,
    build_rag_answer_eval_report,
    calculate_rag_answer_accuracy,
    count_passed_rag_answer_results,
    evaluate_rag_answer_case,
    load_rag_answer_eval_cases,
    run_rag_answer_evals,
    save_rag_answer_eval_report,
)

TEST_RESULTS: list[RagAnswerEvalResult] = [
    {
        "question": "Як отримати спорядження?",
        "expected_answer_contains": ["доставк", "отримати"],
        "missing_answer_fragments": [],
        "passed": True,
        "answer": "Спорядження можна отримати доставкою [Document 1].",
    },
    {
        "question": "Скільки коштує доставка по Києву?",
        "expected_answer_contains": ["200 грн", "Києву"],
        "missing_answer_fragments": [],
        "passed": True,
        "answer": "Доставка по Києву коштує від 200 грн [Document 1].",
    },
    {
        "question": "Чи безкоштовна доставка?",
        "expected_answer_contains": ["доставк", "безкоштовно"],
        "missing_answer_fragments": ["безкоштовно"],
        "passed": False,
        "answer": "Доставка по Києву коштує від 200 грн [Document 1].",
    },
]

TEST_CREATED_AT = "2026-01-01T00:00:00+00:00"


def test_evaluate_rag_answer_case_passes_when_answer_contains_expected_fragments() -> None:
    case: RagAnswerEvalCase = {
        "question": "Як отримати спорядження, не приїжджаючи до вас?",
        "expected_answer_contains": [
            "доставк",
            "попередньою домовленістю",
        ],
    }
    received_questions: list[str] = []
    expected_answer = (
        "Доставка можлива за попередньою домовленістю. По Києву — від 200 грн [Document 1]."
    )

    def fake_answerer(question: str) -> str:
        received_questions.append(question)
        return expected_answer

    result = evaluate_rag_answer_case(case, fake_answerer)

    assert received_questions == [case["question"]]
    assert result["passed"] is True
    assert result["missing_answer_fragments"] == []
    assert result["answer"] == expected_answer


def test_evaluate_rag_answer_case_reports_missing_fragments() -> None:
    case: RagAnswerEvalCase = {
        "question": "Як отримати спорядження, не приїжджаючи до вас?",
        "expected_answer_contains": [
            "доставк",
            "безкоштовно",
        ],
    }
    received_questions: list[str] = []
    expected_answer = (
        "Доставка можлива за попередньою домовленістю. По Києву — від 200 грн [Document 1]."
    )

    def fake_answerer(question: str) -> str:
        received_questions.append(question)
        return expected_answer

    result = evaluate_rag_answer_case(case, fake_answerer)

    assert received_questions == [case["question"]]
    assert result["answer"] == expected_answer
    assert result["passed"] is False
    assert result["missing_answer_fragments"] == ["безкоштовно"]


def test_run_rag_answer_evals_returns_result_for_each_case() -> None:
    cases: list[RagAnswerEvalCase] = [
        {
            "question": "Як отримати спорядження?",
            "expected_answer_contains": [
                "спорядження",
                "отримати",
            ],
        },
        {
            "question": "Чи безкоштовна доставка?",
            "expected_answer_contains": [
                "доставк",
                "безкоштовно",
            ],
        },
    ]

    received_questions: list[str] = []
    answers_by_question = {
        "Як отримати спорядження?": "Спорядження можна отримати доставкою [Document 1].",
        "Чи безкоштовна доставка?": "По Києву доставка коштує від 200 грн [Document 1].",
    }

    def fake_answerer(question: str) -> str:
        received_questions.append(question)
        return answers_by_question[question]

    results = run_rag_answer_evals(cases, fake_answerer)

    assert received_questions == [
        cases[0]["question"],
        cases[1]["question"],
    ]
    assert len(results) == 2
    assert results[0]["passed"] is True
    assert results[1]["passed"] is False
    assert results[1]["missing_answer_fragments"] == ["безкоштовно"]


def test_load_rag_answer_eval_cases_reads_json_file(tmp_path: Path) -> None:
    cases: list[RagAnswerEvalCase] = [
        {
            "question": "Як отримати спорядження, не приїжджаючи до вас?",
            "expected_answer_contains": ["доставк", "попередньою домовленістю"],
        },
    ]

    input_file = tmp_path / "rag_answer_evals.json"

    with open(input_file, "w", encoding="utf-8") as file:
        json.dump(cases, file, ensure_ascii=False, indent=2)

    loaded_data = load_rag_answer_eval_cases(input_file)

    assert loaded_data == cases


def test_count_passed_rag_answer_results_counts_passed_results() -> None:
    count = count_passed_rag_answer_results(TEST_RESULTS)

    assert count == 2


def test_calculate_rag_answer_accuracy_returns_passed_ratio() -> None:
    accuracy = calculate_rag_answer_accuracy(TEST_RESULTS)

    assert accuracy == 2 / 3


def test_calculate_rag_answer_accuracy_returns_zero_for_empty_results() -> None:
    accuracy = calculate_rag_answer_accuracy([])

    assert accuracy == 0.0


def test_build_rag_answer_eval_report_returns_summary_and_results() -> None:
    report = build_rag_answer_eval_report(
        TEST_RESULTS,
        TEST_CREATED_AT,
        git_commit="abc1234",
        git_dirty=False,
    )

    assert report["report_type"] == "rag_answer_eval_report"
    assert report["schema_version"] == 1
    assert report["created_at"] == TEST_CREATED_AT
    assert report["git_commit"] == "abc1234"
    assert report["git_dirty"] is False
    assert report["total_cases"] == 3
    assert report["passed_cases"] == 2
    assert report["accuracy"] == 2 / 3
    assert report["results"] == TEST_RESULTS


def test_save_rag_answer_eval_report_writes_json_file(tmp_path: Path) -> None:
    report: RagAnswerEvalReport = {
        "report_type": "rag_answer_eval_report",
        "schema_version": 1,
        "created_at": "2026-01-01T00:00:00+00:00",
        "git_commit": "abc1234",
        "git_dirty": False,
        "total_cases": 3,
        "passed_cases": 2,
        "accuracy": 2 / 3,
        "results": TEST_RESULTS,
    }
    output_file = tmp_path / "reports" / "rag_answer_eval_report.json"

    save_rag_answer_eval_report(report, output_file)

    with open(output_file, encoding="utf-8") as file:
        loaded_report = json.load(file)

    assert output_file.exists()
    assert loaded_report == report
