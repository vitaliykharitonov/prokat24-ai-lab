import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from prokat24_ai_lab.evals.semantic_grading import (
    GraderComparisonCase,
    GraderComparisonResult,
    GraderComparisonSummary,
    JudgeEvalCase,
    JudgeInput,
    JudgeResult,
    Rubric,
    build_judge_prompt,
    evaluate_grader_comparison_case,
    evaluate_judge_case,
    evaluate_with_judge,
    load_grader_comparison_cases,
    load_judge_eval_cases,
    run_grader_comparison_evals,
    run_judge_evals,
    summarize_grader_comparison,
    summarize_grader_comparison_by_category,
)


def test_judge_result_accepts_valid_score() -> None:
    judge_result = JudgeResult(
        score=5,
        passed=True,
        reason="The answer correctly states the delivery price.",
        failed_criteria=[],
    )

    assert judge_result.score == 5
    assert judge_result.passed is True
    assert judge_result.reason == "The answer correctly states the delivery price."
    assert judge_result.failed_criteria == []


def test_judge_result_rejects_score_below_minimum() -> None:
    with pytest.raises(ValidationError):
        JudgeResult(
            score=0,
            passed=True,
            reason="The answer correctly states the delivery price.",
            failed_criteria=[],
        )


def test_judge_result_rejects_score_above_maximum() -> None:
    with pytest.raises(ValidationError):
        JudgeResult(
            score=6,
            passed=True,
            reason="The answer correctly states the delivery price.",
            failed_criteria=[],
        )


def test_evaluate_with_judge() -> None:
    judge_input: JudgeInput = {
        "question": "Чи безкоштовна доставка?",
        "context": "Доставка по Києву — від 200 грн.",
        "answer": "Доставка по Києву коштує від 200 грн.",
    }
    rubric: Rubric = {
        "name": "delivery_answer",
        "criteria": [
            {
                "name": "delivery_correctness",
                "description": "The answer must correctly state the delivery price.",
            }
        ],
    }

    judge_result: JudgeResult = JudgeResult(
        score=5,
        passed=True,
        reason="The answer correctly states the delivery price.",
        failed_criteria=[],
    )
    received_arguments: list[tuple[JudgeInput, Rubric]] = []

    def fake_judge(input_data: JudgeInput, received_rubric: Rubric) -> JudgeResult:
        received_arguments.append((input_data, received_rubric))
        return judge_result

    result = evaluate_with_judge(judge_input, rubric, fake_judge)

    assert result == judge_result
    assert received_arguments == [
        (
            {
                "question": "Чи безкоштовна доставка?",
                "context": "Доставка по Києву — від 200 грн.",
                "answer": "Доставка по Києву коштує від 200 грн.",
            },
            {
                "name": "delivery_answer",
                "criteria": [
                    {
                        "name": "delivery_correctness",
                        "description": "The answer must correctly state the delivery price.",
                    }
                ],
            },
        )
    ]


def test_build_judge_prompt_includes_evaluation_data() -> None:
    judge_input: JudgeInput = {
        "question": "Чи безкоштовна доставка?",
        "context": "Доставка по Києву — від 200 грн.",
        "answer": "Доставка по Києву коштує від 200 грн.",
    }
    rubric: Rubric = {
        "name": "delivery_answer",
        "criteria": [
            {
                "name": "delivery_correctness",
                "description": "The answer must correctly state the delivery price.",
            },
            {
                "name": "groundedness",
                "description": "The answer must not introduce facts unsupported by the context",
            },
        ],
    }

    prompt = build_judge_prompt(judge_input, rubric)

    assert "Чи безкоштовна доставка?" in prompt
    assert "Доставка по Києву — від 200 грн." in prompt
    assert "Доставка по Києву коштує від 200 грн." in prompt
    assert "delivery_answer" in prompt
    assert "delivery_correctness" in prompt
    assert "groundedness" in prompt
    assert "The answer must correctly state the delivery price." in prompt
    assert "The answer must not introduce facts unsupported by the context" in prompt
    assert "Do not use outside knowledge." in prompt


def test_evaluate_judge_case_returns_true_when_judge_matches_expected() -> None:
    case: JudgeEvalCase = {
        "judge_input": {
            "question": "Чи безкоштовна доставка?",
            "context": "Доставка по Києву — від 200 грн.",
            "answer": "Доставка коштує від 200 грн.",
        },
        "rubric": {
            "name": "delivery_answer",
            "criteria": [
                {
                    "name": "correctness",
                    "description": "The answer must correctly state the delivery price.",
                }
            ],
        },
        "expected_passed": True,
    }
    judge_result: JudgeResult = JudgeResult(
        score=5,
        passed=True,
        reason="The answer correctly states the delivery price.",
        failed_criteria=[],
    )
    received_arguments: list[tuple[JudgeInput, Rubric]] = []

    def fake_judge(judge_input: JudgeInput, rubric: Rubric) -> JudgeResult:
        received_arguments.append((judge_input, rubric))
        return judge_result

    result = evaluate_judge_case(case, fake_judge)

    assert received_arguments == [
        (
            {
                "question": "Чи безкоштовна доставка?",
                "context": "Доставка по Києву — від 200 грн.",
                "answer": "Доставка коштує від 200 грн.",
            },
            {
                "name": "delivery_answer",
                "criteria": [
                    {
                        "name": "correctness",
                        "description": "The answer must correctly state the delivery price.",
                    }
                ],
            },
        )
    ]
    assert result["expected_passed"] is True
    assert result["actual_passed"] is True
    assert result["matched"] is True
    assert result["score"] == 5
    assert result["reason"] == "The answer correctly states the delivery price."
    assert result["failed_criteria"] == []


def test_evaluate_judge_case_returns_false_when_judge_does_not_match_expected() -> None:
    case: JudgeEvalCase = {
        "judge_input": {
            "question": "Чи безкоштовна доставка?",
            "context": "Доставка по Києву — від 200 грн.",
            "answer": "Доставка коштує від 200 грн.",
        },
        "rubric": {
            "name": "delivery_answer",
            "criteria": [
                {
                    "name": "correctness",
                    "description": "The answer must correctly state the delivery price.",
                }
            ],
        },
        "expected_passed": True,
    }
    judge_result: JudgeResult = JudgeResult(
        score=1,
        passed=False,
        reason="The answer is incorrect.",
        failed_criteria=["correctness"],
    )

    def fake_judge(_judge_input: JudgeInput, _rubric: Rubric) -> JudgeResult:
        return judge_result

    result = evaluate_judge_case(case, fake_judge)

    assert result["expected_passed"] is True
    assert result["actual_passed"] is False
    assert result["matched"] is False
    assert result["score"] == 1
    assert result["reason"] == "The answer is incorrect."
    assert result["failed_criteria"] == ["correctness"]


def test_run_judge_evals_evaluates_all_cases() -> None:
    cases: list[JudgeEvalCase] = [
        {
            "judge_input": {
                "question": "Чи безкоштовна доставка?",
                "context": "Доставка по Києву — від 200 грн.",
                "answer": "Доставка коштує від 200 грн.",
            },
            "rubric": {
                "name": "delivery_answer",
                "criteria": [
                    {
                        "name": "correctness",
                        "description": "The answer must correctly state the delivery price.",
                    }
                ],
            },
            "expected_passed": True,
        },
        {
            "judge_input": {
                "question": "Чи можна залишити паспорт як заставу?",
                "context": "Громадянський паспорт не можна залишати як заставу.",
                "answer": "Так, паспорт можна залишити як заставу.",
            },
            "rubric": {
                "name": "deposit_answer",
                "criteria": [
                    {
                        "name": "correctness",
                        "description": "The answer must correctly state whether a passport "
                        "can be left as collateral.",
                    }
                ],
            },
            "expected_passed": False,
        },
    ]
    judge_results = [
        JudgeResult(
            score=5,
            passed=True,
            reason="The answer correctly states the delivery price.",
            failed_criteria=[],
        ),
        JudgeResult(
            score=5,
            passed=True,
            reason="The answer is considered correct by the judge.",
            failed_criteria=[],
        ),
    ]

    received_arguments: list[tuple[JudgeInput, Rubric]] = []

    def fake_judge(judge_input: JudgeInput, rubric: Rubric) -> JudgeResult:
        received_arguments.append((judge_input, rubric))

        return judge_results[len(received_arguments) - 1]

    results = run_judge_evals(cases, fake_judge)

    assert len(results) == 2
    assert results[0]["matched"] is True
    assert results[1]["matched"] is False


def test_load_judge_eval_cases_reads_json_file(tmp_path: Path) -> None:
    cases: list[JudgeEvalCase] = [
        {
            "judge_input": {
                "question": "Чи безкоштовна доставка?",
                "context": "Доставка по Києву — від 200 грн.",
                "answer": "Доставка коштує від 200 грн.",
            },
            "rubric": {
                "name": "delivery_answer",
                "criteria": [
                    {
                        "name": "correctness",
                        "description": "The answer must correctly state the delivery price.",
                    }
                ],
            },
            "expected_passed": True,
        },
        {
            "judge_input": {
                "question": "Чи можна залишити паспорт як заставу?",
                "context": "Громадянський паспорт не можна залишати як заставу.",
                "answer": "Так, паспорт можна залишити як заставу.",
            },
            "rubric": {
                "name": "deposit_answer",
                "criteria": [
                    {
                        "name": "correctness",
                        "description": "The answer must correctly state whether a passport "
                        "can be left as collateral.",
                    }
                ],
            },
            "expected_passed": False,
        },
    ]
    input_file = tmp_path / "judge_eval_cases.json"

    with open(input_file, "w", encoding="utf-8") as file:
        json.dump(cases, file, ensure_ascii=False, indent=2)

    loaded_data = load_judge_eval_cases(input_file)

    assert loaded_data == cases


def test_load_judge_eval_cases_rejects_invalid_case(tmp_path: Path) -> None:
    cases = [
        {
            "judge_input": {
                "question": "Чи безкоштовна доставка?",
                "context": "Доставка по Києву — від 200 грн.",
                "answer": "Доставка коштує від 200 грн.",
            },
            "rubric": {
                "name": "delivery_answer",
                "criteria": [
                    {
                        "name": "correctness",
                        "description": "The answer must correctly state the delivery price.",
                    }
                ],
            },
        },
    ]
    input_file = tmp_path / "judge_eval_cases.json"

    with open(input_file, "w", encoding="utf-8") as file:
        json.dump(cases, file, ensure_ascii=False, indent=2)

    with pytest.raises(ValidationError):
        load_judge_eval_cases(input_file)


def test_load_grader_comparison_cases(tmp_path: Path) -> None:
    cases: list[GraderComparisonCase] = [
        {
            "id": "delivery_correct_01",
            "category": "correct",
            "judge_input": {
                "question": "Скільки коштує доставка по Києву?",
                "context": "Доставка по Києву коштує від 200 грн.",
                "answer": "Доставка по Києву коштує від 200 грн.",
            },
            "expected_passed": True,
            "deterministic": {
                "required_fragments": ["від 200 грн"],
                "forbidden_fragments": [],
            },
            "rubric": {
                "name": "delivery_answer",
                "criteria": [
                    {
                        "name": "correctness",
                        "description": "The answer must correctly state the delivery price.",
                    }
                ],
            },
        }
    ]
    input_file = tmp_path / "grader_comparison_cases.json"

    with open(input_file, "w", encoding="utf-8") as file:
        json.dump(cases, file, ensure_ascii=False, indent=2)

    loaded_data = load_grader_comparison_cases(input_file)

    assert loaded_data == cases


def test_load_grader_comparison_cases_rejects_invalid_case(tmp_path: Path) -> None:
    cases = [
        {
            "id": "delivery_correct_01",
            "judge_input": {
                "question": "Скільки коштує доставка по Києву?",
                "context": "Доставка по Києву коштує від 200 грн.",
                "answer": "Доставка по Києву коштує від 200 грн.",
            },
            "expected_passed": True,
            "deterministic": {
                "required_fragments": ["від 200 грн"],
                "forbidden_fragments": [],
            },
            "rubric": {
                "name": "delivery_answer",
                "criteria": [
                    {
                        "name": "correctness",
                        "description": "The answer must correctly state the delivery price.",
                    }
                ],
            },
        }
    ]
    input_file = tmp_path / "grader_comparison_cases.json"

    with open(input_file, "w", encoding="utf-8") as file:
        json.dump(cases, file, ensure_ascii=False, indent=2)

    with pytest.raises(ValidationError):
        load_grader_comparison_cases(input_file)


def test_evaluate_grader_comparison_case_compares_deterministic_and_llm() -> None:
    case: GraderComparisonCase = {
        "id": "delivery_paraphrase_01",
        "category": "paraphrase",
        "judge_input": {
            "question": "Скільки коштує доставка по Києву?",
            "context": "Доставка по Києву коштує від 200 грн.",
            "answer": "Мінімальна вартість доставки в межах Києва — 200 гривень.",
        },
        "expected_passed": True,
        "deterministic": {
            "required_fragments": ["від 200 грн"],
            "forbidden_fragments": [],
        },
        "rubric": {
            "name": "delivery_answer",
            "criteria": [
                {
                    "name": "correctness",
                    "description": "The answer must correctly state the delivery price in Kyiv.",
                }
            ],
        },
    }

    judge_result = JudgeResult(
        score=5,
        passed=True,
        reason="Correct paraphrase",
        failed_criteria=[],
    )

    received_arguments: list[tuple[JudgeInput, Rubric]] = []

    def fake_judge(judge_input: JudgeInput, rubric: Rubric) -> JudgeResult:
        received_arguments.append((judge_input, rubric))

        return judge_result

    result = evaluate_grader_comparison_case(case, fake_judge)

    assert result["expected_passed"] is True

    assert result["deterministic_passed"] is False
    assert result["deterministic_matched"] is False

    assert result["llm_passed"] is True
    assert result["llm_matched"] is True
    assert result["llm_score"] == 5
    assert result["llm_reason"] == "Correct paraphrase"
    assert result["llm_failed_criteria"] == []


def test_evaluate_grader_comparison_case_detects_deterministic_false_positive() -> None:
    case: GraderComparisonCase = {
        "id": "passport_negation_01",
        "category": "negation",
        "judge_input": {
            "question": "Чи можна залишити громадянський паспорт як заставу?",
            "context": "Громадянський паспорт не можна залишати як заставу.",
            "answer": "Так, громадянський паспорт можна залишити як заставу.",
        },
        "expected_passed": False,
        "deterministic": {
            "required_fragments": ["паспорт", "заставу"],
            "forbidden_fragments": [],
        },
        "rubric": {
            "name": "collateral_answer",
            "criteria": [
                {
                    "name": "correctness",
                    "description": "The answer must correctly state whether a civil passport "
                    "can be left as collateral.",
                }
            ],
        },
    }

    judge_result = JudgeResult(
        score=1,
        passed=False,
        reason="The answer contradicts the context.",
        failed_criteria=["correctness"],
    )

    def fake_judge(_judge_input: JudgeInput, _rubric: Rubric) -> JudgeResult:
        return judge_result

    result = evaluate_grader_comparison_case(case, fake_judge)

    assert result["expected_passed"] is False

    assert result["deterministic_passed"] is True
    assert result["deterministic_matched"] is False

    assert result["llm_passed"] is False
    assert result["llm_matched"] is True
    assert result["llm_score"] == 1
    assert result["llm_reason"] == "The answer contradicts the context."
    assert result["llm_failed_criteria"] == ["correctness"]


def test_run_grader_comparison_evals() -> None:
    cases: list[GraderComparisonCase] = [
        {
            "id": "delivery_correct_01",
            "category": "correct",
            "judge_input": {
                "question": "Скільки коштує доставка по Києву?",
                "context": "Доставка по Києву коштує від 200 грн.",
                "answer": "Доставка по Києву коштує від 200 грн.",
            },
            "expected_passed": True,
            "deterministic": {
                "required_fragments": ["від 200 грн"],
                "forbidden_fragments": [],
            },
            "rubric": {
                "name": "delivery_answer",
                "criteria": [
                    {
                        "name": "correctness",
                        "description": "The answer must correctly state the delivery price.",
                    }
                ],
            },
        },
        {
            "id": "delivery_incorrect_01",
            "category": "incorrect",
            "judge_input": {
                "question": "Скільки коштує доставка по Києву?",
                "context": "Доставка по Києву коштує від 200 грн.",
                "answer": "Доставка по Києву безкоштовна.",
            },
            "expected_passed": False,
            "deterministic": {
                "required_fragments": ["від 200 грн"],
                "forbidden_fragments": ["безкоштовна"],
            },
            "rubric": {
                "name": "delivery_answer",
                "criteria": [
                    {
                        "name": "correctness",
                        "description": "The answer must correctly state the delivery price.",
                    }
                ],
            },
        },
    ]

    judge_results = [
        JudgeResult(
            score=5,
            passed=True,
            reason="The answer is correct",
            failed_criteria=[],
        ),
        JudgeResult(
            score=1,
            passed=False,
            reason="The answer is incorrect",
            failed_criteria=["correctness"],
        ),
    ]

    received_arguments: list[tuple[JudgeInput, Rubric]] = []

    def fake_judge(judge_input: JudgeInput, rubric: Rubric) -> JudgeResult:
        received_arguments.append((judge_input, rubric))

        return judge_results[len(received_arguments) - 1]

    results = run_grader_comparison_evals(cases, fake_judge)

    assert len(results) == 2
    assert received_arguments == [
        (cases[0]["judge_input"], cases[0]["rubric"]),
        (cases[1]["judge_input"], cases[1]["rubric"]),
    ]

    assert results[0]["deterministic_passed"] is True
    assert results[0]["deterministic_matched"] is True
    assert results[0]["llm_passed"] is True
    assert results[0]["llm_matched"] is True

    assert results[1]["deterministic_passed"] is False
    assert results[1]["deterministic_matched"] is True
    assert results[1]["llm_passed"] is False
    assert results[1]["llm_matched"] is True


def test_summarize_grader_comparison() -> None:
    results: list[GraderComparisonResult] = [
        {
            "id": "case_1",
            "category": "correct",
            "expected_passed": True,
            "deterministic_passed": True,
            "deterministic_matched": True,
            "llm_passed": True,
            "llm_matched": True,
            "llm_score": 5,
            "llm_reason": "Test result",
            "llm_failed_criteria": [],
        },
        {
            "id": "case_2",
            "category": "paraphrase",
            "expected_passed": True,
            "deterministic_passed": False,
            "deterministic_matched": False,
            "llm_passed": True,
            "llm_matched": True,
            "llm_score": 5,
            "llm_reason": "Test result",
            "llm_failed_criteria": [],
        },
        {
            "id": "case_3",
            "category": "negation",
            "expected_passed": False,
            "deterministic_passed": True,
            "deterministic_matched": False,
            "llm_passed": False,
            "llm_matched": True,
            "llm_score": 5,
            "llm_reason": "Test result",
            "llm_failed_criteria": [],
        },
        {
            "id": "case_4",
            "category": "incorrect",
            "expected_passed": False,
            "deterministic_passed": False,
            "deterministic_matched": True,
            "llm_passed": True,
            "llm_matched": False,
            "llm_score": 5,
            "llm_reason": "Test result",
            "llm_failed_criteria": [],
        },
    ]
    expected_summary: GraderComparisonSummary = {
        "total_cases": 4,
        "deterministic_matched": 2,
        "llm_matched": 3,
        "deterministic_accuracy": 0.5,
        "llm_accuracy": 0.75,
    }

    summary = summarize_grader_comparison(results)

    assert summary == expected_summary


def test_summarize_grader_comparison_empty_results() -> None:
    results: list[GraderComparisonResult] = []

    expected_summary: GraderComparisonSummary = {
        "total_cases": 0,
        "deterministic_matched": 0,
        "llm_matched": 0,
        "deterministic_accuracy": 0.0,
        "llm_accuracy": 0.0,
    }

    summary = summarize_grader_comparison(results)

    assert summary == expected_summary


def test_summarize_grader_comparison_by_category() -> None:
    results: list[GraderComparisonResult] = [
        {
            "id": "case_1",
            "category": "correct",
            "expected_passed": True,
            "deterministic_passed": True,
            "deterministic_matched": True,
            "llm_passed": True,
            "llm_matched": True,
            "llm_score": 5,
            "llm_reason": "Test result",
            "llm_failed_criteria": [],
        },
        {
            "id": "case_2",
            "category": "paraphrase",
            "expected_passed": True,
            "deterministic_passed": False,
            "deterministic_matched": False,
            "llm_passed": True,
            "llm_matched": True,
            "llm_score": 5,
            "llm_reason": "Test result",
            "llm_failed_criteria": [],
        },
        {
            "id": "case_3",
            "category": "paraphrase",
            "expected_passed": True,
            "deterministic_passed": True,
            "deterministic_matched": True,
            "llm_passed": False,
            "llm_matched": False,
            "llm_score": 5,
            "llm_reason": "Test result",
            "llm_failed_criteria": [],
        },
        {
            "id": "case_4",
            "category": "negation",
            "expected_passed": False,
            "deterministic_passed": True,
            "deterministic_matched": False,
            "llm_passed": False,
            "llm_matched": True,
            "llm_score": 5,
            "llm_reason": "Test result",
            "llm_failed_criteria": [],
        },
    ]
    expected = {
        "correct": {
            "total_cases": 1,
            "deterministic_matched": 1,
            "llm_matched": 1,
            "deterministic_accuracy": 1.0,
            "llm_accuracy": 1.0,
        },
        "paraphrase": {
            "total_cases": 2,
            "deterministic_matched": 1,
            "llm_matched": 1,
            "deterministic_accuracy": 0.5,
            "llm_accuracy": 0.5,
        },
        "negation": {
            "total_cases": 1,
            "deterministic_matched": 0,
            "llm_matched": 1,
            "deterministic_accuracy": 0.0,
            "llm_accuracy": 1.0,
        },
    }

    summaries = summarize_grader_comparison_by_category(results)

    assert summaries == expected


def test_summarize_grader_comparison_by_category_empty_results() -> None:
    results: list[GraderComparisonResult] = []

    summaries = summarize_grader_comparison_by_category(results)

    assert summaries == {}
