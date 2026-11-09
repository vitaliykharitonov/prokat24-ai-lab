import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from prokat24_ai_lab.evals.grading.comparison_runner import (
    GraderComparisonCase,
    evaluate_grader_comparison_case,
    load_grader_comparison_cases,
    run_grader_comparison_evals,
)
from prokat24_ai_lab.evals.grading.judge import (
    CriterionJudgeResult,
    CriterionResult,
    JudgeInput,
    JudgeResult,
    Rubric,
)
from prokat24_ai_lab.project_paths import (
    GRADER_COMPARISON_CASES_FILE,
    GRADER_COMPARISON_HOLDOUT_CASES_FILE,
)


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
    criterion_judge_result = CriterionJudgeResult(
        criteria=[
            CriterionResult(
                name="correctness",
                passed=True,
                reason="The delivery price is correct.",
            ),
        ]
    )

    received_judge_arguments: list[tuple[JudgeInput, Rubric]] = []
    received_criterion_judge_arguments: list[tuple[JudgeInput, Rubric]] = []

    def fake_judge(judge_input: JudgeInput, rubric: Rubric) -> JudgeResult:
        received_judge_arguments.append((judge_input, rubric))

        return judge_result

    def fake_criterion_judge(judge_input: JudgeInput, rubric: Rubric) -> CriterionJudgeResult:
        received_criterion_judge_arguments.append((judge_input, rubric))

        return criterion_judge_result

    result = evaluate_grader_comparison_case(case, fake_judge, fake_criterion_judge)

    assert received_judge_arguments == [(case["judge_input"], case["rubric"])]
    assert received_criterion_judge_arguments == [(case["judge_input"], case["rubric"])]

    assert result["expected_passed"] is True

    assert result["deterministic_passed"] is False
    assert result["deterministic_matched"] is False

    assert result["llm_passed"] is True
    assert result["llm_matched"] is True
    assert result["llm_score"] == 5
    assert result["llm_reason"] == "Correct paraphrase"
    assert result["llm_failed_criteria"] == []

    assert result["criterion_llm_passed"] is True
    assert result["criterion_llm_matched"] is True
    assert result["criterion_llm_failed_criteria"] == []


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
    criterion_judge_result = CriterionJudgeResult(
        criteria=[
            CriterionResult(
                name="correctness",
                passed=False,
                reason="The answer contradicts the context.",
            ),
        ]
    )

    def fake_judge(_judge_input: JudgeInput, _rubric: Rubric) -> JudgeResult:
        return judge_result

    def fake_criterion_judge(judge_input: JudgeInput, rubric: Rubric) -> CriterionJudgeResult:
        return criterion_judge_result

    result = evaluate_grader_comparison_case(case, fake_judge, fake_criterion_judge)

    assert result["expected_passed"] is False

    assert result["deterministic_passed"] is True
    assert result["deterministic_matched"] is False

    assert result["llm_passed"] is False
    assert result["llm_matched"] is True
    assert result["llm_score"] == 1
    assert result["llm_reason"] == "The answer contradicts the context."
    assert result["llm_failed_criteria"] == ["correctness"]

    assert result["criterion_llm_passed"] is False
    assert result["criterion_llm_matched"] is True
    assert result["criterion_llm_failed_criteria"] == ["correctness"]


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
    criterion_judge_results = [
        CriterionJudgeResult(
            criteria=[
                CriterionResult(
                    name="correctness",
                    passed=True,
                    reason="The answer is correct.",
                ),
            ]
        ),
        CriterionJudgeResult(
            criteria=[
                CriterionResult(
                    name="correctness",
                    passed=False,
                    reason="The answer is incorrect.",
                ),
            ]
        ),
    ]

    received_arguments: list[tuple[JudgeInput, Rubric]] = []
    received_criterion_arguments: list[tuple[JudgeInput, Rubric]] = []

    def fake_judge(judge_input: JudgeInput, rubric: Rubric) -> JudgeResult:
        received_arguments.append((judge_input, rubric))

        return judge_results[len(received_arguments) - 1]

    def criterion_fake_judge(judge_input: JudgeInput, rubric: Rubric) -> CriterionJudgeResult:
        received_criterion_arguments.append((judge_input, rubric))

        return criterion_judge_results[len(received_criterion_arguments) - 1]

    results = run_grader_comparison_evals(cases, fake_judge, criterion_fake_judge)

    assert len(results) == 2
    assert received_arguments == [
        (cases[0]["judge_input"], cases[0]["rubric"]),
        (cases[1]["judge_input"], cases[1]["rubric"]),
    ]
    assert received_criterion_arguments == [
        (cases[0]["judge_input"], cases[0]["rubric"]),
        (cases[1]["judge_input"], cases[1]["rubric"]),
    ]

    assert results[0]["deterministic_passed"] is True
    assert results[0]["deterministic_matched"] is True
    assert results[0]["llm_passed"] is True
    assert results[0]["llm_matched"] is True
    assert results[0]["criterion_llm_passed"] is True
    assert results[0]["criterion_llm_matched"] is True
    assert results[0]["criterion_llm_failed_criteria"] == []

    assert results[1]["deterministic_passed"] is False
    assert results[1]["deterministic_matched"] is True
    assert results[1]["llm_passed"] is False
    assert results[1]["llm_matched"] is True
    assert results[1]["criterion_llm_passed"] is False
    assert results[1]["criterion_llm_matched"] is True
    assert results[1]["criterion_llm_failed_criteria"] == ["correctness"]


@pytest.mark.parametrize(
    ("input_file", "expected_count"),
    [
        (GRADER_COMPARISON_CASES_FILE, 11),
        (GRADER_COMPARISON_HOLDOUT_CASES_FILE, 12),
    ],
)
def test_load_grader_comparison_datasets(input_file: str, expected_count: int) -> None:
    loaded_cases = load_grader_comparison_cases(input_file)

    assert len(loaded_cases) == expected_count
