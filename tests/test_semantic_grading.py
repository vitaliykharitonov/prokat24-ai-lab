import pytest
from pydantic import ValidationError

from prokat24_ai_lab.evals.semantic_grading import (
    JudgeEvalCase,
    JudgeInput,
    JudgeResult,
    Rubric,
    build_judge_prompt,
    evaluate_judge_case,
    evaluate_with_judge,
    run_judge_evals,
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
