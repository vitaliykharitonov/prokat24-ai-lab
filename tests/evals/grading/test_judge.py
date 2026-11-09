import pytest
from pydantic import ValidationError

from prokat24_ai_lab.evals.grading.judge import (
    CriterionJudgeResult,
    CriterionResult,
    JudgeInput,
    JudgeResult,
    Rubric,
    build_criterion_judge_prompt,
    build_judge_prompt,
    evaluate_with_criterion_judge,
    evaluate_with_judge,
    summarize_criterion_judge_result,
    validate_criterion_judge_result,
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


def test_judge_result_rejects_passed_with_failed_criteria() -> None:
    with pytest.raises(ValidationError):
        JudgeResult(
            score=4,
            passed=True,
            reason="The answer fails relevance.",
            failed_criteria=["relevance"],
        )


def test_judge_result_accepts_failed_with_failed_criteria() -> None:
    result = JudgeResult(
        score=4,
        passed=False,
        reason="The answer contains irrelevant information.",
        failed_criteria=["relevance"],
    )
    assert result.passed is False
    assert result.failed_criteria == ["relevance"]


def test_criterion_result_accepts_valid_parameters() -> None:
    criterion_result = CriterionResult(
        name="relevance",
        passed=False,
        reason="The answer contains information not needed to answer the question.",
    )

    assert criterion_result.name == "relevance"
    assert criterion_result.passed is False
    assert (
        criterion_result.reason
        == "The answer contains information not needed to answer the question."
    )


def test_criterion_judge_result_accepts_valid_parameters() -> None:
    criterion_judge_result = CriterionJudgeResult(
        criteria=[
            CriterionResult(
                name="correctness",
                passed=True,
                reason="The answer contains information not needed to answer the question.",
            ),
            CriterionResult(
                name="relevance",
                passed=False,
                reason="The answer contains information not needed to answer the question.",
            ),
            CriterionResult(
                name="groundedness",
                passed=True,
                reason="The answer contains information not needed to answer the question.",
            ),
        ]
    )

    assert len(criterion_judge_result.criteria) == 3
    assert criterion_judge_result.criteria[1].name == "relevance"
    assert criterion_judge_result.criteria[1].passed is False


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


def test_evaluate_with_criterion_judge() -> None:
    judge_input: JudgeInput = {
        "question": "Скільки коштує доставка по Києву?",
        "context": (
            "Доставка по Києву коштує від 200 грн. Пункт прокату працює щодня з 10:00 до 19:00."
        ),
        "answer": (
            "Доставка по Києву коштує від 200 грн. Пункт прокату працює щодня з 10:00 до 19:00."
        ),
    }

    rubric: Rubric = {
        "name": "delivery_answer",
        "criteria": [
            {
                "name": "correctness",
                "description": "The answer must correctly state the delivery price in Kyiv.",
            },
            {
                "name": "relevance",
                "description": "The answer must contain only information "
                "needed to answer the question.",
            },
            {
                "name": "groundedness",
                "description": "The answer must use only information from the provided context.",
            },
        ],
    }

    criterion_judge_result = CriterionJudgeResult(
        criteria=[
            CriterionResult(
                name="correctness",
                passed=True,
                reason="The delivery price is correct.",
            ),
            CriterionResult(
                name="relevance",
                passed=False,
                reason="The opening hours are not needed to answer the question.",
            ),
            CriterionResult(
                name="groundedness",
                passed=True,
                reason="All information is supported by the provided context.",
            ),
        ]
    )
    received_arguments: list[tuple[JudgeInput, Rubric]] = []

    def fake_judge(input_data: JudgeInput, received_rubric: Rubric) -> CriterionJudgeResult:
        received_arguments.append((input_data, received_rubric))
        return criterion_judge_result

    result = evaluate_with_criterion_judge(judge_input, rubric, fake_judge)

    assert received_arguments == [(judge_input, rubric)]
    assert result == {
        "passed": False,
        "failed_criteria": ["relevance"],
    }


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
    assert "passed must be false" in prompt
    assert "Groundedness and relevance are separate criteria" in prompt
    assert "Related, useful, grounded, or potentially helpful information" in prompt


def test_build_criterion_judge_prompt() -> None:
    judge_input: JudgeInput = {
        "question": "Скільки коштує доставка по Києву?",
        "context": (
            "Доставка по Києву коштує від 200 грн. Пункт прокату працює щодня з 10:00 до 19:00."
        ),
        "answer": (
            "Доставка по Києву коштує від 200 грн. Пункт прокату працює щодня з 10:00 до 19:00."
        ),
    }

    rubric: Rubric = {
        "name": "delivery_answer",
        "criteria": [
            {
                "name": "correctness",
                "description": "The answer must correctly state the delivery price in Kyiv.",
            },
            {
                "name": "relevance",
                "description": "The answer must contain only information "
                "needed to answer the question.",
            },
            {
                "name": "groundedness",
                "description": "The answer must use only information from the provided context.",
            },
        ],
    }

    prompt = build_criterion_judge_prompt(judge_input, rubric)

    assert "delivery_answer" in prompt
    assert "correctness" in prompt
    assert "relevance" in prompt
    assert "groundedness" in prompt
    assert "Rubric name:" in prompt
    assert "Rubric criteria:" in prompt

    assert "Скільки коштує доставка по Києву?" in prompt
    assert "Доставка по Києву коштує від 200 грн." in prompt
    assert "Пункт прокату працює щодня з 10:00 до 19:00." in prompt

    assert "Return one result for every rubric criterion." in prompt
    assert "- name: exactly the criterion name from the 'Rubric criteria' list above" in prompt
    assert "- passed: true or false" in prompt
    assert "- reason: concise explanation" in prompt
    assert "Evaluate every criterion independently." in prompt
    assert "Do not omit, rename, or add criteria." in prompt
    assert "Do not use the rubric name as a criterion name." in prompt


def test_summarize_criterion_judge_result_with_failed_criterion() -> None:
    criterion_judge_result = CriterionJudgeResult(
        criteria=[
            CriterionResult(
                name="correctness",
                passed=True,
                reason="The delivery price is correct.",
            ),
            CriterionResult(
                name="relevance",
                passed=False,
                reason="The answer contains information not needed to answer the question.",
            ),
            CriterionResult(
                name="groundedness",
                passed=True,
                reason="All information is supported by the context.",
            ),
        ]
    )

    summary = summarize_criterion_judge_result(criterion_judge_result)

    assert summary["passed"] is False
    assert summary["failed_criteria"] == ["relevance"]


def test_summarize_criterion_judge_result_with_passed_criterion() -> None:
    criterion_judge_result = CriterionJudgeResult(
        criteria=[
            CriterionResult(
                name="correctness",
                passed=True,
                reason="The delivery price is correct.",
            ),
            CriterionResult(
                name="relevance",
                passed=True,
                reason="The answer is correct.",
            ),
            CriterionResult(
                name="groundedness",
                passed=True,
                reason="All information is supported by the context.",
            ),
        ]
    )

    summary = summarize_criterion_judge_result(criterion_judge_result)

    assert summary["passed"] is True
    assert summary["failed_criteria"] == []


def test_validate_criterion_judge_result_raises_when_criterion_is_missing() -> None:
    rubric: Rubric = {
        "name": "answer_quality",
        "criteria": [
            {
                "name": "correctness",
                "description": "The answer must be correct.",
            },
            {
                "name": "relevance",
                "description": "The answer must be relevant.",
            },
            {
                "name": "groundedness",
                "description": "The answer must use only the provided context.",
            },
        ],
    }

    result = CriterionJudgeResult(
        criteria=[
            CriterionResult(
                name="correctness",
                passed=True,
                reason="The answer is correct.",
            ),
            CriterionResult(
                name="groundedness",
                passed=True,
                reason="The answer is supported by the context.",
            ),
        ]
    )

    with pytest.raises(ValueError):
        validate_criterion_judge_result(result, rubric)


def test_validate_criterion_judge_result_accepts_matching_criteria() -> None:
    rubric: Rubric = {
        "name": "answer_quality",
        "criteria": [
            {
                "name": "correctness",
                "description": "The answer must be correct.",
            },
            {
                "name": "relevance",
                "description": "The answer must be relevant.",
            },
            {
                "name": "groundedness",
                "description": "The answer must use only the provided context.",
            },
        ],
    }
    result = CriterionJudgeResult(
        criteria=[
            CriterionResult(
                name="correctness",
                passed=True,
                reason="The answer is correct.",
            ),
            CriterionResult(
                name="relevance",
                passed=False,
                reason="The answer contains unnecessary information.",
            ),
            CriterionResult(
                name="groundedness",
                passed=True,
                reason="The answer is supported by the context.",
            ),
        ]
    )

    validate_criterion_judge_result(result, rubric)
