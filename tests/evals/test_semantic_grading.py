import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from prokat24_ai_lab.evals.semantic_grading import (
    CriterionJudgeResult,
    CriterionResult,
    GraderClassificationMetrics,
    GraderComparisonCase,
    GraderComparisonResult,
    GraderComparisonSummary,
    GraderConfusionMatrix,
    GraderKey,
    GraderStabilityResult,
    JudgeEvalCase,
    JudgeInput,
    JudgeResult,
    Rubric,
    build_criterion_judge_prompt,
    build_grader_comparison_report,
    build_grader_confusion_matrix,
    build_judge_prompt,
    calculate_grader_classification_metrics,
    evaluate_grader_comparison_case,
    evaluate_judge_case,
    evaluate_with_criterion_judge,
    evaluate_with_judge,
    load_grader_comparison_cases,
    load_judge_eval_cases,
    run_grader_comparison_evals,
    run_judge_evals,
    save_grader_comparison_report,
    summarize_criterion_judge_result,
    summarize_grader_comparison,
    summarize_grader_comparison_by_category,
    summarize_grader_stability,
    validate_criterion_judge_result,
)
from prokat24_ai_lab.project_paths import (
    GRADER_COMPARISON_CASES_FILE,
    GRADER_COMPARISON_HOLDOUT_CASES_FILE,
)

GRADER_COMPARISON_TEST_RESULTS: list[GraderComparisonResult] = [
    {
        "id": "case_1",
        "category": "correct",
        "expected_passed": True,
        "deterministic_passed": True,
        "deterministic_matched": True,
        "llm_passed": True,
        "llm_matched": True,
        "llm_score": 5,
        "llm_reason": "Correct.",
        "llm_failed_criteria": [],
        "criterion_llm_passed": True,
        "criterion_llm_matched": True,
        "criterion_llm_failed_criteria": [],
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
        "llm_reason": "Correct paraphrase.",
        "llm_failed_criteria": [],
        "criterion_llm_passed": True,
        "criterion_llm_matched": True,
        "criterion_llm_failed_criteria": [],
    },
    {
        "id": "case_3",
        "category": "negation",
        "expected_passed": False,
        "deterministic_passed": True,
        "deterministic_matched": False,
        "llm_passed": False,
        "llm_matched": True,
        "llm_score": 1,
        "llm_reason": "The answer contradicts the context.",
        "llm_failed_criteria": ["correctness"],
        "criterion_llm_passed": False,
        "criterion_llm_matched": True,
        "criterion_llm_failed_criteria": ["correctness"],
    },
    {
        "id": "case_4",
        "category": "incorrect",
        "expected_passed": False,
        "deterministic_passed": False,
        "deterministic_matched": True,
        "llm_passed": False,
        "llm_matched": True,
        "llm_score": 1,
        "llm_reason": "The answer is incorrect.",
        "llm_failed_criteria": ["correctness"],
        "criterion_llm_passed": False,
        "criterion_llm_matched": True,
        "criterion_llm_failed_criteria": ["correctness"],
    },
]


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
            "criterion_llm_passed": True,
            "criterion_llm_matched": True,
            "criterion_llm_failed_criteria": [],
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
            "criterion_llm_passed": True,
            "criterion_llm_matched": True,
            "criterion_llm_failed_criteria": [],
        },
        {
            "id": "case_3",
            "category": "negation",
            "expected_passed": False,
            "deterministic_passed": True,
            "deterministic_matched": False,
            "llm_passed": False,
            "llm_matched": True,
            "llm_score": 1,
            "llm_reason": "Test result",
            "llm_failed_criteria": ["correctness"],
            "criterion_llm_passed": False,
            "criterion_llm_matched": True,
            "criterion_llm_failed_criteria": ["correctness"],
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
            "criterion_llm_passed": False,
            "criterion_llm_matched": True,
            "criterion_llm_failed_criteria": ["correctness"],
        },
    ]

    expected_summary: GraderComparisonSummary = {
        "total_cases": 4,
        "deterministic_matched": 2,
        "llm_matched": 3,
        "criterion_llm_matched": 4,
        "deterministic_accuracy": 0.5,
        "llm_accuracy": 0.75,
        "criterion_llm_accuracy": 1.0,
    }

    summary = summarize_grader_comparison(results)

    assert summary == expected_summary


def test_summarize_grader_comparison_empty_results() -> None:
    results: list[GraderComparisonResult] = []

    expected_summary: GraderComparisonSummary = {
        "total_cases": 0,
        "deterministic_matched": 0,
        "llm_matched": 0,
        "criterion_llm_matched": 0,
        "deterministic_accuracy": 0.0,
        "llm_accuracy": 0.0,
        "criterion_llm_accuracy": 0.0,
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
            "criterion_llm_passed": True,
            "criterion_llm_matched": True,
            "criterion_llm_failed_criteria": [],
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
            "criterion_llm_passed": True,
            "criterion_llm_matched": True,
            "criterion_llm_failed_criteria": [],
        },
        {
            "id": "case_3",
            "category": "paraphrase",
            "expected_passed": True,
            "deterministic_passed": True,
            "deterministic_matched": True,
            "llm_passed": False,
            "llm_matched": False,
            "llm_score": 2,
            "llm_reason": "Test result",
            "llm_failed_criteria": ["correctness"],
            "criterion_llm_passed": True,
            "criterion_llm_matched": True,
            "criterion_llm_failed_criteria": [],
        },
        {
            "id": "case_4",
            "category": "negation",
            "expected_passed": False,
            "deterministic_passed": True,
            "deterministic_matched": False,
            "llm_passed": False,
            "llm_matched": True,
            "llm_score": 1,
            "llm_reason": "Test result",
            "llm_failed_criteria": ["correctness"],
            "criterion_llm_passed": False,
            "criterion_llm_matched": True,
            "criterion_llm_failed_criteria": ["correctness"],
        },
    ]

    expected = {
        "correct": {
            "total_cases": 1,
            "deterministic_matched": 1,
            "llm_matched": 1,
            "criterion_llm_matched": 1,
            "deterministic_accuracy": 1.0,
            "llm_accuracy": 1.0,
            "criterion_llm_accuracy": 1.0,
        },
        "paraphrase": {
            "total_cases": 2,
            "deterministic_matched": 1,
            "llm_matched": 1,
            "criterion_llm_matched": 2,
            "deterministic_accuracy": 0.5,
            "llm_accuracy": 0.5,
            "criterion_llm_accuracy": 1.0,
        },
        "negation": {
            "total_cases": 1,
            "deterministic_matched": 0,
            "llm_matched": 1,
            "criterion_llm_matched": 1,
            "deterministic_accuracy": 0.0,
            "llm_accuracy": 1.0,
            "criterion_llm_accuracy": 1.0,
        },
    }

    summaries = summarize_grader_comparison_by_category(results)

    assert summaries == expected


def test_summarize_grader_comparison_by_category_empty_results() -> None:
    results: list[GraderComparisonResult] = []

    summaries = summarize_grader_comparison_by_category(results)

    assert summaries == {}


@pytest.mark.parametrize(
    ("grader_key", "expected_matrix"),
    [
        (
            "deterministic_passed",
            {
                "true_positive": 1,
                "true_negative": 1,
                "false_positive": 1,
                "false_negative": 1,
            },
        ),
        (
            "llm_passed",
            {
                "true_positive": 2,
                "true_negative": 2,
                "false_positive": 0,
                "false_negative": 0,
            },
        ),
        (
            "criterion_llm_passed",
            {
                "true_positive": 2,
                "true_negative": 2,
                "false_positive": 0,
                "false_negative": 0,
            },
        ),
    ],
)
def test_build_grader_confusion_matrix_for_graders(
    grader_key: GraderKey, expected_matrix: GraderConfusionMatrix
) -> None:
    matrix = build_grader_confusion_matrix(
        GRADER_COMPARISON_TEST_RESULTS,
        grader_key,
    )

    assert matrix == expected_matrix


def test_calculate_grader_classification_metrics() -> None:
    matrix: GraderConfusionMatrix = {
        "true_positive": 1,
        "true_negative": 1,
        "false_positive": 4,
        "false_negative": 2,
    }
    expected: GraderClassificationMetrics = {
        "precision": 0.2,
        "recall": 1 / 3,
    }
    metrics = calculate_grader_classification_metrics(matrix)

    assert metrics == expected


def test_calculate_grader_classification_metrics_with_no_positive_cases() -> None:
    matrix: GraderConfusionMatrix = {
        "true_positive": 0,
        "true_negative": 5,
        "false_positive": 0,
        "false_negative": 0,
    }
    expected: GraderClassificationMetrics = {
        "precision": 0.0,
        "recall": 0.0,
    }

    metrics = calculate_grader_classification_metrics(matrix)

    assert metrics == expected


def test_summarize_grader_stability() -> None:
    runs: list[list[GraderComparisonResult]] = [
        [
            {
                "id": "stable_case",
                "category": "correct",
                "expected_passed": True,
                "deterministic_passed": True,
                "deterministic_matched": True,
                "llm_passed": True,
                "llm_matched": True,
                "llm_score": 5,
                "llm_reason": "Correct.",
                "llm_failed_criteria": [],
                "criterion_llm_passed": True,
                "criterion_llm_matched": True,
                "criterion_llm_failed_criteria": [],
            },
            {
                "id": "unstable_case",
                "category": "irrelevant_extra",
                "expected_passed": False,
                "deterministic_passed": True,
                "deterministic_matched": False,
                "llm_passed": True,
                "llm_matched": False,
                "llm_score": 5,
                "llm_reason": "Accepted.",
                "llm_failed_criteria": [],
                "criterion_llm_passed": False,
                "criterion_llm_matched": True,
                "criterion_llm_failed_criteria": ["relevance"],
            },
        ],
        [
            {
                "id": "stable_case",
                "category": "correct",
                "expected_passed": True,
                "deterministic_passed": True,
                "deterministic_matched": True,
                "llm_passed": True,
                "llm_matched": True,
                "llm_score": 5,
                "llm_reason": "Correct.",
                "llm_failed_criteria": [],
                "criterion_llm_passed": True,
                "criterion_llm_matched": True,
                "criterion_llm_failed_criteria": [],
            },
            {
                "id": "unstable_case",
                "category": "irrelevant_extra",
                "expected_passed": False,
                "deterministic_passed": True,
                "deterministic_matched": False,
                "llm_passed": False,
                "llm_matched": True,
                "llm_score": 2,
                "llm_reason": "Contains irrelevant information.",
                "llm_failed_criteria": ["relevance"],
                "criterion_llm_passed": False,
                "criterion_llm_matched": True,
                "criterion_llm_failed_criteria": ["relevance"],
            },
        ],
        [
            {
                "id": "stable_case",
                "category": "correct",
                "expected_passed": True,
                "deterministic_passed": True,
                "deterministic_matched": True,
                "llm_passed": True,
                "llm_matched": True,
                "llm_score": 5,
                "llm_reason": "Correct.",
                "llm_failed_criteria": [],
                "criterion_llm_passed": True,
                "criterion_llm_matched": True,
                "criterion_llm_failed_criteria": [],
            },
            {
                "id": "unstable_case",
                "category": "irrelevant_extra",
                "expected_passed": False,
                "deterministic_passed": True,
                "deterministic_matched": False,
                "llm_passed": False,
                "llm_matched": True,
                "llm_score": 2,
                "llm_reason": "Contains irrelevant information.",
                "llm_failed_criteria": ["relevance"],
                "criterion_llm_passed": False,
                "criterion_llm_matched": True,
                "criterion_llm_failed_criteria": ["relevance"],
            },
        ],
    ]

    expected: list[GraderStabilityResult] = [
        {
            "id": "stable_case",
            "expected_passed": True,
            "llm_passed_values": [True, True, True],
            "llm_stable": True,
            "llm_match_rate": 1.0,
            "criterion_llm_passed_values": [True, True, True],
            "criterion_llm_stable": True,
            "criterion_llm_match_rate": 1.0,
        },
        {
            "id": "unstable_case",
            "expected_passed": False,
            "llm_passed_values": [True, False, False],
            "llm_stable": False,
            "llm_match_rate": 2 / 3,
            "criterion_llm_passed_values": [False, False, False],
            "criterion_llm_stable": True,
            "criterion_llm_match_rate": 1.0,
        },
    ]

    stability_results = summarize_grader_stability(runs)

    assert stability_results == expected


def test_summarize_grader_stability_empty_runs() -> None:
    stability_results = summarize_grader_stability([])

    assert stability_results == []


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


def test_build_grader_comparison_report() -> None:
    all_runs = [
        GRADER_COMPARISON_TEST_RESULTS,
        GRADER_COMPARISON_TEST_RESULTS,
    ]

    report = build_grader_comparison_report(
        all_runs=all_runs,
        split="holdout",
        run_id="test_run",
        created_at="2026-10-04T16:00:00Z",
        git_commit="abc123",
        git_dirty=False,
    )

    assert report["report_type"] == "grader_comparison_report"
    assert report["schema_version"] == 1
    assert report["created_at"] == "2026-10-04T16:00:00Z"
    assert report["git_commit"] == "abc123"
    assert report["git_dirty"] is False
    assert report["run_id"] == "test_run"
    assert report["split"] == "holdout"
    assert report["run_count"] == 2
    assert report["runs"] == all_runs


def test_save_grader_comparison_report_saves_historical_and_latest_files(
    tmp_path: Path,
) -> None:
    report = build_grader_comparison_report(
        all_runs=[GRADER_COMPARISON_TEST_RESULTS],
        split="holdout",
        run_id="test_run",
        created_at="2026-10-04T16:00:00Z",
        git_commit="abc123",
        git_dirty=False,
    )

    historical_file, latest_file = save_grader_comparison_report(
        report=report,
        output_dir=tmp_path,
    )

    expected_historical_file = tmp_path / "holdout" / "runs" / "test_run.json"
    expected_latest_file = tmp_path / "holdout" / "latest.json"

    assert historical_file == expected_historical_file
    assert latest_file == expected_latest_file

    assert historical_file.exists()
    assert latest_file.exists()

    with open(historical_file, encoding="utf-8") as file:
        historical_report = json.load(file)

    with open(latest_file, encoding="utf-8") as file:
        latest_report = json.load(file)

    assert historical_report == report
    assert latest_report == report
