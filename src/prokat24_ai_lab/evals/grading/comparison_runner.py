import json
from pathlib import Path
from typing import TypedDict, cast

from pydantic import BaseModel

from prokat24_ai_lab.evals.answers.rag import RagAnswerEvalCase, evaluate_rag_answer_case
from prokat24_ai_lab.evals.grading.comparison import GraderComparisonResult
from prokat24_ai_lab.evals.grading.judge import (
    CriterionJudge,
    Judge,
    JudgeInput,
    Rubric,
    evaluate_with_criterion_judge,
    evaluate_with_judge,
)


class DeterministicExpectations(TypedDict):
    required_fragments: list[str]
    forbidden_fragments: list[str]


class GraderComparisonCase(TypedDict):
    id: str
    category: str
    judge_input: JudgeInput
    expected_passed: bool
    deterministic: DeterministicExpectations
    rubric: Rubric


class GraderComparisonCaseModel(BaseModel):
    id: str
    category: str
    judge_input: JudgeInput
    expected_passed: bool
    deterministic: DeterministicExpectations
    rubric: Rubric


def load_grader_comparison_cases(input_file: Path) -> list[GraderComparisonCase]:
    with open(input_file, encoding="utf-8") as file:
        loaded_data = json.load(file)
        validated_cases: list[GraderComparisonCase] = []

        for case in loaded_data:
            validated_case = GraderComparisonCaseModel.model_validate(case)
            validated_cases.append(cast(GraderComparisonCase, validated_case.model_dump()))

    return validated_cases


def evaluate_grader_comparison_case(
    case: GraderComparisonCase,
    judge: Judge,
    criterion_judge: CriterionJudge,
) -> GraderComparisonResult:
    rag_answer_eval_case: RagAnswerEvalCase = {
        "question": case["judge_input"]["question"],
        "expected_answer_contains": case["deterministic"]["required_fragments"],
        "expected_answer_not_contains": case["deterministic"]["forbidden_fragments"],
    }

    def answerer(_question: str) -> str:
        return case["judge_input"]["answer"]

    deterministic_result = evaluate_rag_answer_case(rag_answer_eval_case, answerer)

    judge_result = evaluate_with_judge(case["judge_input"], case["rubric"], judge)

    criterion_judge_result = evaluate_with_criterion_judge(
        case["judge_input"],
        case["rubric"],
        criterion_judge,
    )

    deterministic_matched = deterministic_result["passed"] == case["expected_passed"]
    llm_matched = judge_result.passed == case["expected_passed"]
    criterion_llm_matched = criterion_judge_result["passed"] == case["expected_passed"]

    return {
        "id": case["id"],
        "category": case["category"],
        "expected_passed": case["expected_passed"],
        "deterministic_passed": deterministic_result["passed"],
        "deterministic_matched": deterministic_matched,
        "llm_passed": judge_result.passed,
        "llm_matched": llm_matched,
        "llm_score": judge_result.score,
        "llm_reason": judge_result.reason,
        "llm_failed_criteria": judge_result.failed_criteria,
        "criterion_llm_passed": criterion_judge_result["passed"],
        "criterion_llm_matched": criterion_llm_matched,
        "criterion_llm_failed_criteria": criterion_judge_result["failed_criteria"],
    }


def run_grader_comparison_evals(
    cases: list[GraderComparisonCase],
    judge: Judge,
    criterion_judge: CriterionJudge,
) -> list[GraderComparisonResult]:
    results: list[GraderComparisonResult] = []

    for case in cases:
        results.append(evaluate_grader_comparison_case(case, judge, criterion_judge))

    return results
