import json
from collections.abc import Callable
from pathlib import Path
from typing import Literal, TypedDict, cast

from pydantic import BaseModel, Field

from prokat24_ai_lab.evals.rag_answer_evals import RagAnswerEvalCase, evaluate_rag_answer_case


class RubricCriterion(TypedDict):
    name: str
    description: str


class Rubric(TypedDict):
    name: str
    criteria: list[RubricCriterion]


class JudgeInput(TypedDict):
    question: str
    context: str
    answer: str


class JudgeResult(BaseModel):
    score: int = Field(ge=1, le=5)
    passed: bool
    reason: str
    failed_criteria: list[str]


class JudgeEvalCase(TypedDict):
    judge_input: JudgeInput
    rubric: Rubric
    expected_passed: bool


class JudgeEvalCaseModel(BaseModel):
    judge_input: JudgeInput
    rubric: Rubric
    expected_passed: bool


class JudgeEvalResult(TypedDict):
    expected_passed: bool
    actual_passed: bool
    matched: bool
    score: int
    reason: str
    failed_criteria: list[str]


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


class GraderComparisonResult(TypedDict):
    id: str
    category: str
    expected_passed: bool
    deterministic_passed: bool
    deterministic_matched: bool
    llm_passed: bool
    llm_matched: bool
    llm_score: int
    llm_reason: str
    llm_failed_criteria: list[str]


class GraderComparisonSummary(TypedDict):
    total_cases: int
    deterministic_matched: int
    llm_matched: int
    deterministic_accuracy: float
    llm_accuracy: float


class GraderCategorySummary(TypedDict):
    total_cases: int
    deterministic_matched: int
    llm_matched: int
    deterministic_accuracy: float
    llm_accuracy: float


class GraderConfusionMatrix(TypedDict):
    true_positive: int
    true_negative: int
    false_positive: int
    false_negative: int


class GraderClassificationMetrics(TypedDict):
    precision: float
    recall: float


class GraderStabilityResult(TypedDict):
    id: str
    expected_passed: bool
    llm_passed_values: list[bool]
    stable: bool


Judge = Callable[[JudgeInput, Rubric], JudgeResult]

JUDGE_INSTRUCTIONS: str = (
    "Judge instructions:\n"
    "You are evaluating an assistant answer.\n"
    "Evaluate it strictly using the provided rubric.\n"
    "Use only the provided context.\n"
    "Do not use outside knowledge."
)

RETURN_INSTRUCTIONS: str = (
    "Return:\n"
    "- score: integer from 1 to 5\n"
    "- passed: true or false\n"
    "- reason: concise explanation\n"
    "- failed_criteria: names of failed rubric criteria"
)

GraderKey = Literal["deterministic_passed", "llm_passed"]


def evaluate_with_judge(
    judge_input: JudgeInput,
    rubric: Rubric,
    judge: Judge,
) -> JudgeResult:
    return judge(judge_input, rubric)


def build_judge_prompt(
    judge_input: JudgeInput,
    rubric: Rubric,
) -> str:
    lines: list[str] = []
    lines.append(JUDGE_INSTRUCTIONS)
    lines.append("")
    lines.append("Rubric:")
    lines.append(rubric["name"])

    for criteria in rubric["criteria"]:
        lines.append(f"- {criteria['name']}: {criteria['description']}")

    lines.append("")
    lines.append("Question:")
    lines.append(judge_input["question"])
    lines.append("")
    lines.append("Context:")
    lines.append(judge_input["context"])
    lines.append("")
    lines.append("Assistant answer:")
    lines.append(judge_input["answer"])
    lines.append("")
    lines.append(RETURN_INSTRUCTIONS)

    return "\n".join(lines)


def evaluate_judge_case(case: JudgeEvalCase, judge: Judge) -> JudgeEvalResult:
    result = judge(case["judge_input"], case["rubric"])
    expected_passed: bool = case["expected_passed"]
    actual_passed: bool = result.passed
    matched = expected_passed == actual_passed

    return {
        "expected_passed": expected_passed,
        "actual_passed": actual_passed,
        "matched": matched,
        "score": result.score,
        "reason": result.reason,
        "failed_criteria": result.failed_criteria,
    }


def run_judge_evals(cases: list[JudgeEvalCase], judge: Judge) -> list[JudgeEvalResult]:
    results: list[JudgeEvalResult] = []

    for case in cases:
        results.append(evaluate_judge_case(case, judge))

    return results


def load_judge_eval_cases(input_file: Path) -> list[JudgeEvalCase]:
    with open(input_file, encoding="utf-8") as file:
        loaded_data = json.load(file)
        validated_cases: list[JudgeEvalCase] = []

        for case in loaded_data:
            validated_case = JudgeEvalCaseModel.model_validate(case)
            validated_cases.append(cast(JudgeEvalCase, validated_case.model_dump()))

    return validated_cases


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

    deterministic_matched = deterministic_result["passed"] == case["expected_passed"]
    llm_matched = judge_result.passed == case["expected_passed"]

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
    }


def run_grader_comparison_evals(
    cases: list[GraderComparisonCase],
    judge: Judge,
) -> list[GraderComparisonResult]:
    results: list[GraderComparisonResult] = []

    for case in cases:
        results.append(evaluate_grader_comparison_case(case, judge))

    return results


def summarize_grader_comparison(
    results: list[GraderComparisonResult],
) -> GraderComparisonSummary:
    total_cases = len(results)
    deterministic_matched = 0
    llm_matched = 0

    if not results:
        return {
            "total_cases": 0,
            "deterministic_matched": 0,
            "llm_matched": 0,
            "deterministic_accuracy": 0.0,
            "llm_accuracy": 0.0,
        }

    for result in results:
        if result["deterministic_matched"]:
            deterministic_matched += 1

        if result["llm_matched"]:
            llm_matched += 1

    deterministic_accuracy = deterministic_matched / total_cases
    llm_accuracy = llm_matched / total_cases

    return {
        "total_cases": total_cases,
        "deterministic_matched": deterministic_matched,
        "llm_matched": llm_matched,
        "deterministic_accuracy": deterministic_accuracy,
        "llm_accuracy": llm_accuracy,
    }


def summarize_grader_comparison_by_category(
    results: list[GraderComparisonResult],
) -> dict[str, GraderCategorySummary]:
    grouped_results: dict[str, list[GraderComparisonResult]] = {}
    summaries: dict[str, GraderCategorySummary] = {}

    for result in results:
        if result["category"] not in grouped_results:
            grouped_results[result["category"]] = []

        grouped_results[result["category"]].append(result)

    for group in grouped_results:
        summaries[group] = summarize_grader_comparison(grouped_results[group])

    return summaries


def build_grader_confusion_matrix(
    results: list[GraderComparisonResult],
    grader_key: GraderKey,
) -> GraderConfusionMatrix:
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    for result in results:
        expected = result["expected_passed"]
        actual = result[grader_key]

        if expected is True and actual is True:
            true_positive += 1
        elif expected is False and actual is False:
            true_negative += 1
        elif expected is False and actual is True:
            false_positive += 1
        else:
            false_negative += 1

    return {
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
    }


def calculate_grader_classification_metrics(
    matrix: GraderConfusionMatrix,
) -> GraderClassificationMetrics:
    true_positive = matrix["true_positive"]
    false_positive = matrix["false_positive"]
    false_negative = matrix["false_negative"]

    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative

    precision = true_positive / precision_denominator if precision_denominator else 0.0
    recall = true_positive / recall_denominator if recall_denominator else 0.0

    return {
        "precision": precision,
        "recall": recall,
    }


def summarize_grader_stability(
    runs: list[list[GraderComparisonResult]],
) -> list[GraderStabilityResult]:
    stability_results: list[GraderStabilityResult] = []

    if not runs:
        return stability_results

    for first_run_case in runs[0]:
        case_id = first_run_case["id"]
        llm_passed_values: list[bool] = []

        for run in runs:
            for case in run:
                if case["id"] == case_id:
                    llm_passed_values.append(case["llm_passed"])

        stable = len(set(llm_passed_values)) == 1

        stability_results.append(
            {
                "id": case_id,
                "expected_passed": first_run_case["expected_passed"],
                "llm_passed_values": llm_passed_values,
                "stable": stable,
            }
        )

    return stability_results
