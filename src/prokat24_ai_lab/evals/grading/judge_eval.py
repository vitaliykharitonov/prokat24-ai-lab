import json
from pathlib import Path
from typing import TypedDict, cast

from pydantic import BaseModel

from prokat24_ai_lab.evals.grading.judge import Judge, JudgeInput, Rubric


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
