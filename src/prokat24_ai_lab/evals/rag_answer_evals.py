import json
from collections.abc import Callable
from pathlib import Path
from typing import TypedDict

from prokat24_ai_lab.evals.report_metadata import build_eval_report_metadata


class RagAnswerEvalCase(TypedDict):
    question: str
    expected_answer_contains: list[str]


class RagAnswerEvalResult(TypedDict):
    question: str
    expected_answer_contains: list[str]
    missing_answer_fragments: list[str]
    passed: bool
    answer: str


class RagAnswerEvalReport(TypedDict):
    report_type: str
    schema_version: int
    created_at: str
    git_commit: str | None
    git_dirty: bool | None
    total_cases: int
    passed_cases: int
    accuracy: float
    results: list[RagAnswerEvalResult]


RagAnswerer = Callable[[str], str]


def evaluate_rag_answer_case(
    case: RagAnswerEvalCase,
    answerer: RagAnswerer,
) -> RagAnswerEvalResult:
    question = case["question"]

    answer = answerer(question)

    expected_answer_contains = case["expected_answer_contains"]
    missing_answer_fragments = []

    for fragment in expected_answer_contains:
        if fragment.lower() not in answer.lower():
            missing_answer_fragments.append(fragment)

    passed = not missing_answer_fragments

    return {
        "question": case["question"],
        "expected_answer_contains": expected_answer_contains,
        "missing_answer_fragments": missing_answer_fragments,
        "passed": passed,
        "answer": answer,
    }


def run_rag_answer_evals(
    cases: list[RagAnswerEvalCase],
    answerer: RagAnswerer,
) -> list[RagAnswerEvalResult]:
    results: list[RagAnswerEvalResult] = []
    for case in cases:
        results.append(evaluate_rag_answer_case(case, answerer))

    return results


def load_rag_answer_eval_cases(input_file: Path) -> list[RagAnswerEvalCase]:
    with open(input_file, encoding="utf-8") as file:
        loaded_data: list[RagAnswerEvalCase] = json.load(file)

    return loaded_data


def count_passed_rag_answer_results(results: list[RagAnswerEvalResult]) -> int:
    count: int = 0
    for result in results:
        if result["passed"]:
            count += 1

    return count


def calculate_rag_answer_accuracy(results: list[RagAnswerEvalResult]) -> float:
    if not results:
        return 0.0

    total_cases = len(results)
    passed_cases = count_passed_rag_answer_results(results)

    return passed_cases / total_cases


def build_rag_answer_eval_report(
    results: list[RagAnswerEvalResult],
    created_at: str,
    git_commit: str | None = None,
    git_dirty: bool | None = None,
) -> RagAnswerEvalReport:
    metadata = build_eval_report_metadata(
        report_type="rag_answer_eval_report",
        created_at=created_at,
        git_commit=git_commit,
        git_dirty=git_dirty,
    )

    total_cases = len(results)
    passed_cases = count_passed_rag_answer_results(results)
    accuracy = calculate_rag_answer_accuracy(results)

    return {
        **metadata,
        "total_cases": total_cases,
        "passed_cases": passed_cases,
        "accuracy": accuracy,
        "results": results,
    }


def save_rag_answer_eval_report(
    report: RagAnswerEvalReport,
    output_file: Path,
) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
