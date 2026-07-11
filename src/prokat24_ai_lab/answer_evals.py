import json
from pathlib import Path
from typing import TypedDict

from prokat24_ai_lab.answer_engine import answer_question
from prokat24_ai_lab.eval_report_metadata import build_eval_report_metadata
from prokat24_ai_lab.knowledge_base import KnowledgePage


class AnswerEvalCase(TypedDict):
    question: str
    expected_answer_type: str
    expected_source_url: str | None
    expected_answer_contains: list[str]


class AnswerEvalResult(TypedDict):
    question: str
    expected_answer_type: str
    actual_answer_type: str
    expected_source_url: str | None
    actual_source_url: str | None
    expected_answer_contains: list[str]
    missing_answer_fragments: list[str]
    passed: bool
    answer: str


class AnswerEvalReport(TypedDict):
    report_type: str
    schema_version: int
    created_at: str
    git_commit: str | None
    git_dirty: bool | None
    total_cases: int
    passed_cases: int
    accuracy: float
    results: list[AnswerEvalResult]


def load_answer_eval_cases(input_file: Path) -> list[AnswerEvalCase]:
    with open(input_file, encoding="utf-8") as file:
        return json.load(file)


def evaluate_answer_case(case: AnswerEvalCase, pages: list[KnowledgePage]) -> AnswerEvalResult:
    question = case["question"]
    answer = answer_question(question, pages)
    missing_answer_fragments = []
    answer_text = answer["answer"].lower()

    actual_source_url = None if not answer["sources"] else answer["sources"][0]["url"]

    for fragment in case["expected_answer_contains"]:
        if fragment.lower() not in answer_text:
            missing_answer_fragments.append(fragment)

    passed = (
        answer["answer_type"] == case["expected_answer_type"]
        and actual_source_url == case["expected_source_url"]
        and not missing_answer_fragments
    )

    return {
        "question": question,
        "expected_answer_type": case["expected_answer_type"],
        "actual_answer_type": answer["answer_type"],
        "expected_source_url": case["expected_source_url"],
        "actual_source_url": actual_source_url,
        "expected_answer_contains": case["expected_answer_contains"],
        "missing_answer_fragments": missing_answer_fragments,
        "passed": passed,
        "answer": answer["answer"],
    }


def run_answer_evals(
    cases: list[AnswerEvalCase],
    pages: list[KnowledgePage],
) -> list[AnswerEvalResult]:
    results: list[AnswerEvalResult] = []

    for case in cases:
        results.append(evaluate_answer_case(case, pages))

    return results


def count_passed_answer_results(results: list[AnswerEvalResult]) -> int:
    count = 0

    for result in results:
        if result["passed"]:
            count += 1

    return count


def calculate_answer_accuracy(results: list[AnswerEvalResult]) -> float:
    accuracy = 0.0 if not results else count_passed_answer_results(results) / len(results)

    return accuracy


def build_answer_eval_report(
    results: list[AnswerEvalResult],
    created_at: str,
    git_commit: str | None = None,
    git_dirty: bool | None = None,
) -> AnswerEvalReport:
    metadata = build_eval_report_metadata(
        report_type="answer_eval_report",
        created_at=created_at,
        git_commit=git_commit,
        git_dirty=git_dirty,
    )
    return {
        **metadata,
        "total_cases": len(results),
        "passed_cases": count_passed_answer_results(results),
        "accuracy": calculate_answer_accuracy(results),
        "results": results,
    }


def save_answer_eval_report(report: AnswerEvalReport, output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
