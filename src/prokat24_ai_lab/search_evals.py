import json
from pathlib import Path
from typing import TypedDict

from prokat24_ai_lab.knowledge_base import KnowledgePage
from prokat24_ai_lab.knowledge_search import KnowledgeSearchResult, search_knowledge_base


class SearchEvalCase(TypedDict):
    query: str
    expected_url: str


class SearchEvalResult(TypedDict):
    query: str
    expected_url: str
    actual_url: str | None
    actual_title: str | None
    actual_score: int | None
    passed: bool
    top_results: list[KnowledgeSearchResult]


class SearchEvalReport(TypedDict):
    created_at: str
    total_cases: int
    passed_cases: int
    accuracy: float
    results: list[SearchEvalResult]


def load_search_eval_cases(input_file: Path) -> list[SearchEvalCase]:
    with open(input_file, encoding="utf-8") as file:
        return json.load(file)


def evaluate_search_case(case: SearchEvalCase, pages: list[KnowledgePage]) -> SearchEvalResult:
    query = case["query"]
    expected_url = case["expected_url"]
    actual_url = None
    actual_title = None
    actual_score = None

    search_results = search_knowledge_base(query, pages)
    top_results = search_results[:3]

    if search_results:
        top_result = search_results[0]
        actual_url = top_result["url"]
        actual_title = top_result["title"]
        actual_score = top_result["score"]

    passed = actual_url == expected_url

    return {
        "query": query,
        "expected_url": expected_url,
        "actual_url": actual_url,
        "actual_title": actual_title,
        "actual_score": actual_score,
        "passed": passed,
        "top_results": top_results,
    }


def run_search_evals(
    cases: list[SearchEvalCase],
    pages: list[KnowledgePage],
) -> list[SearchEvalResult]:
    results: list[SearchEvalResult] = []

    for case in cases:
        results.append(evaluate_search_case(case, pages))

    return results


def calculate_accuracy(results: list[SearchEvalResult]) -> float:
    if not results:
        return 0.0

    total_cases = len(results)
    passed_count = count_passed_results(results)

    return passed_count / total_cases


def count_passed_results(results: list[SearchEvalResult]) -> int:
    passed_count = 0
    for result in results:
        if result["passed"]:
            passed_count += 1

    return passed_count


def build_search_eval_report(
    results: list[SearchEvalResult],
    created_at: str,
) -> SearchEvalReport:
    return {
        "created_at": created_at,
        "total_cases": len(results),
        "passed_cases": count_passed_results(results),
        "accuracy": calculate_accuracy(results),
        "results": results,
    }


def save_search_eval_report(
    report: SearchEvalReport,
    output_file: Path,
) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
