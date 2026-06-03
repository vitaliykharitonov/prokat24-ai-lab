import json
from pathlib import Path
from typing import TypedDict

from prokat24_ai_lab.knowledge_base import KnowledgePage
from prokat24_ai_lab.knowledge_search import search_knowledge_base


class SearchEvalCase(TypedDict):
    query: str
    expected_url: str


class SearchEvalResult(TypedDict):
    query: str
    expected_url: str
    actual_url: str | None
    passed: bool


def load_search_eval_cases(input_file: Path) -> list[SearchEvalCase]:
    with open(input_file, encoding="utf-8") as file:
        return json.load(file)


def evaluate_search_case(case: SearchEvalCase, pages: list[KnowledgePage]) -> SearchEvalResult:
    query = case["query"]
    expected_url = case["expected_url"]
    actual_url = None

    search_results = search_knowledge_base(query, pages)

    if search_results:
        actual_url = search_results[0]["url"]

    passed = actual_url == expected_url

    return {
        "query": query,
        "expected_url": expected_url,
        "actual_url": actual_url,
        "passed": passed,
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
    passed_count = 0

    for result in results:
        if result["passed"]:
            passed_count += 1

    return passed_count / total_cases


