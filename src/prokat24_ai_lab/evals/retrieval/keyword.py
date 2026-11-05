import json
from pathlib import Path
from typing import TypedDict

from prokat24_ai_lab.evals.report_metadata import build_eval_report_metadata
from prokat24_ai_lab.evals.retrieval.cases import (
    RetrievalCategory,
    RetrievalDifficulty,
    RetrievalEvalCase,
    RetrievalSplit,
)
from prokat24_ai_lab.knowledge.base import KnowledgePage
from prokat24_ai_lab.knowledge.keyword_search import (
    KnowledgeSearchResult,
    get_query_words,
    search_knowledge_base,
)


class SearchEvalResult(TypedDict):
    id: str
    category: RetrievalCategory
    difficulty: RetrievalDifficulty
    split: RetrievalSplit
    query: str
    query_words: list[str]
    expected_document_id: str
    actual_document_id: str | None
    actual_title: str | None
    actual_score: float | None
    expected_rank: int | None
    passed: bool
    top_results: list[KnowledgeSearchResult]


class SearchEvalReport(TypedDict):
    report_type: str
    schema_version: int
    created_at: str
    git_commit: str | None
    git_dirty: bool | None
    total_cases: int
    passed_cases: int
    accuracy: float
    top_3_accuracy: float
    results: list[SearchEvalResult]


def evaluate_search_case(
    case: RetrievalEvalCase,
    pages: list[KnowledgePage],
) -> SearchEvalResult:
    query = case["query"]
    query_words = get_query_words(query)
    expected_document_id = case["expected_document_id"]

    actual_document_id = None
    actual_title = None
    actual_score = None
    expected_rank = None

    search_results = search_knowledge_base(query, pages)
    top_results = search_results[:3]

    if search_results:
        top_result = search_results[0]
        actual_document_id = top_result["url"]
        actual_title = top_result["title"]
        actual_score = top_result["score"]

    for index, search_result in enumerate(search_results, start=1):
        if search_result["url"] == expected_document_id:
            expected_rank = index
            break

    passed = actual_document_id == expected_document_id

    return {
        "id": case["id"],
        "category": case["category"],
        "difficulty": case["difficulty"],
        "split": case["split"],
        "query": query,
        "query_words": query_words,
        "expected_document_id": expected_document_id,
        "actual_document_id": actual_document_id,
        "actual_title": actual_title,
        "actual_score": actual_score,
        "expected_rank": expected_rank,
        "passed": passed,
        "top_results": top_results,
    }


def run_search_evals(
    cases: list[RetrievalEvalCase],
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
    git_commit: str | None = None,
    git_dirty: bool | None = None,
) -> SearchEvalReport:
    metadata = build_eval_report_metadata(
        report_type="search_eval_report",
        created_at=created_at,
        git_commit=git_commit,
        git_dirty=git_dirty,
    )
    return {
        **metadata,
        "total_cases": len(results),
        "passed_cases": count_passed_results(results),
        "accuracy": calculate_accuracy(results),
        "top_3_accuracy": calculate_top_k_accuracy(results, k=3),
        "results": results,
    }


def save_search_eval_report(
    report: SearchEvalReport,
    output_file: Path,
) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)


def calculate_top_k_accuracy(
    results: list[SearchEvalResult],
    k: int,
) -> float:
    if not results:
        return 0.0

    passed_count = 0

    for result in results:
        expected_rank = result["expected_rank"]

        if expected_rank is not None and expected_rank <= k:
            passed_count += 1

    return passed_count / len(results)
