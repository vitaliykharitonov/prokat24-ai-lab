import json
from pathlib import Path
from typing import TypedDict

from prokat24_ai_lab.evals.embedding_evals import EmbeddingSearchEvalCase
from prokat24_ai_lab.evals.report_metadata import build_eval_report_metadata
from prokat24_ai_lab.knowledge.base import KnowledgePage
from prokat24_ai_lab.retrieval.hybrid_search import (
    rank_query_with_reciprocal_rank_fusion,
    rank_query_with_weighted_reciprocal_rank_fusion,
)


class HybridSearchEvalResult(TypedDict):
    query: str
    expected_document_id: str
    actual_document_id: str | None
    actual_score: float | None
    expected_rank: int | None
    passed: bool


class HybridSearchEvalSummary(TypedDict):
    total_cases: int
    passed_cases: int
    accuracy: float
    top_3_passed_cases: int
    top_3_accuracy: float


class HybridSearchEvalReport(TypedDict):
    report_type: str
    schema_version: int
    created_at: str
    git_commit: str | None
    git_dirty: bool | None

    rank_constant: int
    total_cases: int
    passed_cases: int
    accuracy: float
    top_3_passed_cases: int
    top_3_accuracy: float

    results: list[HybridSearchEvalResult]


def build_hybrid_search_eval_result(
    case: EmbeddingSearchEvalCase,
    search_results: list[tuple[str, float]],
) -> HybridSearchEvalResult:
    query = case["query"]
    expected_document_id = case["expected_document_id"]
    actual_document_id: str | None = None
    actual_score: float | None = None
    expected_rank: int | None = None

    if search_results:
        actual_document_id, actual_score = search_results[0]

    for index, (document_id, _score) in enumerate(search_results, start=1):
        if document_id == expected_document_id:
            expected_rank = index
            break

    passed = expected_document_id == actual_document_id

    return {
        "query": query,
        "expected_document_id": expected_document_id,
        "actual_document_id": actual_document_id,
        "actual_score": actual_score,
        "expected_rank": expected_rank,
        "passed": passed,
    }


def evaluate_hybrid_search_case(
    case: EmbeddingSearchEvalCase,
    pages: list[KnowledgePage],
    document_embeddings: dict[str, list[float]],
    query_embedding: list[float],
    rank_constant: int = 60,
) -> HybridSearchEvalResult:
    query = case["query"]

    search_results = rank_query_with_reciprocal_rank_fusion(
        query,
        pages,
        document_embeddings,
        query_embedding,
        rank_constant,
    )

    return build_hybrid_search_eval_result(case, search_results)


def evaluate_weighted_hybrid_search_case(
    case: EmbeddingSearchEvalCase,
    pages: list[KnowledgePage],
    document_embeddings: dict[str, list[float]],
    query_embedding: list[float],
    keyword_weight: float,
    embedding_weight: float,
    rank_constant: int = 60,
) -> HybridSearchEvalResult:
    query = case["query"]

    search_results = rank_query_with_weighted_reciprocal_rank_fusion(
        query,
        pages,
        document_embeddings,
        query_embedding,
        keyword_weight,
        embedding_weight,
        rank_constant,
    )

    return build_hybrid_search_eval_result(case, search_results)


def run_hybrid_search_evals(
    cases: list[EmbeddingSearchEvalCase],
    pages: list[KnowledgePage],
    document_embeddings: dict[str, list[float]],
    query_embeddings: dict[str, list[float]],
    rank_constant: int = 60,
) -> list[HybridSearchEvalResult]:
    results = []
    for case in cases:
        query = case["query"]
        try:
            query_embedding = query_embeddings[query]
        except KeyError:
            raise ValueError(f"Query embedding not found: {query}") from None

        results.append(
            evaluate_hybrid_search_case(
                case,
                pages,
                document_embeddings,
                query_embedding,
                rank_constant=rank_constant,
            )
        )

    return results


def run_weighted_hybrid_search_evals(
    cases: list[EmbeddingSearchEvalCase],
    pages: list[KnowledgePage],
    document_embeddings: dict[str, list[float]],
    query_embeddings: dict[str, list[float]],
    keyword_weight: float,
    embedding_weight: float,
    rank_constant: int = 60,
) -> list[HybridSearchEvalResult]:
    results = []
    for case in cases:
        query = case["query"]
        try:
            query_embedding = query_embeddings[query]
        except KeyError:
            raise ValueError(f"Query embedding not found: {query}") from None

        results.append(
            evaluate_weighted_hybrid_search_case(
                case,
                pages,
                document_embeddings,
                query_embedding,
                keyword_weight=keyword_weight,
                embedding_weight=embedding_weight,
                rank_constant=rank_constant,
            )
        )

    return results


def build_hybrid_search_eval_summary(
    results: list[HybridSearchEvalResult],
) -> HybridSearchEvalSummary:
    total_cases = len(results)
    passed_cases = 0
    top_3_passed_cases = 0

    for result in results:
        if result["passed"]:
            passed_cases += 1

        if result["expected_rank"] is not None and result["expected_rank"] <= 3:
            top_3_passed_cases += 1

    if total_cases == 0:
        accuracy = 0.0
        top_3_accuracy = 0.0

    else:
        accuracy = passed_cases / total_cases
        top_3_accuracy = top_3_passed_cases / total_cases

    return {
        "total_cases": total_cases,
        "passed_cases": passed_cases,
        "accuracy": accuracy,
        "top_3_passed_cases": top_3_passed_cases,
        "top_3_accuracy": top_3_accuracy,
    }


def build_hybrid_search_eval_report(
    results: list[HybridSearchEvalResult],
    created_at: str,
    rank_constant: int,
    git_commit: str | None = None,
    git_dirty: bool | None = None,
) -> HybridSearchEvalReport:
    metadata = build_eval_report_metadata(
        report_type="hybrid_search_eval_report",
        created_at=created_at,
        git_commit=git_commit,
        git_dirty=git_dirty,
    )

    summary = build_hybrid_search_eval_summary(results)

    return {
        **metadata,
        "rank_constant": rank_constant,
        **summary,
        "results": results,
    }


def save_hybrid_search_eval_report(
    report: HybridSearchEvalReport,
    output_file: Path,
) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
