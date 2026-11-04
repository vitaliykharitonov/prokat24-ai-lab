import json
from collections.abc import Callable
from pathlib import Path
from typing import TypedDict

from prokat24_ai_lab.embeddings.search import (
    rank_query_by_document_embeddings,
    rank_query_embedding_by_document_embeddings,
)
from prokat24_ai_lab.evals.report_metadata import build_eval_report_metadata
from prokat24_ai_lab.evals.retrieval_cases import (
    RetrievalCategory,
    RetrievalDifficulty,
    RetrievalEvalCase,
    RetrievalSplit,
)


class EmbeddingSearchEvalResult(TypedDict):
    id: str
    category: RetrievalCategory
    difficulty: RetrievalDifficulty
    split: RetrievalSplit
    query: str
    expected_document_id: str
    actual_document_id: str | None
    actual_score: float | None
    passed: bool
    expected_rank: int | None
    top_results: list[tuple[str, float]]


class EmbeddingSearchEvalReport(TypedDict):
    report_type: str
    schema_version: int
    created_at: str
    git_commit: str | None
    git_dirty: bool | None
    total_cases: int
    passed_cases: int
    accuracy: float
    top_3_accuracy: float
    results: list[EmbeddingSearchEvalResult]


def evaluate_embedding_search_case(
    case: RetrievalEvalCase,
    document_embeddings: dict[str, list[float]],
    embed_text: Callable[[str], list[float]],
) -> EmbeddingSearchEvalResult:
    query = case["query"]

    search_results = rank_query_by_document_embeddings(query, document_embeddings, embed_text)

    return build_embedding_search_eval_result(case, search_results)


def evaluate_precomputed_embedding_search_case(
    case: RetrievalEvalCase,
    document_embeddings: dict[str, list[float]],
    query_embedding: list[float],
) -> EmbeddingSearchEvalResult:
    search_results = rank_query_embedding_by_document_embeddings(
        query_embedding,
        document_embeddings,
    )

    return build_embedding_search_eval_result(case, search_results)


def build_embedding_search_eval_result(
    case: RetrievalEvalCase,
    search_results: list[tuple[str, float]],
) -> EmbeddingSearchEvalResult:
    expected_document_id = case["expected_document_id"]
    actual_document_id = None
    actual_score = None
    expected_rank = None

    if search_results:
        actual_document_id, actual_score = search_results[0]

    passed = actual_document_id == expected_document_id

    for index, (document_id, _) in enumerate(search_results, start=1):
        if document_id == expected_document_id:
            expected_rank = index
            break

    return {
        "id": case["id"],
        "category": case["category"],
        "difficulty": case["difficulty"],
        "split": case["split"],
        "query": case["query"],
        "expected_document_id": expected_document_id,
        "actual_document_id": actual_document_id,
        "actual_score": actual_score,
        "passed": passed,
        "expected_rank": expected_rank,
        "top_results": search_results,
    }


def run_embedding_search_evals(
    cases: list[RetrievalEvalCase],
    document_embeddings: dict[str, list[float]],
    embed_text: Callable[[str], list[float]],
) -> list[EmbeddingSearchEvalResult]:
    results: list[EmbeddingSearchEvalResult] = []

    for case in cases:
        results.append(evaluate_embedding_search_case(case, document_embeddings, embed_text))

    return results


def run_precomputed_embedding_search_evals(
    cases: list[RetrievalEvalCase],
    document_embeddings: dict[str, list[float]],
    query_embeddings: dict[str, list[float]],
) -> list[EmbeddingSearchEvalResult]:
    results: list[EmbeddingSearchEvalResult] = []

    for case in cases:
        query = case["query"]
        try:
            query_embedding = query_embeddings[query]

        except KeyError:
            raise ValueError(f"Query embedding not found: {query}") from None

        results.append(
            evaluate_precomputed_embedding_search_case(
                case,
                document_embeddings,
                query_embedding,
            )
        )

    return results


def count_passed_results(
    results: list[EmbeddingSearchEvalResult],
) -> int:
    passed_count = 0
    for result in results:
        if result["passed"]:
            passed_count += 1

    return passed_count


def calculate_accuracy(
    results: list[EmbeddingSearchEvalResult],
) -> float:
    if not results:
        return 0.0

    total_cases = len(results)
    passed_count = count_passed_results(results)

    return passed_count / total_cases


def calculate_top_k_accuracy(
    results: list[EmbeddingSearchEvalResult],
    top_k: int,
) -> float:
    if not results:
        return 0.0

    passed_count = 0

    for result in results:
        for top_result in result["top_results"][:top_k]:
            if top_result[0] == result["expected_document_id"]:
                passed_count += 1
                break

    return passed_count / len(results)


def build_embedding_search_eval_report(
    results: list[EmbeddingSearchEvalResult],
    created_at: str,
    git_commit: str | None = None,
    git_dirty: bool | None = None,
) -> EmbeddingSearchEvalReport:
    metadata = build_eval_report_metadata(
        report_type="embedding_search_eval_report",
        created_at=created_at,
        git_commit=git_commit,
        git_dirty=git_dirty,
    )

    return {
        **metadata,
        "total_cases": len(results),
        "passed_cases": count_passed_results(results),
        "accuracy": calculate_accuracy(results),
        "top_3_accuracy": calculate_top_k_accuracy(results, top_k=3),
        "results": results,
    }


def save_embedding_search_eval_report(
    report: EmbeddingSearchEvalReport,
    output_file: Path,
) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
