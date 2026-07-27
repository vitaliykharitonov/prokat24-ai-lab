from typing import TypedDict

from prokat24_ai_lab.embeddings.search import (
    rank_query_embedding_by_document_embeddings,
)
from prokat24_ai_lab.evals.embedding_evals import EmbeddingSearchEvalCase
from prokat24_ai_lab.knowledge.base import KnowledgePage
from prokat24_ai_lab.knowledge.keyword_search import search_knowledge_base


class RetrievalComparisonResult(TypedDict):
    query: str
    expected_document_id: str

    keyword_actual_document_id: str | None
    keyword_expected_rank: int | None
    keyword_passed: bool

    embedding_actual_document_id: str | None
    embedding_expected_rank: int | None
    embedding_passed: bool


class RetrievalComparisonSummary(TypedDict):
    total_cases: int

    keyword_passed_cases: int
    keyword_accuracy: float
    keyword_top_3_accuracy: float

    embedding_passed_cases: int
    embedding_accuracy: float
    embedding_top_3_accuracy: float

    both_passed_cases: int
    keyword_only_passed_cases: int
    embedding_only_passed_cases: int
    both_failed_cases: int


def build_retrieval_comparison_result(
    case: EmbeddingSearchEvalCase,
    keyword_document_ids: list[str],
    embedding_document_ids: list[str],
) -> RetrievalComparisonResult:
    query = case["query"]
    expected_document_id = case["expected_document_id"]
    keyword_actual_document_id = None
    embedding_actual_document_id = None

    if keyword_document_ids:
        keyword_actual_document_id = keyword_document_ids[0]

    keyword_passed = keyword_actual_document_id == expected_document_id

    if embedding_document_ids:
        embedding_actual_document_id = embedding_document_ids[0]

    embedding_passed = embedding_actual_document_id == expected_document_id

    def find_document_rank(document_ids: list[str], expected_document_id: str) -> int | None:
        for index, document_id in enumerate(document_ids, start=1):
            if document_id == expected_document_id:
                return index

        return None

    keyword_expected_rank = find_document_rank(keyword_document_ids, expected_document_id)

    embedding_expected_rank = find_document_rank(embedding_document_ids, expected_document_id)

    return {
        "query": query,
        "expected_document_id": expected_document_id,
        "keyword_actual_document_id": keyword_actual_document_id,
        "keyword_expected_rank": keyword_expected_rank,
        "keyword_passed": keyword_passed,
        "embedding_actual_document_id": embedding_actual_document_id,
        "embedding_expected_rank": embedding_expected_rank,
        "embedding_passed": embedding_passed,
    }


def evaluate_retrieval_comparison_case(
    case: EmbeddingSearchEvalCase,
    pages: list[KnowledgePage],
    document_embeddings: dict[str, list[float]],
    query_embedding: list[float],
) -> RetrievalComparisonResult:
    keyword_results = search_knowledge_base(case["query"], pages)
    keyword_document_ids = [keyword_result["url"] for keyword_result in keyword_results]
    embedding_results = rank_query_embedding_by_document_embeddings(
        query_embedding,
        document_embeddings,
    )
    embedding_document_ids = [document_id for document_id, _score in embedding_results]

    return build_retrieval_comparison_result(
        case,
        keyword_document_ids,
        embedding_document_ids,
    )


def run_retrieval_comparison_evals(
    cases: list[EmbeddingSearchEvalCase],
    pages: list[KnowledgePage],
    document_embeddings: dict[str, list[float]],
    query_embeddings: dict[str, list[float]],
) -> list[RetrievalComparisonResult]:
    results: list[RetrievalComparisonResult] = []
    for case in cases:
        query = case["query"]

        try:
            query_embedding = query_embeddings[query]

        except KeyError:
            raise ValueError(f"Query embedding not found: {query}") from None

        results.append(
            evaluate_retrieval_comparison_case(case, pages, document_embeddings, query_embedding)
        )

    return results


def build_retrieval_comparison_summary(
    results: list[RetrievalComparisonResult],
) -> RetrievalComparisonSummary:
    total_cases = len(results)
    keyword_passed_cases = 0
    embedding_passed_cases = 0
    keyword_top_3_passed_count = 0
    embedding_top_3_passed_count = 0
    both_passed_cases = 0
    keyword_only_passed_cases = 0
    embedding_only_passed_cases = 0
    both_failed_cases = 0

    for result in results:
        if result["keyword_passed"]:
            keyword_passed_cases += 1

        if result["embedding_passed"]:
            embedding_passed_cases += 1

        if result["keyword_expected_rank"] is not None and result["keyword_expected_rank"] <= 3:
            keyword_top_3_passed_count += 1

        if result["embedding_expected_rank"] is not None and result["embedding_expected_rank"] <= 3:
            embedding_top_3_passed_count += 1

        if result["keyword_passed"] and result["embedding_passed"]:
            both_passed_cases += 1

        if result["keyword_passed"] and not result["embedding_passed"]:
            keyword_only_passed_cases += 1

        if result["embedding_passed"] and not result["keyword_passed"]:
            embedding_only_passed_cases += 1

        if not result["keyword_passed"] and not result["embedding_passed"]:
            both_failed_cases += 1

    if total_cases == 0:
        keyword_accuracy = 0.0
        embedding_accuracy = 0.0
        keyword_top_3_accuracy = 0.0
        embedding_top_3_accuracy = 0.0
    else:
        keyword_accuracy = keyword_passed_cases / total_cases
        embedding_accuracy = embedding_passed_cases / total_cases
        keyword_top_3_accuracy = keyword_top_3_passed_count / total_cases
        embedding_top_3_accuracy = embedding_top_3_passed_count / total_cases

    return {
        "total_cases": total_cases,
        "keyword_passed_cases": keyword_passed_cases,
        "keyword_accuracy": keyword_accuracy,
        "keyword_top_3_accuracy": keyword_top_3_accuracy,
        "embedding_passed_cases": embedding_passed_cases,
        "embedding_accuracy": embedding_accuracy,
        "embedding_top_3_accuracy": embedding_top_3_accuracy,
        "both_passed_cases": both_passed_cases,
        "keyword_only_passed_cases": keyword_only_passed_cases,
        "embedding_only_passed_cases": embedding_only_passed_cases,
        "both_failed_cases": both_failed_cases,
    }
