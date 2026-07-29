from typing import TypedDict

from prokat24_ai_lab.evals.embedding_evals import EmbeddingSearchEvalCase
from prokat24_ai_lab.knowledge.base import KnowledgePage
from prokat24_ai_lab.retrieval.hybrid_search import rank_query_with_reciprocal_rank_fusion


class HybridSearchEvalResult(TypedDict):
    query: str
    expected_document_id: str
    actual_document_id: str | None
    actual_score: float | None
    expected_rank: int | None
    passed: bool


def evaluate_hybrid_search_case(
    case: EmbeddingSearchEvalCase,
    pages: list[KnowledgePage],
    document_embeddings: dict[str, list[float]],
    query_embedding: list[float],
    rank_constant: int = 60,
) -> HybridSearchEvalResult:
    query = case["query"]
    expected_document_id = case["expected_document_id"]
    actual_document_id: str | None = None
    actual_score: float | None = None
    expected_rank: int | None = None

    search_results = rank_query_with_reciprocal_rank_fusion(
        query,
        pages,
        document_embeddings,
        query_embedding,
        rank_constant,
    )

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
