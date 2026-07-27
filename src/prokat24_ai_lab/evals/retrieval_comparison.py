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
