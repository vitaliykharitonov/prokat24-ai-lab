from prokat24_ai_lab.embeddings.search import rank_query_embedding_by_document_embeddings
from prokat24_ai_lab.knowledge.base import KnowledgePage
from prokat24_ai_lab.knowledge.keyword_search import search_knowledge_base
from prokat24_ai_lab.retrieval.rrf import reciprocal_rank_fusion


def rank_query_with_reciprocal_rank_fusion(
    query: str,
    pages: list[KnowledgePage],
    document_embeddings: dict[str, list[float]],
    query_embedding: list[float],
    rank_constant: int = 60,
) -> list[tuple[str, float]]:
    keyword_search_result = search_knowledge_base(query, pages)
    keyword_document_list: list[str] = []
    embedding_document_list: list[str] = []

    for result in keyword_search_result:
        keyword_document_list.append(result["url"])

    embedding_search_result = rank_query_embedding_by_document_embeddings(
        query_embedding,
        document_embeddings,
    )

    for document_id, _score in embedding_search_result:
        embedding_document_list.append(document_id)

    return reciprocal_rank_fusion(
        [keyword_document_list, embedding_document_list],
        rank_constant,
    )
