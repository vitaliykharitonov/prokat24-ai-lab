from collections.abc import Callable

from prokat24_ai_lab.knowledge.base import KnowledgePage
from prokat24_ai_lab.rag.answer import DocumentRanker
from prokat24_ai_lab.retrieval.hybrid_search import rank_query_with_reciprocal_rank_fusion

EmbedText = Callable[[str], list[float]]


def create_hybrid_document_ranker(
    pages: list[KnowledgePage],
    document_embeddings: dict[str, list[float]],
    embed_text: EmbedText,
    rank_constant: int = 60,
) -> DocumentRanker:
    def rank_documents(query: str) -> list[tuple[str, float]]:
        query_embedding = embed_text(query)

        ranked_query = rank_query_with_reciprocal_rank_fusion(
            query,
            pages,
            document_embeddings,
            query_embedding,
            rank_constant,
        )

        return ranked_query

    return rank_documents
