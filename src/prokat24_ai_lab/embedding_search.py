from collections.abc import Callable

from prokat24_ai_lab.knowledge_base import KnowledgePage
from prokat24_ai_lab.vector_search import rank_vectors_by_similarity


def rank_texts_by_embeddings(
    query: str,
    documents: dict[str, str],
    embed_text: Callable[[str], list[float]],
    top_k: int | None = None,
) -> list[tuple[str, float]]:
    if not documents:
        return []

    embedding_cache: dict[str, list[float]] = {}

    def get_embedding(text: str) -> list[float]:
        if text not in embedding_cache:
            embedding_cache[text] = embed_text(text)
        return embedding_cache[text]

    query_vector = get_embedding(query)
    document_vectors: dict[str, list[float]] = {}

    for document_id, text in documents.items():
        document_vectors[document_id] = get_embedding(text)

    return rank_query_embedding_by_document_embeddings(query_vector, document_vectors, top_k=top_k)


def build_embedding_documents(pages: list[KnowledgePage]) -> dict[str, str]:
    documents: dict[str, str] = {}

    for page in pages:
        url = page["url"]
        text = page["title"] + "\n\n" + page["text"]
        documents[url] = text

    return documents


def rank_query_by_document_embeddings(
    query: str,
    document_embeddings: dict[str, list[float]],
    embed_text: Callable[[str], list[float]],
    top_k: int | None = None,
) -> list[tuple[str, float]]:
    if not document_embeddings:
        return []

    query_vector = embed_text(query)

    return rank_query_embedding_by_document_embeddings(
        query_vector, document_embeddings, top_k=top_k
    )


def rank_query_embedding_by_document_embeddings(
    query_embedding,
    document_embeddings,
    top_k: int | None = None,
) -> list[tuple[str, float]]:
    if not document_embeddings or not query_embedding:
        return []

    return rank_vectors_by_similarity(query_embedding, document_embeddings, top_k=top_k)
