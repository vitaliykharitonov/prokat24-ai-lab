from collections.abc import Callable

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

    return rank_vectors_by_similarity(query_vector, document_vectors, top_k=top_k)
