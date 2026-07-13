from prokat24_ai_lab.vector_math import cosine_similarity


def rank_vectors_by_similarity(
    query_vector: list[float],
    document_vectors: dict[str, list[float]],
    top_k: int | None = None,
) -> list[tuple[str, float]]:
    if top_k is not None and top_k <= 0:
        raise ValueError("top_k must be > 0")

    documents_by_similarity = []

    for title, document_vector in document_vectors.items():
        document_similarity = cosine_similarity(query_vector, document_vector)

        documents_by_similarity.append((title, document_similarity))

    return sorted(documents_by_similarity, key=lambda x: x[1], reverse=True)[:top_k]
