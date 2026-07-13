from prokat24_ai_lab.text_vectorizer import text_to_binary_vector
from prokat24_ai_lab.vector_search import rank_vectors_by_similarity


def rank_texts_by_similarity(
    query: str,
    documents: dict[str, str],
    vocabulary: list[str],
    top_k: int | None = None,
    word_forms: dict[str, list[str]] | None = None,
) -> list[tuple[str, float]]:
    query_vector = text_to_binary_vector(query, vocabulary, word_forms)

    if not any(query_vector):
        return []

    document_vectors: dict[str, list[float]] = {}

    for document, text in documents.items():
        document_vector = text_to_binary_vector(text, vocabulary, word_forms)

        if any(document_vector):
            document_vectors[document] = document_vector

    return rank_vectors_by_similarity(query_vector, document_vectors, top_k)
