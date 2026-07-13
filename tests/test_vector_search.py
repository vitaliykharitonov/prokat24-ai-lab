import pytest

from prokat24_ai_lab.vector_search import rank_vectors_by_similarity


def test_rank_vectors_by_similarity_returns_most_similar_first() -> None:
    query_vector = [1.0, 0.0]

    document_vectors = {
        "same": [1.0, 0.0],
        "perpendicular": [0.0, 1.0],
        "opposite": [-1.0, 0.0],
    }

    result = rank_vectors_by_similarity(query_vector, document_vectors)

    assert result == [
        ("same", 1.0),
        ("perpendicular", 0.0),
        ("opposite", -1.0),
    ]


def test_rank_vectors_by_similarity_returns_empty_list_for_no_documents() -> None:
    query_vector = [1.0, 0.0]
    document_vectors = {}

    result = rank_vectors_by_similarity(query_vector, document_vectors)

    assert result == []


def test_rank_vectors_by_similarity_raises_value_error_for_different_vector_lengths() -> None:
    query_vector = [1.0, 0.0]

    document_vectors = {
        "invalid": [1.0, 0.0, 0.0],
    }

    with pytest.raises(ValueError):
        rank_vectors_by_similarity(query_vector, document_vectors)


def test_rank_vectors_by_similarity_sorts_fractional_scores() -> None:
    query_vector = [1.0, 0.0]

    document_vectors = {
        "closest": [1.0, 0.2],
        "middle": [1.0, 1.0],
        "farthest": [0.2, 1.0],
    }

    result = rank_vectors_by_similarity(query_vector, document_vectors)

    assert [title for title, _ in result] == [
        "closest",
        "middle",
        "farthest",
    ]


def test_rank_vectors_by_similarity_limits_number_of_results() -> None:
    query_vector = [1.0, 0.0]

    document_vectors = {
        "same": [1.0, 0.0],
        "perpendicular": [0.0, 1.0],
        "opposite": [-1.0, 0.0],
    }

    result = rank_vectors_by_similarity(query_vector, document_vectors, top_k=2)

    assert result == [
        ("same", 1.0),
        ("perpendicular", 0.0),
    ]


@pytest.mark.parametrize(
    "top_k",
    [0, -1],
)
def test_rank_vectors_by_similarity_raises_value_error_for_invalid_top_k(top_k: int) -> None:
    query_vector = [1.0, 0.0]

    document_vectors = {
        "document": [1.0, 0.0],
    }

    with pytest.raises(ValueError):
        rank_vectors_by_similarity(
            query_vector,
            document_vectors,
            top_k=top_k,
        )
