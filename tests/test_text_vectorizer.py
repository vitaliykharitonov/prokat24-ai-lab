import pytest

from prokat24_ai_lab.text_vectorizer import text_to_binary_vector, text_to_count_vector
from prokat24_ai_lab.vector_math import cosine_similarity


def test_text_to_binary_vector_marks_present_words() -> None:
    text = "Яка застава та чи можлива доставка?"
    vocabulary = ["доставка", "адреса", "застава"]

    result = text_to_binary_vector(text, vocabulary)

    assert result == [1.0, 0.0, 1.0]


def test_text_to_binary_vector_does_not_match_part_of_word() -> None:
    text = "Потрібна застава"
    vocabulary = ["став", "застава"]

    result = text_to_binary_vector(text, vocabulary)

    assert result == [0.0, 1.0]


def test_text_to_binary_vector_returns_zero_vector_for_unknown_words() -> None:
    text = "Хочу взяти лижі"
    vocabulary = ["доставка", "адреса", "застава"]

    result = text_to_binary_vector(text, vocabulary)

    assert result == [0.0, 0.0, 0.0]


def test_text_to_binary_vector_matches_configured_word_forms() -> None:
    text = "Інформація про доставку та заставу"
    vocabulary = ["доставка", "застава"]

    word_forms = {
        "доставка": ["доставку"],
        "застава": ["заставу"],
    }

    result = text_to_binary_vector(text, vocabulary, word_forms)

    assert result == [1.0, 1.0]


def test_text_to_count_vector_counts_word_occurrences() -> None:
    text = "Доставка доставка та застава"
    vocabulary = ["доставка", "адреса", "застава"]

    result = text_to_count_vector(text, vocabulary)

    assert result == [2.0, 0.0, 1.0]


def test_repeated_words_reduce_count_vector_similarity() -> None:
    query = "доставка застава"
    balanced_document = "доставка застава"
    repeated_document = "доставка доставка доставка застава"
    vocabulary = ["доставка", "застава"]

    query_vector = text_to_count_vector(query, vocabulary)
    balanced_document_vector = text_to_count_vector(balanced_document, vocabulary)
    repeated_document_vector = text_to_count_vector(repeated_document, vocabulary)

    balanced_similarity = cosine_similarity(query_vector, balanced_document_vector)
    repeated_similarity = cosine_similarity(query_vector, repeated_document_vector)

    assert balanced_similarity == pytest.approx(1.0)
    assert repeated_similarity < balanced_similarity


def test_count_vectors_do_not_capture_semantic_similarity() -> None:
    query = "привезти додому"
    document = "доставка по києву"
    vocabulary = [
        "привезти",
        "додому",
        "доставка",
        "києву",
    ]

    query_vector = text_to_count_vector(query, vocabulary)
    document_vector = text_to_count_vector(document, vocabulary)

    similarity = cosine_similarity(query_vector, document_vector)

    assert similarity == pytest.approx(0.0)
