import pytest

from prokat24_ai_lab.text_similarity_search import rank_texts_by_similarity


def test_rank_texts_by_similarity_returns_relevant_document_first() -> None:
    query = "Чи можлива доставка?"
    documents = {
        "delivery": "Доставка спорядження по Києву",
        "contacts": "Наші адреси та телефони",
        "rules": "Для оренди потрібна застава",
    }
    vocabulary = ["доставка", "адреси", "застава"]

    result = rank_texts_by_similarity(query, documents, vocabulary)

    assert result[0][0] == "delivery"


def test_rank_texts_by_similarity_returns_empty_list_for_unknown_query_words() -> None:
    query = "Хочу взяти лижі"
    documents = {
        "delivery": "Доставка спорядження по Києву",
        "contacts": "Наші адреси та телефони",
        "rules": "Для оренди потрібна застава",
    }
    vocabulary = ["доставка", "адреси", "застава"]

    result = rank_texts_by_similarity(query, documents, vocabulary)

    assert result == []


def test_rank_texts_by_similarity_skips_documents_with_unknown_words() -> None:
    query = "Чи можлива доставка?"
    documents = {
        "delivery": "Доставка спорядження по Києву",
        "unknown": "Прокат лиж та сноубордів",
    }
    vocabulary = ["доставка", "адреси", "застава"]

    result = rank_texts_by_similarity(query, documents, vocabulary)

    assert result == [("delivery", 1.0)]


def test_rank_texts_by_similarity_ranks_more_relevant_document_first() -> None:
    query = "Чи є доставка та яка застава?"

    documents = {
        "delivery_and_deposit": "Інформація: доставка та застава",
        "delivery_only": "Доставка спорядження по Києву",
    }

    vocabulary = ["доставка", "адреси", "застава"]

    result = rank_texts_by_similarity(query, documents, vocabulary)

    assert [document for document, _ in result] == [
        "delivery_and_deposit",
        "delivery_only",
    ]
    assert result[0][1] > result[1][1]


def test_rank_texts_by_similarity_matches_word_forms_in_vocabulary() -> None:
    query = "Чи є доставка та застава?"
    documents = {
        "different_forms": "Інформація про доставку та заставу",
    }
    vocabulary = ["доставка", "застава"]
    word_forms = {
        "доставка": ["доставку"],
        "застава": ["заставу"],
    }

    result = rank_texts_by_similarity(query, documents, vocabulary, word_forms=word_forms)

    assert result[0][0] == "different_forms"
    assert result[0][1] == pytest.approx(1.0)


def test_rank_texts_by_similarity_matches_word_forms_in_query() -> None:
    query = "Чи можна доставку?"
    documents = {
        "delivery": "Доставка спорядження по Києву",
    }
    vocabulary = ["доставка"]
    word_forms = {
        "доставка": ["доставку"],
    }

    result = rank_texts_by_similarity(query, documents, vocabulary, word_forms=word_forms)

    assert result[0][0] == "delivery"
    assert result[0][1] == pytest.approx(1.0)


def test_rank_texts_by_similarity_limits_number_of_results() -> None:
    query = "Доставка та застава"

    documents = {
        "both": "Доставка та застава",
        "delivery": "Доставка спорядження",
        "deposit": "Потрібна застава",
    }

    vocabulary = ["доставка", "застава"]

    result = rank_texts_by_similarity(query, documents, vocabulary, top_k=2)

    assert len(result) == 2
    assert [document for document, _ in result] == [
        "both",
        "delivery",
    ]
