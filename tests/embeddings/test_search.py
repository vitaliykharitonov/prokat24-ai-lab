import pytest

from prokat24_ai_lab.embeddings.search import (
    build_embedding_documents,
    rank_query_by_document_embeddings,
    rank_query_embedding_by_document_embeddings,
    rank_texts_by_embeddings,
)
from prokat24_ai_lab.knowledge.base import KnowledgePage


def fake_embed_text(text: str) -> list[float]:
    embeddings = {
        "привезти додому": [1.0, 0.0],
        "доставка по києву": [0.9, 0.1],
        "адреса пункту прокату": [0.0, 1.0],
        "умови застави": [-1.0, 0.0],
    }

    return embeddings[text]


def test_rank_texts_by_embeddings_ranks_top_k_and_embeds_each_text_once() -> None:
    query = "привезти додому"
    documents = {
        "delivery": "доставка по києву",
        "contacts": "адреса пункту прокату",
        "deposit": "умови застави",
    }
    embedded_texts: list[str] = []

    def tracking_embed_text(text: str) -> list[float]:
        embedded_texts.append(text)
        return fake_embed_text(text)

    result = rank_texts_by_embeddings(query, documents, tracking_embed_text, top_k=2)

    assert result[0][0] == "delivery"
    assert result[1][0] == "contacts"
    assert result[0][1] > result[1][1]
    assert len(result) == 2
    assert embedded_texts == [
        "привезти додому",
        "доставка по києву",
        "адреса пункту прокату",
        "умови застави",
    ]


def test_rank_texts_by_embeddings_returns_empty_list_without_embedding_query() -> None:
    query = "привезти додому"
    documents = {}
    embedded_texts: list[str] = []

    def tracking_embed_text(text: str) -> list[float]:
        embedded_texts.append(text)
        return fake_embed_text(text)

    result = rank_texts_by_embeddings(query, documents, tracking_embed_text)

    assert embedded_texts == []
    assert result == []


def test_rank_texts_by_embeddings_reuses_embedding_for_duplicate_texts() -> None:
    query = "привезти додому"
    documents = {
        "delivery_main": "доставка по києву",
        "delivery_copy": "доставка по києву",
    }
    embedded_texts: list[str] = []

    def tracking_embed_text(text: str) -> list[float]:
        embedded_texts.append(text)
        return fake_embed_text(text)

    result = rank_texts_by_embeddings(query, documents, tracking_embed_text)

    assert embedded_texts == [
        "привезти додому",
        "доставка по києву",
    ]
    assert {document_id for document_id, _ in result} == {
        "delivery_main",
        "delivery_copy",
    }
    assert result[0][1] == pytest.approx(result[1][1])
    assert len(result) == 2


def test_rank_texts_by_embeddings_reuses_embedding_when_query_matches_document() -> None:
    query = "доставка по києву"
    documents = {
        "delivery_main": "доставка по києву",
    }
    embedded_texts: list[str] = []

    def tracking_embed_text(text: str) -> list[float]:
        embedded_texts.append(text)
        return fake_embed_text(text)

    rank_texts_by_embeddings(query, documents, tracking_embed_text)

    assert embedded_texts == [
        "доставка по києву",
    ]


def test_build_embedding_documents_uses_urls_and_combines_title_with_text() -> None:
    pages: list[KnowledgePage] = [
        {
            "url": "https://prokat24.com.ua/ua/dostavka/",
            "title": "Доставка",
            "text": "Доставляємо спорядження по Києву.",
            "text_length": len("Доставляємо спорядження по Києву."),
        },
        {
            "url": "https://prokat24.com.ua/ua/other-page/",
            "title": "Контакти",
            "text": "Адреса та номер телефону.",
            "text_length": len("Адреса та номер телефону."),
        },
    ]

    result = build_embedding_documents(pages)

    assert result == {
        "https://prokat24.com.ua/ua/dostavka/": "Доставка\n\nДоставляємо спорядження по Києву.",
        "https://prokat24.com.ua/ua/other-page/": "Контакти\n\nАдреса та номер телефону.",
    }


def test_build_embedding_documents_adds_page_embedding_context() -> None:
    pages: list[KnowledgePage] = [
        {
            "url": "https://prokat24.com.ua/ua/contact-us/",
            "title": "Контакти",
            "text": "Адреса та номер телефону.",
            "text_length": len("Адреса та номер телефону."),
        },
    ]

    result = build_embedding_documents(pages)

    contact_document = result["https://prokat24.com.ua/ua/contact-us/"]

    assert "Контакти" in contact_document
    assert "Адреса та номер телефону." in contact_document
    assert "Адреси, телефони та контакти пунктів прокату." in contact_document
    assert "Місця самовивозу" in contact_document


def test_rank_query_by_document_embeddings_embeds_query_and_ranks_document_vectors() -> None:
    query = "привезти додому"
    query_vector = [1.0, 0.0]
    embedded_texts: list[str] = []

    def fake_embed_text(query: str) -> list[float]:
        embedded_texts.append(query)
        return query_vector

    document_embeddings = {
        "delivery": [1.0, 0.0],
        "contacts": [0.0, 1.0],
        "deposit": [-1.0, 0.0],
    }

    result = rank_query_by_document_embeddings(query, document_embeddings, fake_embed_text)

    assert result == [
        ("delivery", 1.0),
        ("contacts", 0.0),
        ("deposit", -1.0),
    ]
    assert embedded_texts == ["привезти додому"]


def test_rank_query_by_document_embeddings_returns_empty_list_without_embedding_query() -> None:
    query = "привезти додому"
    document_embeddings: dict[str, list[float]] = {}
    embedded_texts: list[str] = []

    def fake_embed_text(query: str) -> list[float]:
        embedded_texts.append(query)
        raise AssertionError("embed_text should not be called")

    result = rank_query_by_document_embeddings(query, document_embeddings, fake_embed_text)

    assert embedded_texts == []
    assert result == []


def test_rank_query_embedding_by_document_embeddings_ranks_precomputed_vectors() -> None:
    query_embedding = [1.0, 0.0]

    document_embeddings = {
        "delivery": [1.0, 0.0],
        "contacts": [0.0, 1.0],
        "deposit": [-1.0, 0.0],
    }

    result = rank_query_embedding_by_document_embeddings(
        query_embedding,
        document_embeddings,
        top_k=3,
    )

    assert result == [
        ("delivery", 1.0),
        ("contacts", 0.0),
        ("deposit", -1.0),
    ]


def test_rank_query_embedding_by_document_embeddings_limits_results_by_top_k() -> None:
    query_embedding = [1.0, 0.0]

    document_embeddings = {
        "delivery": [1.0, 0.0],
        "contacts": [0.0, 1.0],
        "deposit": [-1.0, 0.0],
    }

    result = rank_query_embedding_by_document_embeddings(
        query_embedding,
        document_embeddings,
        top_k=2,
    )

    assert result == [
        ("delivery", 1.0),
        ("contacts", 0.0),
    ]


def test_rank_query_embedding_by_document_embeddings_returns_empty_list_without_documents() -> None:
    query_embedding = [1.0, 0.0]

    document_embeddings = {}

    result = rank_query_embedding_by_document_embeddings(
        query_embedding,
        document_embeddings,
    )

    assert result == []


def test_rank_query_embedding_by_document_embeddings_returns_empty_list_for_empty_query() -> None:
    query_embedding = []

    document_embeddings = {
        "delivery": [1.0, 0.0],
        "contacts": [0.0, 1.0],
        "deposit": [-1.0, 0.0],
    }

    result = rank_query_embedding_by_document_embeddings(
        query_embedding,
        document_embeddings,
    )

    assert result == []
