import pytest

from prokat24_ai_lab.knowledge.base import KnowledgePage
from prokat24_ai_lab.rag.context import build_retrieved_context_documents, format_retrieved_context

PAGES: list[KnowledgePage] = [
    {
        "url": "delivery",
        "title": "Доставка",
        "text": "Спорядження можна отримати за допомогою доставки.",
        "text_length": 52,
    },
    {
        "url": "contacts",
        "title": "Контакти",
        "text": "Пункт прокату знаходиться біля метро Берестейська.",
        "text_length": 54,
    },
    {
        "url": "rules",
        "title": "Правила прокату",
        "text": "Клієнт повинен повернути спорядження у погоджений строк.",
        "text_length": 60,
    },
]

RANKED_DOCUMENTS: list[tuple[str, float]] = [
    ("delivery", 0.9),
    ("contacts", 0.8),
    ("rules", 0.7),
]

DOCUMENTS = [
    {
        "document_id": "delivery",
        "title": "Доставка",
        "text": "Спорядження можна отримати за допомогою доставки.",
        "score": 0.9,
    },
    {
        "document_id": "contacts",
        "title": "Контакти",
        "text": "Пункт прокату знаходиться біля метро Берестейська.",
        "score": 0.8,
    },
]


def test_build_retrieved_context_documents_returns_top_k_in_ranking_order():
    result = build_retrieved_context_documents(RANKED_DOCUMENTS, PAGES, top_k=2)

    assert result == [
        {
            "document_id": "delivery",
            "title": "Доставка",
            "text": "Спорядження можна отримати за допомогою доставки.",
            "score": 0.9,
        },
        {
            "document_id": "contacts",
            "title": "Контакти",
            "text": "Пункт прокату знаходиться біля метро Берестейська.",
            "score": 0.8,
        },
    ]


def test_build_retrieved_context_documents_raises_for_non_positive_top_k():
    with pytest.raises(ValueError, match="top_k must be > 0"):
        build_retrieved_context_documents(RANKED_DOCUMENTS, PAGES, top_k=-1)


def test_build_retrieved_context_documents_raises_for_missing_page():
    pages = []
    with pytest.raises(ValueError, match="Knowledge page not found for document_id: delivery"):
        build_retrieved_context_documents(RANKED_DOCUMENTS, pages, top_k=1)


def test_format_retrieved_context_formats_documents_in_order() -> None:
    result = format_retrieved_context(DOCUMENTS)
    assert result == (
        "[Document 1]\n"
        "ID: delivery\n"
        "Title: Доставка\n"
        "Content:\n"
        "Спорядження можна отримати за допомогою доставки.\n\n"
        "[Document 2]\n"
        "ID: contacts\n"
        "Title: Контакти\n"
        "Content:\n"
        "Пункт прокату знаходиться біля метро Берестейська."
    )


def test_format_retrieved_context_raises_for_empty_documents() -> None:
    documents = []

    with pytest.raises(ValueError, match="No context documents provided"):
        format_retrieved_context(documents)
