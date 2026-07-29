import pytest

from prokat24_ai_lab.evals.embedding_evals import EmbeddingSearchEvalCase
from prokat24_ai_lab.evals.hybrid_search_evals import (
    evaluate_hybrid_search_case,
    run_hybrid_search_evals,
)
from prokat24_ai_lab.knowledge.base import KnowledgePage


@pytest.fixture
def knowledge_pages() -> list[KnowledgePage]:
    return [
        {
            "url": "dostavka",
            "title": "Доставка спорядження",
            "text": "Доставка спорядження по Києву. Умови та вартість доставки.",
            "text_length": 61,
        },
        {
            "url": "contact-us",
            "title": "Контакти",
            "text": "Зателефонуйте нам, щоб отримати спорядження.",
            "text_length": 47,
        },
        {
            "url": "pravila-prokata",
            "title": "Правила прокату",
            "text": "Строк оренди, застава та повернення речей.",
            "text_length": 44,
        },
    ]


def test_evaluate_hybrid_search_case_returns_passed_result(
    knowledge_pages: list[KnowledgePage],
) -> None:
    case: EmbeddingSearchEvalCase = {
        "query": "де вас знайти",
        "expected_document_id": "contact-us",
    }
    document_embeddings = {
        "dostavka": [1.0, 0.0],
        "contact-us": [0.0, 1.0],
        "pravila-prokata": [0.8, 0.2],
    }
    query_embedding = [1.0, 0.0]

    result = evaluate_hybrid_search_case(
        case,
        knowledge_pages,
        document_embeddings,
        query_embedding,
    )

    assert result["query"] == case["query"]
    assert result["expected_document_id"] == case["expected_document_id"]
    assert result["actual_document_id"] == case["expected_document_id"]
    assert result["actual_score"] is not None
    assert result["expected_rank"] == 1
    assert result["passed"] is True


def test_evaluate_hybrid_search_case_returns_expected_document_rank(
    knowledge_pages: list[KnowledgePage],
) -> None:
    case: EmbeddingSearchEvalCase = {
        "query": "доставка спорядження",
        "expected_document_id": "contact-us",
    }
    document_embeddings = {
        "contact-us": [1.0, 0.0],
        "dostavka": [0.9, 0.1],
        "pravila-prokata": [0.0, 1.0],
    }
    query_embedding = [1.0, 0.0]

    result = evaluate_hybrid_search_case(
        case,
        knowledge_pages,
        document_embeddings,
        query_embedding,
    )

    assert result["actual_document_id"] == "dostavka"
    assert result["expected_rank"] == 2
    assert result["passed"] is False


def test_evaluate_hybrid_search_case_returns_empty_result() -> None:
    case: EmbeddingSearchEvalCase = {
        "query": "невідомий запит",
        "expected_document_id": "missing-document",
    }

    result = evaluate_hybrid_search_case(
        case,
        [],
        {},
        [1.0, 0.0],
    )

    assert result["actual_document_id"] is None
    assert result["actual_score"] is None
    assert result["expected_rank"] is None
    assert result["passed"] is False


def test_run_hybrid_search_evals_returns_results_for_all_cases(
    knowledge_pages: list[KnowledgePage],
) -> None:
    cases: list[EmbeddingSearchEvalCase] = [
        {
            "query": "доставка спорядження",
            "expected_document_id": "dostavka",
        },
        {
            "query": "контактний телефон",
            "expected_document_id": "contact-us",
        },
    ]
    document_embeddings = {
        "dostavka": [1.0, 0.0, 0.0],
        "contact-us": [0.0, 1.0, 0.0],
        "pravila-prokata": [0.0, 0.0, 1.0],
    }
    query_embeddings = {
        "доставка спорядження": [1.0, 0.0, 0.0],
        "контактний телефон": [0.0, 1.0, 0.0],
    }

    results = run_hybrid_search_evals(
        cases,
        knowledge_pages,
        document_embeddings,
        query_embeddings,
    )

    assert len(results) == 2

    assert results[0]["query"] == "доставка спорядження"
    assert results[0]["actual_document_id"] == "dostavka"
    assert results[0]["passed"] is True

    assert results[1]["query"] == "контактний телефон"
    assert results[1]["actual_document_id"] == "contact-us"
    assert results[1]["passed"] is True


def test_run_hybrid_search_evals_raises_error_for_missing_query_embedding() -> None:
    cases: list[EmbeddingSearchEvalCase] = [
        {
            "query": "доставка спорядження",
            "expected_document_id": "dostavka",
        },
    ]

    with pytest.raises(ValueError, match="Query embedding not found: доставка спорядження"):
        run_hybrid_search_evals(
            cases,
            [],
            {},
            {},
        )
