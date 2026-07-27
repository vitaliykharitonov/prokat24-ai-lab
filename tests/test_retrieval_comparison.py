import pytest

from prokat24_ai_lab.evals.retrieval_comparison import (
    build_retrieval_comparison_result,
    evaluate_retrieval_comparison_case,
    run_retrieval_comparison_evals,
)
from prokat24_ai_lab.knowledge.base import KnowledgePage

TEST_PAGES: list[KnowledgePage] = [
    {
        "url": "contact-us",
        "title": "Контакти",
        "text": "Наша адреса та телефон",
        "text_length": 21,
    },
    {
        "url": "dostavka",
        "title": "Доставка",
        "text": "Доставка спорядження по Києву",
        "text_length": 30,
    },
]


def test_comparison_when_both_retrievers_return_expected_first() -> None:
    case = {
        "query": "де знаходяться пункти прокату?",
        "expected_document_id": "contact-us",
    }

    keyword_document_ids = [
        "contact-us",
        "o-nas",
        "dostavka",
    ]

    embedding_document_ids = [
        "contact-us",
        "pravila-prokata",
        "o-nas",
    ]

    result = build_retrieval_comparison_result(case, keyword_document_ids, embedding_document_ids)

    assert result["query"] == case["query"]
    assert result["expected_document_id"] == case["expected_document_id"]

    assert result["keyword_actual_document_id"] == "contact-us"
    assert result["keyword_expected_rank"] == 1
    assert result["keyword_passed"] is True

    assert result["embedding_actual_document_id"] == "contact-us"
    assert result["embedding_expected_rank"] == 1
    assert result["embedding_passed"] is True


def test_build_retrieval_comparison_result_when_keyword_is_first_and_embedding_is_second() -> None:
    case = {
        "query": "чи можна забрати спорядження самостійно?",
        "expected_document_id": "dostavka",
    }

    keyword_document_ids = [
        "dostavka",
        "pravila-prokata",
        "o-nas",
    ]

    embedding_document_ids = [
        "pravila-prokata",
        "dostavka",
        "o-nas",
    ]

    result = build_retrieval_comparison_result(case, keyword_document_ids, embedding_document_ids)

    assert result["query"] == case["query"]
    assert result["expected_document_id"] == case["expected_document_id"]

    assert result["keyword_actual_document_id"] == "dostavka"
    assert result["keyword_expected_rank"] == 1
    assert result["keyword_passed"] is True

    assert result["embedding_actual_document_id"] == "pravila-prokata"
    assert result["embedding_expected_rank"] == 2
    assert result["embedding_passed"] is False


def test_build_retrieval_comparison_result_when_both_retrievers_return_no_documents() -> None:
    case = {
        "query": "невідомий запит",
        "expected_document_id": "contact-us",
    }

    keyword_document_ids = []
    embedding_document_ids = []

    result = build_retrieval_comparison_result(case, keyword_document_ids, embedding_document_ids)

    assert result["query"] == case["query"]
    assert result["expected_document_id"] == case["expected_document_id"]

    assert result["keyword_actual_document_id"] is None
    assert result["keyword_expected_rank"] is None
    assert result["keyword_passed"] is False

    assert result["embedding_actual_document_id"] is None
    assert result["embedding_expected_rank"] is None
    assert result["embedding_passed"] is False


def test_evaluate_retrieval_comparison_case_returns_expected_first() -> None:
    case = {
        "query": "де ваша адреса?",
        "expected_document_id": "contact-us",
    }

    document_embeddings = {
        "contact-us": [1.0, 0.0],
        "dostavka": [0.0, 1.0],
    }

    query_embedding = [1.0, 0.0]

    result = evaluate_retrieval_comparison_case(
        case,
        TEST_PAGES,
        document_embeddings,
        query_embedding,
    )

    assert result["keyword_actual_document_id"] == "contact-us"
    assert result["keyword_expected_rank"] == 1
    assert result["keyword_passed"] is True

    assert result["embedding_actual_document_id"] == "contact-us"
    assert result["embedding_expected_rank"] == 1
    assert result["embedding_passed"] is True


def test_evaluate_retrieval_comparison_case_returns_expected_second() -> None:
    case = {
        "query": "де ваша адреса?",
        "expected_document_id": "contact-us",
    }

    document_embeddings = {
        "contact-us": [1.0, 0.0],
        "dostavka": [0.0, 1.0],
    }

    query_embedding = [0.0, 1.0]

    result = evaluate_retrieval_comparison_case(
        case,
        TEST_PAGES,
        document_embeddings,
        query_embedding,
    )

    assert result["keyword_actual_document_id"] == "contact-us"
    assert result["keyword_expected_rank"] == 1
    assert result["keyword_passed"] is True

    assert result["embedding_actual_document_id"] == "dostavka"
    assert result["embedding_expected_rank"] == 2
    assert result["embedding_passed"] is False


def test_run_retrieval_comparison_evals_returns_result_for_each_case() -> None:
    cases = [
        {
            "query": "де ваша адреса?",
            "expected_document_id": "contact-us",
        },
        {
            "query": "чи є доставка?",
            "expected_document_id": "dostavka",
        },
    ]
    document_embeddings = {
        "contact-us": [1.0, 0.0],
        "dostavka": [0.0, 1.0],
    }

    query_embeddings = {
        "де ваша адреса?": [1.0, 0.0],
        "чи є доставка?": [0.0, 1.0],
    }

    results = run_retrieval_comparison_evals(
        cases,
        TEST_PAGES,
        document_embeddings,
        query_embeddings,
    )

    assert len(results) == 2
    assert results[0]["query"] == "де ваша адреса?"
    assert results[1]["query"] == "чи є доставка?"
    assert results[0]["embedding_passed"] is True
    assert results[1]["embedding_passed"] is True


def test_run_retrieval_comparison_evals_raises_for_missing_query_embedding() -> None:
    cases = [
        {
            "query": "де ваша адреса?",
            "expected_document_id": "contact-us",
        },
    ]
    document_embeddings = {
        "contact-us": [1.0, 0.0],
        "dostavka": [0.0, 1.0],
    }

    query_embeddings = {}

    with pytest.raises(ValueError, match="Query embedding not found: де ваша адреса"):
        run_retrieval_comparison_evals(
            cases,
            TEST_PAGES,
            document_embeddings,
            query_embeddings,
        )
