from prokat24_ai_lab.evals.retrieval_comparison import (
    build_retrieval_comparison_result,
)


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
