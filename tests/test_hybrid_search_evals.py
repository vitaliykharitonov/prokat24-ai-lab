from prokat24_ai_lab.evals.embedding_evals import EmbeddingSearchEvalCase
from prokat24_ai_lab.evals.hybrid_search_evals import evaluate_hybrid_search_case
from prokat24_ai_lab.knowledge.base import KnowledgePage


def test_evaluate_hybrid_search_case_returns_passed_result() -> None:
    case: EmbeddingSearchEvalCase = {
        "query": "де вас знайти",
        "expected_document_id": "contact-us",
    }
    pages: list[KnowledgePage] = [
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
    document_embeddings = {
        "dostavka": [1.0, 0.0],
        "contact-us": [0.0, 1.0],
        "pravila-prokata": [0.8, 0.2],
    }
    query_embedding = [1.0, 0.0]

    result = evaluate_hybrid_search_case(
        case,
        pages,
        document_embeddings,
        query_embedding,
        rank_constant=60,
    )

    assert result["query"] == case["query"]
    assert result["expected_document_id"] == case["expected_document_id"]
    assert result["actual_document_id"] == case["expected_document_id"]
    assert result["actual_score"] is not None
    assert result["expected_rank"] == 1
    assert result["passed"] is True


def test_evaluate_hybrid_search_case_returns_expected_document_rank() -> None:
    case: EmbeddingSearchEvalCase = {
        "query": "доставка спорядження",
        "expected_document_id": "contact-us",
    }
    pages: list[KnowledgePage] = [
        {
            "url": "dostavka",
            "title": "Доставка спорядження",
            "text": "Умови та вартість доставки спорядження.",
            "text_length": 40,
        },
        {
            "url": "contact-us",
            "title": "Контакти",
            "text": "Телефон та електронна пошта.",
            "text_length": 28,
        },
        {
            "url": "pravila-prokata",
            "title": "Правила прокату",
            "text": "Строк оренди та застава.",
            "text_length": 25,
        },
    ]
    document_embeddings = {
        "contact-us": [1.0, 0.0],
        "dostavka": [0.9, 0.1],
        "pravila-prokata": [0.0, 1.0],
    }
    query_embedding = [1.0, 0.0]

    result = evaluate_hybrid_search_case(
        case,
        pages,
        document_embeddings,
        query_embedding,
        rank_constant=60,
    )

    assert result["actual_document_id"] == "dostavka"
    assert result["expected_rank"] == 2
    assert result["passed"] is False
