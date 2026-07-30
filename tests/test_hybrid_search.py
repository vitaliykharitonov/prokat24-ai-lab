import pytest

from prokat24_ai_lab.knowledge.base import KnowledgePage
from prokat24_ai_lab.retrieval.hybrid_search import (
    rank_query_with_reciprocal_rank_fusion,
    rank_query_with_weighted_reciprocal_rank_fusion,
)


@pytest.fixture
def pages() -> list[KnowledgePage]:
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


def test_rank_query_with_reciprocal_rank_fusion_combines_search_results(
    pages: list[KnowledgePage],
) -> None:
    query = "доставка спорядження"

    document_embeddings = {
        "dostavka": [1.0, 0.0],
        "contact-us": [0.0, 1.0],
        "pravila-prokata": [0.8, 0.2],
    }

    query_embedding = [1.0, 0.0]

    result = rank_query_with_reciprocal_rank_fusion(
        query,
        pages,
        document_embeddings,
        query_embedding,
        rank_constant=60,
    )

    document_ids = [document_id for document_id, _score in result]
    scores = [score for _document_id, score in result]

    assert result[0][0] == "dostavka"
    assert len(document_ids) == 3
    assert len(document_ids) == len(set(document_ids))
    assert scores == sorted(scores, reverse=True)


def test_rank_query_with_weighted_reciprocal_rank_fusion_applies_source_weights(
    pages: list[KnowledgePage],
) -> None:
    query = "підкажіть контакти, щоб отримати спорядження з доставкою по Києву"

    document_embeddings = {
        "dostavka": [1.0, 0.0],
        "contact-us": [0.0, 1.0],
        "pravila-prokata": [0.8, 0.2],
    }

    query_embedding = [1.0, 0.0]
    keyword_weight = 1
    embedding_weight = 2

    result = rank_query_with_weighted_reciprocal_rank_fusion(
        query,
        pages,
        document_embeddings,
        query_embedding,
        keyword_weight,
        embedding_weight,
    )

    assert result[0][0] == "dostavka"
    assert len(result) == 3
    assert len({document_id for document_id, _score in result}) == 3
