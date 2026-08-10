from prokat24_ai_lab.knowledge.base import KnowledgePage
from prokat24_ai_lab.rag.ranker import create_hybrid_document_ranker


def test_create_hybrid_document_ranker_embeds_query_and_returns_ranking() -> None:
    pages: list[KnowledgePage] = [
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
    ]
    document_embeddings = {
        "delivery": [1.0, 0.0],
        "contacts": [0.0, 1.0],
    }
    query = "доставка"

    received_queries: list[str] = []

    def fake_embed_text(text: str) -> list[float]:
        received_queries.append(text)
        return [1.0, 0.0]

    rank_documents = create_hybrid_document_ranker(
        pages=pages,
        document_embeddings=document_embeddings,
        embed_text=fake_embed_text,
    )

    result = rank_documents(query)

    assert received_queries == [query]
    assert result[0][0] == "delivery"

    document_ids = {document_id for document_id, _score in result}

    assert document_ids == {"delivery", "contacts"}
