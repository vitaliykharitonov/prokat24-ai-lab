import pytest

from prokat24_ai_lab.embedding_search import rank_texts_by_embeddings


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
