import json
from pathlib import Path

from prokat24_ai_lab.embedding_index import (
    build_document_embedding_index,
    load_document_embedding_index,
    save_document_embedding_index,
)


def test_build_document_embedding_index_returns_embeddings_by_document_id() -> None:
    documents = {
        "delivery": "Доставка спорядження",
        "contacts": "Адреса пункту прокату",
    }
    embeddings = {
        "Доставка спорядження": [1.0, 0.0],
        "Адреса пункту прокату": [0.0, 1.0],
    }

    def fake_embed_text(text: str) -> list[float]:
        return embeddings[text]

    result = build_document_embedding_index(documents, fake_embed_text)

    assert result == {
        "delivery": [1.0, 0.0],
        "contacts": [0.0, 1.0],
    }


def test_save_document_embedding_index_writes_json(tmp_path: Path) -> None:
    output_file = tmp_path / "data" / "embedding_index.json"

    test_index = {
        "delivery": [1.0, 0.0],
        "contacts": [0.0, 1.0],
    }

    save_document_embedding_index(test_index, output_file)

    with open(output_file, encoding="utf-8") as file:
        loaded_index = json.load(file)

    assert loaded_index == test_index


def test_load_document_embedding_index_returns_saved_vectors(tmp_path: Path) -> None:
    intput_file = tmp_path / "embedding_index.json"

    test_index = {
        "delivery": [1.0, 0.0],
        "contacts": [0.0, 1.0],
    }

    with open(intput_file, "w", encoding="utf-8") as file:
        json.dump(test_index, file, ensure_ascii=False, indent=2)

    loaded_index = load_document_embedding_index(intput_file)

    assert loaded_index == test_index
