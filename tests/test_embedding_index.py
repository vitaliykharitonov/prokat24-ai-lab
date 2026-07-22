import json
from pathlib import Path

import pytest

from prokat24_ai_lab.embedding_index import (
    build_document_embedding_index,
    create_document_embedding_index,
    load_document_embedding_index,
    save_document_embedding_index,
)


def test_build_document_embedding_index_returns_versioned_index() -> None:
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

    result = build_document_embedding_index(documents, fake_embed_text, "test-embedding-model")

    assert result == {
        "schema_version": 1,
        "model": "test-embedding-model",
        "dimensions": 2,
        "embeddings": {
            "delivery": [1.0, 0.0],
            "contacts": [0.0, 1.0],
        },
    }


def test_build_document_embedding_index_raises_value_error_for_empty_documents() -> None:
    documents = {}

    def fake_embed_text(text: str) -> list[float]:
        raise AssertionError("embed_text must not be called")

    with pytest.raises(ValueError, match="No embeddings found"):
        build_document_embedding_index(documents, fake_embed_text, "test-embedding-model")


def test_create_document_embedding_index_returns_versioned_index() -> None:
    embeddings = {
        "delivery": [1.0, 0.0],
        "contacts": [0.0, 1.0],
    }

    result = create_document_embedding_index(embeddings, "test-embedding-model")

    assert result == {
        "schema_version": 1,
        "model": "test-embedding-model",
        "dimensions": 2,
        "embeddings": {
            "delivery": [1.0, 0.0],
            "contacts": [0.0, 1.0],
        },
    }


def test_create_document_embedding_index_raises_value_error_for_different_dimensions() -> None:
    embeddings = {
        "delivery": [1.0, 0.0],
        "contacts": [0.0, 1.0, 0.5],
    }

    with pytest.raises(ValueError, match="Embedding dimensions do not match"):
        create_document_embedding_index(embeddings, "test-embedding-model")


def test_create_document_embedding_index_raises_value_error_for_empty_embeddings() -> None:
    embeddings = {}

    with pytest.raises(ValueError, match="No embeddings found"):
        create_document_embedding_index(embeddings, "test-embedding-model")


def test_save_document_embedding_index_writes_json(tmp_path: Path) -> None:
    output_file = tmp_path / "data" / "embedding_index.json"

    test_index = {
        "schema_version": 1,
        "model": "test-embedding-model",
        "dimensions": 2,
        "embeddings": {
            "delivery": [1.0, 0.0],
            "contacts": [0.0, 1.0],
        },
    }

    save_document_embedding_index(test_index, output_file)

    with open(output_file, encoding="utf-8") as file:
        loaded_index = json.load(file)

    assert loaded_index == test_index


def test_load_document_embedding_index_returns_saved_vectors(tmp_path: Path) -> None:
    input_file = tmp_path / "embedding_index.json"

    test_index = {
        "schema_version": 1,
        "model": "test-embedding-model",
        "dimensions": 2,
        "embeddings": {
            "delivery": [1.0, 0.0],
            "contacts": [0.0, 1.0],
        },
    }

    with open(input_file, "w", encoding="utf-8") as file:
        json.dump(test_index, file, ensure_ascii=False, indent=2)

    loaded_index = load_document_embedding_index(input_file)

    assert loaded_index == test_index
