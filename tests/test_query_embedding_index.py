from pathlib import Path

import pytest

from prokat24_ai_lab.query_embedding_index import (
    build_query_embedding_index,
    load_query_embedding_index,
    save_query_embedding_index,
)


def test_build_query_embedding_index_returns_versioned_index() -> None:
    queries = [
        "можете привезти спорядження мені додому?",
        "що потрібно залишити вам до повернення речей?",
    ]
    embeddings = {
        "можете привезти спорядження мені додому?": [1.0, 0.0],
        "що потрібно залишити вам до повернення речей?": [0.0, 1.0],
    }

    def fake_embed_text(text: str) -> list[float]:
        return embeddings[text]

    result = build_query_embedding_index(queries, fake_embed_text, "test-embedding-model")

    assert result == {
        "schema_version": 1,
        "model": "test-embedding-model",
        "dimensions": 2,
        "embeddings": {
            "можете привезти спорядження мені додому?": [1.0, 0.0],
            "що потрібно залишити вам до повернення речей?": [0.0, 1.0],
        },
    }


def test_build_query_embedding_index_raises_value_error_for_empty_queries() -> None:
    queries = []

    def fake_embed_text(_text: str) -> list[float]:
        raise AssertionError("embed_text must not be called")

    with pytest.raises(ValueError, match="No embeddings found"):
        build_query_embedding_index(queries, fake_embed_text, "test-embedding-model")


def test_build_query_embedding_index_raises_value_error_for_different_dimensions() -> None:
    queries = [
        "можете привезти спорядження мені додому?",
        "що потрібно залишити вам до повернення речей?",
    ]
    embeddings = {
        "можете привезти спорядження мені додому?": [1.0, 0.0],
        "що потрібно залишити вам до повернення речей?": [0.0, 1.0, 0.5],
    }

    def fake_embed_text(text: str) -> list[float]:
        return embeddings[text]

    with pytest.raises(ValueError, match="Embedding dimensions do not match"):
        build_query_embedding_index(queries, fake_embed_text, "test-embedding-model")


def test_save_and_load_query_embedding_index_preserves_data(tmp_path: Path) -> None:
    output_file = tmp_path / "data" / "query_embedding_index.json"

    test_index = {
        "schema_version": 1,
        "model": "test-embedding-model",
        "dimensions": 2,
        "embeddings": {
            "query one": [1.0, 0.0],
            "query two": [0.0, 1.0],
        },
    }

    save_query_embedding_index(test_index, output_file)

    loaded_index = load_query_embedding_index(output_file)

    assert loaded_index == test_index
