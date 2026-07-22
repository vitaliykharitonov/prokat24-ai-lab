import json
from collections.abc import Callable
from pathlib import Path
from typing import TypedDict


class DocumentEmbeddingIndex(TypedDict):
    schema_version: int
    model: str
    dimensions: int
    embeddings: dict[str, list[float]]


def build_document_embedding_index(
    documents: dict[str, str],
    embed_text: Callable[[str], list[float]],
    model: str,
) -> DocumentEmbeddingIndex:
    embeddings: dict[str, list[float]] = {}

    for document_id, text in documents.items():
        embeddings[document_id] = embed_text(text)

    return create_document_embedding_index(embeddings, model)


def create_document_embedding_index(
    embeddings: dict[str, list[float]],
    model: str,
) -> DocumentEmbeddingIndex:
    if not embeddings:
        raise ValueError("No embeddings found")

    first_embedding = next(iter(embeddings.values()))
    dimensions = len(first_embedding)

    for embedding in embeddings.values():
        if len(embedding) != dimensions:
            raise ValueError("Embedding dimensions do not match")

    return {
        "schema_version": 1,
        "model": model,
        "dimensions": dimensions,
        "embeddings": embeddings,
    }


def save_document_embedding_index(index: DocumentEmbeddingIndex, output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8", newline="\n") as file:
        json.dump(index, file, ensure_ascii=False, indent=2)


def load_document_embedding_index(input_file: Path) -> DocumentEmbeddingIndex:
    with open(input_file, encoding="utf-8") as file:
        loaded_index: DocumentEmbeddingIndex = json.load(file)

    return loaded_index
