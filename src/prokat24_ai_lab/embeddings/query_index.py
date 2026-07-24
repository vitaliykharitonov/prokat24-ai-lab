import json
from collections.abc import Callable
from pathlib import Path
from typing import TypedDict


class QueryEmbeddingIndex(TypedDict):
    schema_version: int
    model: str
    dimensions: int
    embeddings: dict[str, list[float]]


def build_query_embedding_index(
    queries: list[str],
    embed_text: Callable[[str], list[float]],
    model: str,
) -> QueryEmbeddingIndex:
    query_embeddings: dict[str, list[float]] = {}

    for query in queries:
        query_embeddings[query] = embed_text(query)

    if not query_embeddings:
        raise ValueError("No embeddings found")

    first_embedding = next(iter(query_embeddings.values()))
    dimensions = len(first_embedding)

    for embedding in query_embeddings.values():
        if len(embedding) != dimensions:
            raise ValueError("Embedding dimensions do not match")

    return {
        "schema_version": 1,
        "model": model,
        "dimensions": dimensions,
        "embeddings": query_embeddings,
    }


def save_query_embedding_index(index: QueryEmbeddingIndex, output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8", newline="\n") as file:
        json.dump(index, file, ensure_ascii=False, indent=2)


def load_query_embedding_index(input_file: Path) -> QueryEmbeddingIndex:
    with open(input_file, encoding="utf-8") as file:
        loaded_index: QueryEmbeddingIndex = json.load(file)

    return loaded_index
