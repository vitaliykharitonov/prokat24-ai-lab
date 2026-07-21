import json
from collections.abc import Callable
from pathlib import Path


def build_document_embedding_index(
    documents: dict[str, str],
    embed_text: Callable[[str], list[float]],
) -> dict[str, list[float]]:
    index: dict[str, list[float]] = {}

    for document_id, text in documents.items():
        index[document_id] = embed_text(text)

    return index


def save_document_embedding_index(index: dict[str, list[float]], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(index, file, ensure_ascii=False, indent=2)


def load_document_embedding_index(input_file: Path) -> dict[str, list[float]]:
    with open(input_file, encoding="utf-8") as file:
        loaded_index: dict[str, list[float]] = json.load(file)

    return loaded_index
