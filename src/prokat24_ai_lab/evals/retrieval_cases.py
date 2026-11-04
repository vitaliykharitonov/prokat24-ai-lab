import json
from pathlib import Path
from typing import Literal, TypedDict

RetrievalCategory = Literal[
    "contacts",
    "delivery",
    "rental_rules",
    "rental_period",
    "deposit",
    "company_services",
    "sales",
    "equipment_care",
]

RetrievalDifficulty = Literal["easy", "medium", "hard"]
RetrievalSplit = Literal["dev", "holdout"]

RETRIEVAL_CATEGORIES: set[str] = {
    "contacts",
    "delivery",
    "rental_rules",
    "rental_period",
    "deposit",
    "company_services",
    "sales",
    "equipment_care",
}

RETRIEVAL_DIFFICULTIES: set[str] = {"easy", "medium", "hard"}

RETRIEVAL_SPLITS: set[str] = {"dev", "holdout"}


class RetrievalEvalCase(TypedDict):
    id: str
    query: str
    expected_document_id: str
    category: RetrievalCategory
    difficulty: RetrievalDifficulty
    split: RetrievalSplit


def validate_retrieval_eval_cases(
    cases: list[RetrievalEvalCase],
) -> None:
    seen_ids: set[str] = set()

    for case in cases:
        if not case["id"]:
            raise ValueError("Empty case id")

        if case["id"] in seen_ids:
            raise ValueError(f"Duplicate eval case id: {case['id']}")

        if not case["query"]:
            raise ValueError("Empty query")

        if not case["expected_document_id"]:
            raise ValueError("Empty expected document id")

        if case["category"] not in RETRIEVAL_CATEGORIES:
            raise ValueError(f"Unknown eval category: {case['category']}")

        if case["difficulty"] not in RETRIEVAL_DIFFICULTIES:
            raise ValueError(f"Unknown eval difficulty: {case['difficulty']}")

        if case["split"] not in RETRIEVAL_SPLITS:
            raise ValueError(f"Unknown eval split: {case['split']}")

        seen_ids.add(case["id"])


def load_retrieval_eval_cases(input_file: Path) -> list[RetrievalEvalCase]:
    with open(input_file, encoding="utf-8") as file:
        loaded_cases: list[RetrievalEvalCase] = json.load(file)

        validate_retrieval_eval_cases(loaded_cases)

    return loaded_cases
