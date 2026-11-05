import json
from pathlib import Path

import pytest

from prokat24_ai_lab.evals.retrieval.cases import RetrievalEvalCase, load_retrieval_eval_cases


def test_load_retrieval_eval_cases_returns_cases_from_json(tmp_path: Path) -> None:
    cases: list[RetrievalEvalCase] = [
        {
            "id": "test_case_001",
            "query": "привезти спорядження додому",
            "expected_document_id": "delivery",
            "category": "delivery",
            "difficulty": "easy",
            "split": "dev",
        },
        {
            "id": "test_case_002",
            "query": "де вас знайти",
            "expected_document_id": "contacts",
            "category": "contacts",
            "difficulty": "easy",
            "split": "dev",
        },
    ]
    input_file = tmp_path / "retrieval_eval_cases.json"

    with open(input_file, "w", encoding="utf-8") as file:
        json.dump(cases, file, ensure_ascii=False, indent=2)

    loaded_cases = load_retrieval_eval_cases(input_file)

    assert loaded_cases == cases


def test_load_retrieval_eval_cases_rejects_duplicate_ids(tmp_path: Path) -> None:
    cases: list[RetrievalEvalCase] = [
        {
            "id": "test_case_001",
            "query": "привезти спорядження додому",
            "expected_document_id": "delivery",
            "category": "delivery",
            "difficulty": "easy",
            "split": "dev",
        },
        {
            "id": "test_case_001",
            "query": "де вас знайти",
            "expected_document_id": "contacts",
            "category": "contacts",
            "difficulty": "easy",
            "split": "dev",
        },
    ]
    input_file = tmp_path / "retrieval_eval_cases.json"

    with open(input_file, "w", encoding="utf-8") as file:
        json.dump(cases, file, ensure_ascii=False, indent=2)

    with pytest.raises(ValueError, match="Duplicate eval case id: test_case_001"):
        load_retrieval_eval_cases(input_file)


def test_load_retrieval_eval_cases_rejects_invalid_category(tmp_path: Path) -> None:
    cases: list[RetrievalEvalCase] = [
        {
            "id": "test_case_001",
            "query": "привезти спорядження додому",
            "expected_document_id": "delivery",
            "category": "wrong_category",
            "difficulty": "easy",
            "split": "dev",
        },
    ]
    input_file = tmp_path / "retrieval_eval_cases.json"

    with open(input_file, "w", encoding="utf-8") as file:
        json.dump(cases, file, ensure_ascii=False, indent=2)

    with pytest.raises(ValueError, match="Unknown eval category: wrong_category"):
        load_retrieval_eval_cases(input_file)


def test_load_retrieval_eval_cases_rejects_invalid_difficulty(tmp_path: Path) -> None:
    cases: list[RetrievalEvalCase] = [
        {
            "id": "test_case_001",
            "query": "привезти спорядження додому",
            "expected_document_id": "delivery",
            "category": "delivery",
            "difficulty": "wrong_difficulty",
            "split": "dev",
        },
    ]
    input_file = tmp_path / "retrieval_eval_cases.json"

    with open(input_file, "w", encoding="utf-8") as file:
        json.dump(cases, file, ensure_ascii=False, indent=2)

    with pytest.raises(ValueError, match="Unknown eval difficulty: wrong_difficulty"):
        load_retrieval_eval_cases(input_file)


def test_load_retrieval_eval_cases_rejects_invalid_split(tmp_path: Path) -> None:
    cases: list[RetrievalEvalCase] = [
        {
            "id": "test_case_001",
            "query": "привезти спорядження додому",
            "expected_document_id": "delivery",
            "category": "delivery",
            "difficulty": "easy",
            "split": "wrong_split",
        },
    ]
    input_file = tmp_path / "retrieval_eval_cases.json"

    with open(input_file, "w", encoding="utf-8") as file:
        json.dump(cases, file, ensure_ascii=False, indent=2)

    with pytest.raises(ValueError, match="Unknown eval split: wrong_split"):
        load_retrieval_eval_cases(input_file)
