import json
from pathlib import Path

import pytest

from prokat24_ai_lab.evals.embedding_evals import (
    EmbeddingSearchEvalCase,
    EmbeddingSearchEvalResult,
    build_embedding_search_eval_report,
    calculate_accuracy,
    calculate_top_k_accuracy,
    count_passed_results,
    evaluate_embedding_search_case,
    evaluate_precomputed_embedding_search_case,
    load_embedding_search_eval_cases,
    run_embedding_search_evals,
    run_precomputed_embedding_search_evals,
    save_embedding_search_eval_report,
)

TEST_EMBEDDING_EVAL_RESULTS: list[EmbeddingSearchEvalResult] = [
    {
        "query": "привезти додому",
        "expected_document_id": "delivery",
        "actual_document_id": "delivery",
        "actual_score": 1.0,
        "passed": True,
        "top_results": [
            ("delivery", 1.0),
            ("contacts", 0.0),
        ],
    },
    {
        "query": "де вас знайти",
        "expected_document_id": "delivery",
        "actual_document_id": "contacts",
        "actual_score": 1.0,
        "passed": False,
        "top_results": [
            ("contacts", 1.0),
            ("delivery", 0.0),
        ],
    },
    {
        "query": "яка потрібна застава",
        "expected_document_id": "deposit",
        "actual_document_id": "deposit",
        "actual_score": 0.9,
        "passed": True,
        "top_results": [
            ("deposit", 0.9),
            ("delivery", 0.2),
        ],
    },
]

TEST_EMBEDDING_EVAL_REPORT = {
    "report_type": "embedding_search_eval_report",
    "schema_version": 1,
    "created_at": "2026-01-01T00:00:00+00:00",
    "git_commit": "abc1234",
    "git_dirty": False,
    "total_cases": 3,
    "passed_cases": 2,
    "accuracy": 2 / 3,
    "top_3_accuracy": 1.0,
    "results": TEST_EMBEDDING_EVAL_RESULTS,
}


def test_evaluate_embedding_search_case_passes_when_expected_document_is_first() -> None:
    case: EmbeddingSearchEvalCase = {
        "query": "привезти спорядження додому",
        "expected_document_id": "delivery",
    }
    document_embeddings = {
        "delivery": [1.0, 0.0],
        "contacts": [0.0, 1.0],
        "deposit": [-1.0, 0.0],
    }
    query_embeddings = {
        "привезти спорядження додому": [1.0, 0.0],
    }

    def fake_embed_text(query: str) -> list[float]:
        return query_embeddings[query]

    result = evaluate_embedding_search_case(case, document_embeddings, fake_embed_text)

    assert result["actual_document_id"] == "delivery"
    assert result["actual_score"] == 1.0
    assert result["passed"] is True
    assert result["top_results"][0] == ("delivery", 1.0)


def test_evaluate_embedding_search_case_fails_when_another_document_is_first() -> None:
    case: EmbeddingSearchEvalCase = {
        "query": "де вас знайти",
        "expected_document_id": "delivery",
    }
    document_embeddings = {
        "delivery": [0.0, 1.0],
        "contacts": [1.0, 0.0],
        "deposit": [-1.0, 0.0],
    }
    query_embeddings = {
        "де вас знайти": [1.0, 0.0],
    }

    def fake_embed_text(query: str) -> list[float]:
        return query_embeddings[query]

    result = evaluate_embedding_search_case(case, document_embeddings, fake_embed_text)

    assert result["actual_document_id"] == "contacts"
    assert result["actual_score"] == 1.0
    assert result["passed"] is False
    assert result["top_results"][0] == ("contacts", 1.0)


def test_evaluate_embedding_search_case_handles_empty_document_embeddings() -> None:
    case: EmbeddingSearchEvalCase = {
        "query": "привезти спорядження додому",
        "expected_document_id": "delivery",
    }
    document_embeddings: dict[str, list[float]] = {}

    def fail_embed_text(_text: str) -> list[float]:
        raise AssertionError("embed_text should not be called")

    result = evaluate_embedding_search_case(case, document_embeddings, fail_embed_text)

    assert result["actual_document_id"] is None
    assert result["actual_score"] is None
    assert result["passed"] is False
    assert result["top_results"] == []


def test_evaluate_precomputed_embedding_search_case_uses_query_embedding() -> None:
    case = {
        "query": "привезти додому",
        "expected_document_id": "delivery",
    }

    query_embedding = [1.0, 0.0]

    document_embeddings = {
        "delivery": [1.0, 0.0],
        "contacts": [0.0, 1.0],
        "deposit": [-1.0, 0.0],
    }

    result = evaluate_precomputed_embedding_search_case(case, document_embeddings, query_embedding)

    assert result["actual_document_id"] == "delivery"
    assert result["actual_score"] == pytest.approx(1.0)
    assert result["passed"] is True
    assert len(result["top_results"]) == 3


def test_run_embedding_search_evals_returns_results_for_all_cases() -> None:
    cases: list[EmbeddingSearchEvalCase] = [
        {
            "query": "привезти додому",
            "expected_document_id": "delivery",
        },
        {
            "query": "де вас знайти",
            "expected_document_id": "delivery",
        },
    ]
    document_embeddings = {
        "delivery": [1.0, 0.0],
        "contacts": [0.0, 1.0],
        "deposit": [-1.0, 0.0],
    }
    query_embeddings = {
        "привезти додому": [1.0, 0.0],
        "де вас знайти": [0.0, 1.0],
    }

    def fake_embed_text(query: str) -> list[float]:
        return query_embeddings[query]

    results = run_embedding_search_evals(cases, document_embeddings, fake_embed_text)

    assert len(results) == 2
    assert results[0]["actual_document_id"] == "delivery"
    assert results[0]["passed"] is True
    assert results[1]["actual_document_id"] == "contacts"
    assert results[1]["passed"] is False


def test_run_precomputed_embedding_search_evals_evaluates_all_cases() -> None:
    cases: list[EmbeddingSearchEvalCase] = [
        {
            "query": "привезти додому",
            "expected_document_id": "delivery",
        },
        {
            "query": "де вас знайти",
            "expected_document_id": "delivery",
        },
    ]
    document_embeddings = {
        "delivery": [1.0, 0.0],
        "contacts": [0.0, 1.0],
        "deposit": [-1.0, 0.0],
    }
    query_embeddings = {
        "привезти додому": [1.0, 0.0],
        "де вас знайти": [0.0, 1.0],
    }

    results = run_precomputed_embedding_search_evals(cases, document_embeddings, query_embeddings)

    assert len(results) == 2
    assert results[0]["actual_document_id"] == "delivery"
    assert results[0]["passed"] is True
    assert results[1]["actual_document_id"] == "contacts"
    assert results[1]["passed"] is False


def test_run_precomputed_embedding_search_evals_raises_for_missing_query_embedding() -> None:
    cases: list[EmbeddingSearchEvalCase] = [
        {
            "query": "привезти додому",
            "expected_document_id": "delivery",
        },
    ]
    document_embeddings = {
        "delivery": [1.0, 0.0],
        "contacts": [0.0, 1.0],
        "deposit": [-1.0, 0.0],
    }
    query_embeddings = {
        "де вас знайти": [0.0, 1.0],
    }

    with pytest.raises(ValueError, match="Query embedding not found: привезти додому"):
        run_precomputed_embedding_search_evals(cases, document_embeddings, query_embeddings)


def test_count_passed_results_returns_number_of_passed_cases() -> None:
    count_passed = count_passed_results(TEST_EMBEDDING_EVAL_RESULTS)

    assert count_passed == 2


def test_calculate_accuracy_returns_passed_ratio() -> None:
    accuracy = calculate_accuracy(TEST_EMBEDDING_EVAL_RESULTS)

    assert accuracy == pytest.approx(2 / 3)


def test_calculate_accuracy_returns_zero_for_empty_results() -> None:
    accuracy = calculate_accuracy([])

    assert accuracy == 0.0


def test_calculate_top_k_accuracy_checks_first_result() -> None:
    top_k_accuracy = calculate_top_k_accuracy(TEST_EMBEDDING_EVAL_RESULTS, top_k=1)

    assert top_k_accuracy == pytest.approx(2 / 3)


def test_calculate_top_k_accuracy_includes_results_within_k() -> None:
    top_k_accuracy = calculate_top_k_accuracy(TEST_EMBEDDING_EVAL_RESULTS, top_k=2)

    assert top_k_accuracy == 1.0


def test_calculate_top_k_accuracy_returns_zero_for_empty_results() -> None:
    top_k_accuracy = calculate_top_k_accuracy([], top_k=3)

    assert top_k_accuracy == 0.0


def test_build_embedding_search_eval_report_returns_summary() -> None:
    report = build_embedding_search_eval_report(
        TEST_EMBEDDING_EVAL_RESULTS,
        created_at="2026-01-01T00:00:00+00:00",
        git_commit="abc1234",
        git_dirty=False,
    )

    assert report == TEST_EMBEDDING_EVAL_REPORT


def test_save_embedding_search_eval_report_writes_json(tmp_path: Path) -> None:
    output_file = tmp_path / "reports" / "embedding_eval_report.json"

    save_embedding_search_eval_report(TEST_EMBEDDING_EVAL_REPORT, output_file)

    with open(output_file, encoding="utf-8") as file:
        loaded_report = json.load(file)

    assert loaded_report == json.loads(json.dumps(TEST_EMBEDDING_EVAL_REPORT, ensure_ascii=False))


def test_load_embedding_search_eval_cases_returns_cases_from_json(tmp_path: Path) -> None:
    cases = [
        {
            "query": "привезти спорядження додому",
            "expected_document_id": "delivery",
        },
        {
            "query": "де вас знайти",
            "expected_document_id": "contacts",
        },
    ]
    input_file = tmp_path / "embedding_search_eval_cases.json"

    with open(input_file, "w", encoding="utf-8") as file:
        json.dump(cases, file, ensure_ascii=False, indent=2)

    loaded_cases = load_embedding_search_eval_cases(input_file)

    assert loaded_cases == cases
