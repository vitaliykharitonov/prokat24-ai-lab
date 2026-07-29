import json
from pathlib import Path

import pytest

from prokat24_ai_lab.evals.embedding_evals import EmbeddingSearchEvalCase
from prokat24_ai_lab.evals.hybrid_search_evals import (
    HybridSearchEvalReport,
    HybridSearchEvalResult,
    build_hybrid_search_eval_report,
    build_hybrid_search_eval_summary,
    evaluate_hybrid_search_case,
    run_hybrid_search_evals,
    save_hybrid_search_eval_report,
)
from prokat24_ai_lab.knowledge.base import KnowledgePage


@pytest.fixture
def knowledge_pages() -> list[KnowledgePage]:
    return [
        {
            "url": "dostavka",
            "title": "Доставка спорядження",
            "text": "Доставка спорядження по Києву. Умови та вартість доставки.",
            "text_length": 61,
        },
        {
            "url": "contact-us",
            "title": "Контакти",
            "text": "Зателефонуйте нам, щоб отримати спорядження.",
            "text_length": 47,
        },
        {
            "url": "pravila-prokata",
            "title": "Правила прокату",
            "text": "Строк оренди, застава та повернення речей.",
            "text_length": 44,
        },
    ]


@pytest.fixture
def results() -> list[HybridSearchEvalResult]:
    return [
        {
            "query": "очікуваний документ на першому місці",
            "expected_document_id": "dostavka",
            "actual_document_id": "dostavka",
            "actual_score": 0.032,
            "expected_rank": 1,
            "passed": True,
        },
        {
            "query": "очікуваний документ на другому місці",
            "expected_document_id": "contact-us",
            "actual_document_id": "dostavka",
            "actual_score": 0.031,
            "expected_rank": 2,
            "passed": False,
        },
        {
            "query": "очікуваний документ на четвертому місці",
            "expected_document_id": "pravila-prokata",
            "actual_document_id": "contact-us",
            "actual_score": 0.030,
            "expected_rank": 4,
            "passed": False,
        },
        {
            "query": "очікуваний документ не знайдено",
            "expected_document_id": "o-nas",
            "actual_document_id": None,
            "actual_score": None,
            "expected_rank": None,
            "passed": False,
        },
    ]


def test_evaluate_hybrid_search_case_returns_passed_result(
    knowledge_pages: list[KnowledgePage],
) -> None:
    case: EmbeddingSearchEvalCase = {
        "query": "де вас знайти",
        "expected_document_id": "contact-us",
    }
    document_embeddings = {
        "dostavka": [1.0, 0.0],
        "contact-us": [0.0, 1.0],
        "pravila-prokata": [0.8, 0.2],
    }
    query_embedding = [1.0, 0.0]

    result = evaluate_hybrid_search_case(
        case,
        knowledge_pages,
        document_embeddings,
        query_embedding,
    )

    assert result["query"] == case["query"]
    assert result["expected_document_id"] == case["expected_document_id"]
    assert result["actual_document_id"] == case["expected_document_id"]
    assert result["actual_score"] is not None
    assert result["expected_rank"] == 1
    assert result["passed"] is True


def test_evaluate_hybrid_search_case_returns_expected_document_rank(
    knowledge_pages: list[KnowledgePage],
) -> None:
    case: EmbeddingSearchEvalCase = {
        "query": "доставка спорядження",
        "expected_document_id": "contact-us",
    }
    document_embeddings = {
        "contact-us": [1.0, 0.0],
        "dostavka": [0.9, 0.1],
        "pravila-prokata": [0.0, 1.0],
    }
    query_embedding = [1.0, 0.0]

    result = evaluate_hybrid_search_case(
        case,
        knowledge_pages,
        document_embeddings,
        query_embedding,
    )

    assert result["actual_document_id"] == "dostavka"
    assert result["expected_rank"] == 2
    assert result["passed"] is False


def test_evaluate_hybrid_search_case_returns_empty_result() -> None:
    case: EmbeddingSearchEvalCase = {
        "query": "невідомий запит",
        "expected_document_id": "missing-document",
    }

    result = evaluate_hybrid_search_case(
        case,
        [],
        {},
        [1.0, 0.0],
    )

    assert result["actual_document_id"] is None
    assert result["actual_score"] is None
    assert result["expected_rank"] is None
    assert result["passed"] is False


def test_run_hybrid_search_evals_returns_results_for_all_cases(
    knowledge_pages: list[KnowledgePage],
) -> None:
    cases: list[EmbeddingSearchEvalCase] = [
        {
            "query": "доставка спорядження",
            "expected_document_id": "dostavka",
        },
        {
            "query": "контактний телефон",
            "expected_document_id": "contact-us",
        },
    ]
    document_embeddings = {
        "dostavka": [1.0, 0.0, 0.0],
        "contact-us": [0.0, 1.0, 0.0],
        "pravila-prokata": [0.0, 0.0, 1.0],
    }
    query_embeddings = {
        "доставка спорядження": [1.0, 0.0, 0.0],
        "контактний телефон": [0.0, 1.0, 0.0],
    }

    results = run_hybrid_search_evals(
        cases,
        knowledge_pages,
        document_embeddings,
        query_embeddings,
    )

    assert len(results) == 2

    assert results[0]["query"] == "доставка спорядження"
    assert results[0]["actual_document_id"] == "dostavka"
    assert results[0]["passed"] is True

    assert results[1]["query"] == "контактний телефон"
    assert results[1]["actual_document_id"] == "contact-us"
    assert results[1]["passed"] is True


def test_run_hybrid_search_evals_raises_error_for_missing_query_embedding() -> None:
    cases: list[EmbeddingSearchEvalCase] = [
        {
            "query": "доставка спорядження",
            "expected_document_id": "dostavka",
        },
    ]

    with pytest.raises(ValueError, match="Query embedding not found: доставка спорядження"):
        run_hybrid_search_evals(
            cases,
            [],
            {},
            {},
        )


def test_build_hybrid_search_eval_summary_calculates_metrics(results) -> None:
    summary = build_hybrid_search_eval_summary(results)

    assert summary["total_cases"] == 4
    assert summary["passed_cases"] == 1
    assert summary["accuracy"] == 0.25
    assert summary["top_3_passed_cases"] == 2
    assert summary["top_3_accuracy"] == 0.5


def test_build_hybrid_search_eval_summary_returns_zeros_for_empty_results() -> None:
    results: list[HybridSearchEvalResult] = []

    summary = build_hybrid_search_eval_summary(results)

    assert summary["total_cases"] == 0
    assert summary["passed_cases"] == 0
    assert summary["accuracy"] == 0.0
    assert summary["top_3_passed_cases"] == 0
    assert summary["top_3_accuracy"] == 0.0


def test_build_hybrid_search_eval_report_includes_metadata_and_metrics(
    results: list[HybridSearchEvalResult],
) -> None:
    report = build_hybrid_search_eval_report(
        results,
        created_at="2026-01-01T00:00:00+00:00",
        rank_constant=60,
        git_commit="abc1234",
        git_dirty=False,
    )

    assert report["report_type"] == "hybrid_search_eval_report"
    assert report["schema_version"] == 1
    assert report["created_at"] == "2026-01-01T00:00:00+00:00"
    assert report["git_commit"] == "abc1234"
    assert report["git_dirty"] is False
    assert report["rank_constant"] == 60
    assert report["total_cases"] == 4
    assert report["passed_cases"] == 1
    assert report["accuracy"] == 0.25
    assert report["top_3_passed_cases"] == 2
    assert report["top_3_accuracy"] == 0.5
    assert report["results"] == results


def test_save_hybrid_search_eval_report_writes_json_file(tmp_path: Path) -> None:
    report: HybridSearchEvalReport = {
        "report_type": "hybrid_search_eval_report",
        "schema_version": 1,
        "created_at": "2026-07-29T18:00:00+00:00",
        "git_commit": "abc123",
        "git_dirty": False,
        "rank_constant": 60,
        "total_cases": 1,
        "passed_cases": 1,
        "accuracy": 1.0,
        "top_3_passed_cases": 1,
        "top_3_accuracy": 1.0,
        "results": [
            {
                "query": "чи можна замовити доставку?",
                "expected_document_id": "dostavka",
                "actual_document_id": "dostavka",
                "actual_score": 0.03278688524590164,
                "expected_rank": 1,
                "passed": True,
            },
        ],
    }

    output_file = tmp_path / "reports" / "hybrid_search_eval_report.json"

    save_hybrid_search_eval_report(report, output_file)

    with open(output_file, encoding="utf-8") as file:
        loaded_report = json.load(file)

    assert output_file.exists()
    assert loaded_report == report
