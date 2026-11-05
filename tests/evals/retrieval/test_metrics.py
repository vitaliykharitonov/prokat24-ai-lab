from prokat24_ai_lab.evals.retrieval.hybrid import HybridSearchEvalResult
from prokat24_ai_lab.evals.retrieval.metrics import (
    build_grouped_retrieval_eval_summaries,
    build_retrieval_eval_summary,
)

TEST_RETRIEVAL_RESULTS: list[HybridSearchEvalResult] = [
    {
        "id": "test_case_001",
        "query": "чи можна замовити доставку?",
        "expected_document_id": "delivery",
        "actual_document_id": "delivery",
        "actual_score": 0.032,
        "expected_rank": 1,
        "passed": True,
        "category": "delivery",
        "difficulty": "easy",
        "split": "dev",
    },
    {
        "id": "test_case_002",
        "query": "можете привезти спорядження за місто?",
        "expected_document_id": "delivery",
        "actual_document_id": "rental-rules",
        "actual_score": 0.031,
        "expected_rank": 2,
        "passed": False,
        "category": "delivery",
        "difficulty": "medium",
        "split": "holdout",
    },
    {
        "id": "test_case_003",
        "query": "де можна забрати речі?",
        "expected_document_id": "contacts",
        "actual_document_id": "contacts",
        "actual_score": 0.033,
        "expected_rank": 1,
        "passed": True,
        "category": "contacts",
        "difficulty": "hard",
        "split": "holdout",
    },
]


def test_build_retrieval_eval_summary_calculates_metrics():
    summary = build_retrieval_eval_summary(TEST_RETRIEVAL_RESULTS)

    assert summary == {
        "total_cases": 3,
        "passed_cases": 2,
        "accuracy": 2 / 3,
        "top_3_passed_cases": 3,
        "top_3_accuracy": 1.0,
    }


def test_build_retrieval_eval_summary_returns_zeros_for_empty_results() -> None:
    summary = build_retrieval_eval_summary([])

    assert summary == {
        "accuracy": 0.0,
        "passed_cases": 0,
        "top_3_accuracy": 0.0,
        "top_3_passed_cases": 0,
        "total_cases": 0,
    }


def test_build_grouped_retrieval_eval_summaries_groups_by_category() -> None:
    grouped_summary = build_grouped_retrieval_eval_summaries(TEST_RETRIEVAL_RESULTS, "category")

    assert grouped_summary == {
        "delivery": {
            "total_cases": 2,
            "passed_cases": 1,
            "accuracy": 0.5,
            "top_3_passed_cases": 2,
            "top_3_accuracy": 1.0,
        },
        "contacts": {
            "total_cases": 1,
            "passed_cases": 1,
            "accuracy": 1.0,
            "top_3_passed_cases": 1,
            "top_3_accuracy": 1.0,
        },
    }


def test_build_grouped_retrieval_eval_summaries_groups_by_split() -> None:
    grouped_summary = build_grouped_retrieval_eval_summaries(TEST_RETRIEVAL_RESULTS, "split")

    assert grouped_summary == {
        "dev": {
            "total_cases": 1,
            "passed_cases": 1,
            "accuracy": 1.0,
            "top_3_passed_cases": 1,
            "top_3_accuracy": 1.0,
        },
        "holdout": {
            "total_cases": 2,
            "passed_cases": 1,
            "accuracy": 0.5,
            "top_3_passed_cases": 2,
            "top_3_accuracy": 1.0,
        },
    }
