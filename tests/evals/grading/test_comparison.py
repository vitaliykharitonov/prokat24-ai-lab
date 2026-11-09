import pytest

from prokat24_ai_lab.evals.grading.comparison import (
    GraderClassificationMetrics,
    GraderComparisonResult,
    GraderComparisonSummary,
    GraderConfusionMatrix,
    GraderKey,
    GraderStabilityResult,
    build_grader_confusion_matrix,
    calculate_grader_classification_metrics,
    summarize_grader_comparison,
    summarize_grader_comparison_by_category,
    summarize_grader_stability,
)
from tests.evals.grading.fixtures import GRADER_COMPARISON_TEST_RESULTS


def test_summarize_grader_comparison() -> None:
    results: list[GraderComparisonResult] = [
        {
            "id": "case_1",
            "category": "correct",
            "expected_passed": True,
            "deterministic_passed": True,
            "deterministic_matched": True,
            "llm_passed": True,
            "llm_matched": True,
            "llm_score": 5,
            "llm_reason": "Test result",
            "llm_failed_criteria": [],
            "criterion_llm_passed": True,
            "criterion_llm_matched": True,
            "criterion_llm_failed_criteria": [],
        },
        {
            "id": "case_2",
            "category": "paraphrase",
            "expected_passed": True,
            "deterministic_passed": False,
            "deterministic_matched": False,
            "llm_passed": True,
            "llm_matched": True,
            "llm_score": 5,
            "llm_reason": "Test result",
            "llm_failed_criteria": [],
            "criterion_llm_passed": True,
            "criterion_llm_matched": True,
            "criterion_llm_failed_criteria": [],
        },
        {
            "id": "case_3",
            "category": "negation",
            "expected_passed": False,
            "deterministic_passed": True,
            "deterministic_matched": False,
            "llm_passed": False,
            "llm_matched": True,
            "llm_score": 1,
            "llm_reason": "Test result",
            "llm_failed_criteria": ["correctness"],
            "criterion_llm_passed": False,
            "criterion_llm_matched": True,
            "criterion_llm_failed_criteria": ["correctness"],
        },
        {
            "id": "case_4",
            "category": "incorrect",
            "expected_passed": False,
            "deterministic_passed": False,
            "deterministic_matched": True,
            "llm_passed": True,
            "llm_matched": False,
            "llm_score": 5,
            "llm_reason": "Test result",
            "llm_failed_criteria": [],
            "criterion_llm_passed": False,
            "criterion_llm_matched": True,
            "criterion_llm_failed_criteria": ["correctness"],
        },
    ]

    expected_summary: GraderComparisonSummary = {
        "total_cases": 4,
        "deterministic_matched": 2,
        "llm_matched": 3,
        "criterion_llm_matched": 4,
        "deterministic_accuracy": 0.5,
        "llm_accuracy": 0.75,
        "criterion_llm_accuracy": 1.0,
    }

    summary = summarize_grader_comparison(results)

    assert summary == expected_summary


def test_summarize_grader_comparison_empty_results() -> None:
    results: list[GraderComparisonResult] = []

    expected_summary: GraderComparisonSummary = {
        "total_cases": 0,
        "deterministic_matched": 0,
        "llm_matched": 0,
        "criterion_llm_matched": 0,
        "deterministic_accuracy": 0.0,
        "llm_accuracy": 0.0,
        "criterion_llm_accuracy": 0.0,
    }

    summary = summarize_grader_comparison(results)

    assert summary == expected_summary


def test_summarize_grader_comparison_by_category() -> None:
    results: list[GraderComparisonResult] = [
        {
            "id": "case_1",
            "category": "correct",
            "expected_passed": True,
            "deterministic_passed": True,
            "deterministic_matched": True,
            "llm_passed": True,
            "llm_matched": True,
            "llm_score": 5,
            "llm_reason": "Test result",
            "llm_failed_criteria": [],
            "criterion_llm_passed": True,
            "criterion_llm_matched": True,
            "criterion_llm_failed_criteria": [],
        },
        {
            "id": "case_2",
            "category": "paraphrase",
            "expected_passed": True,
            "deterministic_passed": False,
            "deterministic_matched": False,
            "llm_passed": True,
            "llm_matched": True,
            "llm_score": 5,
            "llm_reason": "Test result",
            "llm_failed_criteria": [],
            "criterion_llm_passed": True,
            "criterion_llm_matched": True,
            "criterion_llm_failed_criteria": [],
        },
        {
            "id": "case_3",
            "category": "paraphrase",
            "expected_passed": True,
            "deterministic_passed": True,
            "deterministic_matched": True,
            "llm_passed": False,
            "llm_matched": False,
            "llm_score": 2,
            "llm_reason": "Test result",
            "llm_failed_criteria": ["correctness"],
            "criterion_llm_passed": True,
            "criterion_llm_matched": True,
            "criterion_llm_failed_criteria": [],
        },
        {
            "id": "case_4",
            "category": "negation",
            "expected_passed": False,
            "deterministic_passed": True,
            "deterministic_matched": False,
            "llm_passed": False,
            "llm_matched": True,
            "llm_score": 1,
            "llm_reason": "Test result",
            "llm_failed_criteria": ["correctness"],
            "criterion_llm_passed": False,
            "criterion_llm_matched": True,
            "criterion_llm_failed_criteria": ["correctness"],
        },
    ]

    expected = {
        "correct": {
            "total_cases": 1,
            "deterministic_matched": 1,
            "llm_matched": 1,
            "criterion_llm_matched": 1,
            "deterministic_accuracy": 1.0,
            "llm_accuracy": 1.0,
            "criterion_llm_accuracy": 1.0,
        },
        "paraphrase": {
            "total_cases": 2,
            "deterministic_matched": 1,
            "llm_matched": 1,
            "criterion_llm_matched": 2,
            "deterministic_accuracy": 0.5,
            "llm_accuracy": 0.5,
            "criterion_llm_accuracy": 1.0,
        },
        "negation": {
            "total_cases": 1,
            "deterministic_matched": 0,
            "llm_matched": 1,
            "criterion_llm_matched": 1,
            "deterministic_accuracy": 0.0,
            "llm_accuracy": 1.0,
            "criterion_llm_accuracy": 1.0,
        },
    }

    summaries = summarize_grader_comparison_by_category(results)

    assert summaries == expected


def test_summarize_grader_comparison_by_category_empty_results() -> None:
    results: list[GraderComparisonResult] = []

    summaries = summarize_grader_comparison_by_category(results)

    assert summaries == {}


@pytest.mark.parametrize(
    ("grader_key", "expected_matrix"),
    [
        (
            "deterministic_passed",
            {
                "true_positive": 1,
                "true_negative": 1,
                "false_positive": 1,
                "false_negative": 1,
            },
        ),
        (
            "llm_passed",
            {
                "true_positive": 2,
                "true_negative": 2,
                "false_positive": 0,
                "false_negative": 0,
            },
        ),
        (
            "criterion_llm_passed",
            {
                "true_positive": 2,
                "true_negative": 2,
                "false_positive": 0,
                "false_negative": 0,
            },
        ),
    ],
)
def test_build_grader_confusion_matrix_for_graders(
    grader_key: GraderKey, expected_matrix: GraderConfusionMatrix
) -> None:
    matrix = build_grader_confusion_matrix(
        GRADER_COMPARISON_TEST_RESULTS,
        grader_key,
    )

    assert matrix == expected_matrix


def test_calculate_grader_classification_metrics() -> None:
    matrix: GraderConfusionMatrix = {
        "true_positive": 1,
        "true_negative": 1,
        "false_positive": 4,
        "false_negative": 2,
    }
    expected: GraderClassificationMetrics = {
        "precision": 0.2,
        "recall": 1 / 3,
    }
    metrics = calculate_grader_classification_metrics(matrix)

    assert metrics == expected


def test_calculate_grader_classification_metrics_with_no_positive_cases() -> None:
    matrix: GraderConfusionMatrix = {
        "true_positive": 0,
        "true_negative": 5,
        "false_positive": 0,
        "false_negative": 0,
    }
    expected: GraderClassificationMetrics = {
        "precision": 0.0,
        "recall": 0.0,
    }

    metrics = calculate_grader_classification_metrics(matrix)

    assert metrics == expected


def test_summarize_grader_stability() -> None:
    runs: list[list[GraderComparisonResult]] = [
        [
            {
                "id": "stable_case",
                "category": "correct",
                "expected_passed": True,
                "deterministic_passed": True,
                "deterministic_matched": True,
                "llm_passed": True,
                "llm_matched": True,
                "llm_score": 5,
                "llm_reason": "Correct.",
                "llm_failed_criteria": [],
                "criterion_llm_passed": True,
                "criterion_llm_matched": True,
                "criterion_llm_failed_criteria": [],
            },
            {
                "id": "unstable_case",
                "category": "irrelevant_extra",
                "expected_passed": False,
                "deterministic_passed": True,
                "deterministic_matched": False,
                "llm_passed": True,
                "llm_matched": False,
                "llm_score": 5,
                "llm_reason": "Accepted.",
                "llm_failed_criteria": [],
                "criterion_llm_passed": False,
                "criterion_llm_matched": True,
                "criterion_llm_failed_criteria": ["relevance"],
            },
        ],
        [
            {
                "id": "stable_case",
                "category": "correct",
                "expected_passed": True,
                "deterministic_passed": True,
                "deterministic_matched": True,
                "llm_passed": True,
                "llm_matched": True,
                "llm_score": 5,
                "llm_reason": "Correct.",
                "llm_failed_criteria": [],
                "criterion_llm_passed": True,
                "criterion_llm_matched": True,
                "criterion_llm_failed_criteria": [],
            },
            {
                "id": "unstable_case",
                "category": "irrelevant_extra",
                "expected_passed": False,
                "deterministic_passed": True,
                "deterministic_matched": False,
                "llm_passed": False,
                "llm_matched": True,
                "llm_score": 2,
                "llm_reason": "Contains irrelevant information.",
                "llm_failed_criteria": ["relevance"],
                "criterion_llm_passed": False,
                "criterion_llm_matched": True,
                "criterion_llm_failed_criteria": ["relevance"],
            },
        ],
        [
            {
                "id": "stable_case",
                "category": "correct",
                "expected_passed": True,
                "deterministic_passed": True,
                "deterministic_matched": True,
                "llm_passed": True,
                "llm_matched": True,
                "llm_score": 5,
                "llm_reason": "Correct.",
                "llm_failed_criteria": [],
                "criterion_llm_passed": True,
                "criterion_llm_matched": True,
                "criterion_llm_failed_criteria": [],
            },
            {
                "id": "unstable_case",
                "category": "irrelevant_extra",
                "expected_passed": False,
                "deterministic_passed": True,
                "deterministic_matched": False,
                "llm_passed": False,
                "llm_matched": True,
                "llm_score": 2,
                "llm_reason": "Contains irrelevant information.",
                "llm_failed_criteria": ["relevance"],
                "criterion_llm_passed": False,
                "criterion_llm_matched": True,
                "criterion_llm_failed_criteria": ["relevance"],
            },
        ],
    ]

    expected: list[GraderStabilityResult] = [
        {
            "id": "stable_case",
            "expected_passed": True,
            "llm_passed_values": [True, True, True],
            "llm_stable": True,
            "llm_match_rate": 1.0,
            "criterion_llm_passed_values": [True, True, True],
            "criterion_llm_stable": True,
            "criterion_llm_match_rate": 1.0,
        },
        {
            "id": "unstable_case",
            "expected_passed": False,
            "llm_passed_values": [True, False, False],
            "llm_stable": False,
            "llm_match_rate": 2 / 3,
            "criterion_llm_passed_values": [False, False, False],
            "criterion_llm_stable": True,
            "criterion_llm_match_rate": 1.0,
        },
    ]

    stability_results = summarize_grader_stability(runs)

    assert stability_results == expected


def test_summarize_grader_stability_empty_runs() -> None:
    stability_results = summarize_grader_stability([])

    assert stability_results == []
