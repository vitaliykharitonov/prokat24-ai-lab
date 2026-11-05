from collections.abc import Sequence
from typing import Literal, TypedDict

from prokat24_ai_lab.evals.retrieval.embedding import EmbeddingSearchEvalResult
from prokat24_ai_lab.evals.retrieval.hybrid import HybridSearchEvalResult
from prokat24_ai_lab.evals.retrieval.keyword import SearchEvalResult

RetrievalEvalResult = SearchEvalResult | EmbeddingSearchEvalResult | HybridSearchEvalResult
RetrievalGroupBy = Literal["split", "category", "difficulty"]


class RetrievalEvalSummary(TypedDict):
    total_cases: int
    passed_cases: int
    accuracy: float
    top_3_passed_cases: int
    top_3_accuracy: float


def build_retrieval_eval_summary(
    results: Sequence[RetrievalEvalResult],
) -> RetrievalEvalSummary:
    total_cases = len(results)
    passed_cases = 0
    top_3_passed_cases = 0

    if not results:
        return {
            "total_cases": 0,
            "passed_cases": 0,
            "accuracy": 0.0,
            "top_3_passed_cases": 0,
            "top_3_accuracy": 0.0,
        }

    for result in results:
        if result["passed"]:
            passed_cases += 1

        if result["expected_rank"] is not None and result["expected_rank"] <= 3:
            top_3_passed_cases += 1

    accuracy = passed_cases / total_cases
    top_3_accuracy = top_3_passed_cases / total_cases

    return {
        "total_cases": total_cases,
        "passed_cases": passed_cases,
        "accuracy": accuracy,
        "top_3_passed_cases": top_3_passed_cases,
        "top_3_accuracy": top_3_accuracy,
    }


def build_grouped_retrieval_eval_summaries(
    results: Sequence[RetrievalEvalResult],
    group_by: RetrievalGroupBy,
) -> dict[str, RetrievalEvalSummary]:
    grouped_results: dict[str, list[RetrievalEvalResult]] = {}
    summaries: dict[str, RetrievalEvalSummary] = {}

    for result in results:
        group_value: str

        if group_by == "split":
            group_value = result["split"]
        elif group_by == "category":
            group_value = result["category"]
        elif group_by == "difficulty":
            group_value = result["difficulty"]
        else:
            raise ValueError("Unknown group")

        grouped_results.setdefault(group_value, []).append(result)

    for group in grouped_results:
        summaries[group] = build_retrieval_eval_summary(grouped_results[group])

    return summaries
