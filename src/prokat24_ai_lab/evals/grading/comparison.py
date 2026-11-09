from typing import Literal, TypedDict


class GraderComparisonResult(TypedDict):
    id: str
    category: str
    expected_passed: bool
    deterministic_passed: bool
    deterministic_matched: bool
    llm_passed: bool
    llm_matched: bool
    llm_score: int
    llm_reason: str
    llm_failed_criteria: list[str]
    criterion_llm_passed: bool
    criterion_llm_matched: bool
    criterion_llm_failed_criteria: list[str]


class GraderComparisonSummary(TypedDict):
    total_cases: int
    deterministic_matched: int
    llm_matched: int
    criterion_llm_matched: int
    deterministic_accuracy: float
    llm_accuracy: float
    criterion_llm_accuracy: float


class GraderCategorySummary(TypedDict):
    total_cases: int
    deterministic_matched: int
    llm_matched: int
    criterion_llm_matched: int
    deterministic_accuracy: float
    llm_accuracy: float
    criterion_llm_accuracy: float


class GraderConfusionMatrix(TypedDict):
    true_positive: int
    true_negative: int
    false_positive: int
    false_negative: int


class GraderClassificationMetrics(TypedDict):
    precision: float
    recall: float


class GraderStabilityResult(TypedDict):
    id: str
    expected_passed: bool
    llm_passed_values: list[bool]
    llm_stable: bool
    llm_match_rate: float
    criterion_llm_passed_values: list[bool]
    criterion_llm_stable: bool
    criterion_llm_match_rate: float


GraderKey = Literal["deterministic_passed", "llm_passed", "criterion_llm_passed"]


def summarize_grader_comparison(
    results: list[GraderComparisonResult],
) -> GraderComparisonSummary:
    total_cases = len(results)
    deterministic_matched = 0
    llm_matched = 0
    criterion_llm_matched = 0

    if not results:
        return {
            "total_cases": 0,
            "deterministic_matched": 0,
            "llm_matched": 0,
            "criterion_llm_matched": 0,
            "deterministic_accuracy": 0.0,
            "llm_accuracy": 0.0,
            "criterion_llm_accuracy": 0.0,
        }

    for result in results:
        if result["deterministic_matched"]:
            deterministic_matched += 1

        if result["llm_matched"]:
            llm_matched += 1

        if result["criterion_llm_matched"]:
            criterion_llm_matched += 1

    deterministic_accuracy = deterministic_matched / total_cases
    llm_accuracy = llm_matched / total_cases
    criterion_llm_accuracy = criterion_llm_matched / total_cases

    return {
        "total_cases": total_cases,
        "deterministic_matched": deterministic_matched,
        "llm_matched": llm_matched,
        "criterion_llm_matched": criterion_llm_matched,
        "deterministic_accuracy": deterministic_accuracy,
        "llm_accuracy": llm_accuracy,
        "criterion_llm_accuracy": criterion_llm_accuracy,
    }


def summarize_grader_comparison_by_category(
    results: list[GraderComparisonResult],
) -> dict[str, GraderCategorySummary]:
    grouped_results: dict[str, list[GraderComparisonResult]] = {}
    summaries: dict[str, GraderCategorySummary] = {}

    for result in results:
        if result["category"] not in grouped_results:
            grouped_results[result["category"]] = []

        grouped_results[result["category"]].append(result)

    for group in grouped_results:
        summaries[group] = summarize_grader_comparison(grouped_results[group])

    return summaries


def build_grader_confusion_matrix(
    results: list[GraderComparisonResult],
    grader_key: GraderKey,
) -> GraderConfusionMatrix:
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    for result in results:
        expected = result["expected_passed"]
        actual = result[grader_key]

        if expected is True and actual is True:
            true_positive += 1
        elif expected is False and actual is False:
            true_negative += 1
        elif expected is False and actual is True:
            false_positive += 1
        else:
            false_negative += 1

    return {
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
    }


def calculate_grader_classification_metrics(
    matrix: GraderConfusionMatrix,
) -> GraderClassificationMetrics:
    true_positive = matrix["true_positive"]
    false_positive = matrix["false_positive"]
    false_negative = matrix["false_negative"]

    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative

    precision = true_positive / precision_denominator if precision_denominator else 0.0
    recall = true_positive / recall_denominator if recall_denominator else 0.0

    return {
        "precision": precision,
        "recall": recall,
    }


def summarize_grader_stability(
    runs: list[list[GraderComparisonResult]],
) -> list[GraderStabilityResult]:
    stability_results: list[GraderStabilityResult] = []

    if not runs:
        return stability_results

    for first_run_case in runs[0]:
        case_id = first_run_case["id"]
        llm_passed_values: list[bool] = []
        criterion_llm_passed_values: list[bool] = []

        for run in runs:
            for case in run:
                if case["id"] == case_id:
                    llm_passed_values.append(case["llm_passed"])
                    criterion_llm_passed_values.append(case["criterion_llm_passed"])

        llm_matched = 0
        criterion_llm_matched = 0

        for value in llm_passed_values:
            if value == first_run_case["expected_passed"]:
                llm_matched += 1

        for value in criterion_llm_passed_values:
            if value == first_run_case["expected_passed"]:
                criterion_llm_matched += 1

        llm_match_rate = llm_matched / len(llm_passed_values)
        criterion_llm_match_rate = criterion_llm_matched / len(criterion_llm_passed_values)

        llm_stable = len(set(llm_passed_values)) == 1
        criterion_stable = len(set(criterion_llm_passed_values)) == 1

        stability_results.append(
            {
                "id": case_id,
                "expected_passed": first_run_case["expected_passed"],
                "llm_passed_values": llm_passed_values,
                "llm_stable": llm_stable,
                "llm_match_rate": llm_match_rate,
                "criterion_llm_passed_values": criterion_llm_passed_values,
                "criterion_llm_stable": criterion_stable,
                "criterion_llm_match_rate": criterion_llm_match_rate,
            }
        )

    return stability_results
