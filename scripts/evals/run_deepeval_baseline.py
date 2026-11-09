from argparse import ArgumentParser, Namespace
from typing import TypedDict

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, SingleTurnParams

from prokat24_ai_lab.evals.grading.comparison_runner import (
    GraderComparisonCase,
    load_grader_comparison_cases,
)
from prokat24_ai_lab.evals.grading.judge import RubricCriterion
from prokat24_ai_lab.project_paths import (
    GRADER_COMPARISON_CASES_FILE,
    GRADER_COMPARISON_HOLDOUT_CASES_FILE,
)


class DeepEvalCriterionResult(TypedDict):
    name: str
    score: float
    passed: bool
    reason: str


class DeepEvalComparisonResult(TypedDict):
    id: str
    category: str
    expected_passed: bool
    actual_passed: bool
    matched: bool
    criteria: list[DeepEvalCriterionResult]


class DeepEvalComparisonSummary(TypedDict):
    total_cases: int
    matched: int
    accuracy: float


class DeepEvalCategorySummary(TypedDict):
    total_cases: int
    matched: int
    accuracy: float


THRESHOLDS = {
    "correctness": 0.5,
    "groundedness": 0.9,
    "relevance": 0.9,
    "completeness": 0.5,
}


def create_deepeval_metric(criterion: RubricCriterion) -> GEval:
    threshold = THRESHOLDS[criterion["name"]]

    evaluation_steps = [
        f"Evaluate whether the actual output satisfies this criterion: {criterion['description']}",
        "Evaluate the actual output specifically for the user's input "
        "using only the provided context.",
        "Penalize any violation of this criterion even if the rest of the answer is correct.",
    ]

    if criterion["name"] == "relevance":
        evaluation_steps.extend(
            [
                "Do not treat information as relevant merely because it is related "
                "to the same topic.",
                "Ask whether each piece of information is necessary to answer "
                "the specific user question.",
                "If the answer would remain complete and correct after removing "
                "a piece of information, treat that information as irrelevant.",
            ]
        )

    return GEval(
        name=criterion["name"].title(),
        evaluation_steps=evaluation_steps,
        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.CONTEXT,
        ],
        threshold=threshold,
        model="gpt-5.4",
    )


def evaluate_deepeval_case(case: GraderComparisonCase) -> DeepEvalComparisonResult:
    test_case = LLMTestCase(
        input=case["judge_input"]["question"],
        actual_output=case["judge_input"]["answer"],
        context=[case["judge_input"]["context"]],
    )

    metrics: list[GEval] = []

    for criterion in case["rubric"]["criteria"]:
        metrics.append(create_deepeval_metric(criterion))

    for metric in metrics:
        metric.measure(test_case)

    overall_passed = all(metric.is_successful() for metric in metrics)

    matched = overall_passed == case["expected_passed"]

    criteria: list[DeepEvalCriterionResult] = []

    for metric in metrics:
        result = DeepEvalCriterionResult(
            name=metric.name,
            score=metric.score,
            passed=metric.is_successful(),
            reason=metric.reason,
        )
        criteria.append(result)

    return DeepEvalComparisonResult(
        id=case["id"],
        category=case["category"],
        expected_passed=case["expected_passed"],
        actual_passed=overall_passed,
        matched=matched,
        criteria=criteria,
    )


def run_deepeval_comparison_evals(
    cases: list[GraderComparisonCase],
) -> list[DeepEvalComparisonResult]:
    results: list[DeepEvalComparisonResult] = []

    for case in cases:
        results.append(evaluate_deepeval_case(case))

    return results


def summarize_deepeval_comparison(
    results: list[DeepEvalComparisonResult],
) -> DeepEvalComparisonSummary:
    total_cases = len(results)

    if total_cases == 0:
        return DeepEvalComparisonSummary(
            total_cases=0,
            matched=0,
            accuracy=0.0,
        )

    matched = 0

    for result in results:
        if result["matched"]:
            matched += 1

    accuracy = matched / total_cases

    return DeepEvalComparisonSummary(
        total_cases=total_cases,
        matched=matched,
        accuracy=accuracy,
    )


def summarize_deepeval_comparison_by_category(
    results: list[DeepEvalComparisonResult],
) -> dict[str, DeepEvalCategorySummary]:
    grouped_results: dict[str, list[DeepEvalComparisonResult]] = {}
    summaries: dict[str, DeepEvalCategorySummary] = {}

    for result in results:
        if result["category"] not in grouped_results:
            grouped_results[result["category"]] = []

        grouped_results[result["category"]].append(result)

    for category in grouped_results:
        summaries[category] = summarize_deepeval_comparison(grouped_results[category])

    return summaries


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Run DeepEval grader comparison on dev or holdout dataset")
    parser.add_argument(
        "--split",
        help="Dataset split to evaluate",
        choices=("dev", "holdout"),
        default="dev",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases_file = GRADER_COMPARISON_CASES_FILE

    if args.split == "holdout":
        cases_file = GRADER_COMPARISON_HOLDOUT_CASES_FILE

    cases = load_grader_comparison_cases(cases_file)

    results = run_deepeval_comparison_evals(cases)

    summary = summarize_deepeval_comparison(results)

    summary_by_category = summarize_deepeval_comparison_by_category(results)

    print(f"\nDataset: {args.split}")
    print("\nDEEPEVAL COMPARISON\n")
    print(f"Cases: {summary['total_cases']}")
    print(f"Matched: {summary['matched']}/{summary['total_cases']} ({summary['accuracy']:.2%})")

    print("\nBy category:")

    for category, category_summary in summary_by_category.items():
        print(f"{category}:")
        print(
            f"{category_summary['matched']}/{category_summary['total_cases']} "
            f"({category_summary['accuracy']:.2%})"
        )
        print()

    print("Mismatches:")

    for result in results:
        if result["matched"]:
            continue

        print(f"ID: {result['id']} [{result['category']}]")
        print(f"Expected: {result['expected_passed']}")
        print(f"DeepEval: {result['actual_passed']}")

        for criterion in result["criteria"]:
            verdict = "PASS" if criterion["passed"] else "FAIL"

            print(f"{criterion['name']}: {verdict}, score={criterion['score']:.2f}")
            print(f"Reason: {criterion['reason']}")

        print()


if __name__ == "__main__":
    main()
