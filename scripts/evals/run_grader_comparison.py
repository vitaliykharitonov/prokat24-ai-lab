import datetime
from argparse import ArgumentParser, Namespace

from openai import OpenAI

from prokat24_ai_lab.evals.grading.comparison import (
    GraderComparisonResult,
    build_grader_confusion_matrix,
    calculate_grader_classification_metrics,
    summarize_grader_comparison,
    summarize_grader_comparison_by_category,
    summarize_grader_stability,
)
from prokat24_ai_lab.evals.grading.comparison_runner import (
    load_grader_comparison_cases,
    run_grader_comparison_evals,
)
from prokat24_ai_lab.evals.grading.report import (
    build_grader_comparison_report,
    save_grader_comparison_report,
)
from prokat24_ai_lab.git_metadata import get_current_git_commit, is_git_worktree_dirty
from prokat24_ai_lab.judges.openai_judge import create_openai_criterion_judge, create_openai_judge
from prokat24_ai_lab.project_paths import (
    GRADER_COMPARISON_CASES_FILE,
    GRADER_COMPARISON_HOLDOUT_CASES_FILE,
    GRADER_COMPARISON_REPORTS_DIR,
)

RUNS = 3


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Compare graders on dev or holdout dataset")
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
    client = OpenAI()
    judge = create_openai_judge(client)
    criterion_judge = create_openai_criterion_judge(client)

    all_runs: list[list[GraderComparisonResult]] = []

    for _ in range(RUNS):
        results = run_grader_comparison_evals(cases, judge, criterion_judge)
        all_runs.append(results)

    run_time = datetime.datetime.now(datetime.UTC)

    created_at = run_time.isoformat(timespec="seconds")
    run_timestamp = run_time.strftime("%Y%m%dT%H%M%SZ")

    git_commit = get_current_git_commit()
    git_dirty = is_git_worktree_dirty()

    short_commit = git_commit[:7] if git_commit is not None else "no-git"
    run_id = f"{run_timestamp}_{short_commit}"

    report = build_grader_comparison_report(
        all_runs=all_runs,
        split=args.split,
        run_id=run_id,
        created_at=created_at,
        git_commit=git_commit,
        git_dirty=git_dirty,
    )

    historical_file, latest_file = save_grader_comparison_report(
        report=report,
        output_dir=GRADER_COMPARISON_REPORTS_DIR,
    )

    stability_results = summarize_grader_stability(all_runs)

    summary = summarize_grader_comparison(all_runs[0])
    summaries = summarize_grader_comparison_by_category(all_runs[0])

    total_cases = summary["total_cases"]

    deterministic_matrix = build_grader_confusion_matrix(
        all_runs[0],
        "deterministic_passed",
    )

    llm_matrix = build_grader_confusion_matrix(
        all_runs[0],
        "llm_passed",
    )

    criterion_llm_matrix = build_grader_confusion_matrix(
        all_runs[0],
        "criterion_llm_passed",
    )

    deterministic_metrics = calculate_grader_classification_metrics(deterministic_matrix)
    llm_metrics = calculate_grader_classification_metrics(llm_matrix)
    criterion_llm_metrics = calculate_grader_classification_metrics(criterion_llm_matrix)

    print(f"\nDataset: {args.split}")
    print("\nGRADER COMPARISON\n")
    print(f"Cases: {total_cases}\n")
    print("Overall:")
    print(
        f"Deterministic: {summary['deterministic_matched']}/{total_cases} "
        f"({summary['deterministic_accuracy']:.2%})"
    )
    print(f"LLM judge: {summary['llm_matched']}/{total_cases} ({summary['llm_accuracy']:.2%})")
    print(
        f"Criterion LLM: {summary['criterion_llm_matched']}/{total_cases} "
        f"({summary['criterion_llm_accuracy']:.2%})"
    )
    print("\nBy category:\n")

    for category in summaries:
        category_total = summaries[category]["total_cases"]
        print(f"{category}:")
        print(
            f"Deterministic: {summaries[category]['deterministic_matched']}/{category_total} "
            f"({summaries[category]['deterministic_accuracy']:.2%})"
        )
        print(
            f"LLM judge: {summaries[category]['llm_matched']}/{category_total} "
            f"({summaries[category]['llm_accuracy']:.2%})"
        )
        print(
            f"Criterion LLM: {summaries[category]['criterion_llm_matched']}/{category_total} "
            f"({summaries[category]['criterion_llm_accuracy']:.2%})"
        )
        print()

    print("Mismatches:\n")

    for result in all_runs[0]:
        if not result["deterministic_matched"]:
            print(f"ID: {result['id']} [{result['category']}]")
            print(f"Expected: {result['expected_passed']}")
            print(f"Deterministic: {result['deterministic_passed']}")
            print()

        if not result["llm_matched"]:
            print(f"ID: {result['id']} [{result['category']}]")
            print(f"Expected: {result['expected_passed']}")
            print(f"LLM judge: {result['llm_passed']}")
            print(f"Score: {result['llm_score']}")
            print(f"Reason: {result['llm_reason']}")
            print(f"Failed criteria: {result['llm_failed_criteria']}")
            print()

        if not result["criterion_llm_matched"]:
            print(f"ID: {result['id']} [{result['category']}]")
            print(f"Expected: {result['expected_passed']}")
            print(f"Criterion LLM: {result['criterion_llm_passed']}")
            print(f"Failed criteria: {result['criterion_llm_failed_criteria']}")
            print()

    print("Confusion matrices:\n")
    print("Deterministic:")
    print("                 Human")
    print("             PASS     FAIL")
    print(
        f"Grader PASS  {deterministic_matrix['true_positive']:>4} "
        f"{deterministic_matrix['false_positive']:>8}"
    )
    print(
        f"Grader FAIL  {deterministic_matrix['false_negative']:>4} "
        f"{deterministic_matrix['true_negative']:>8}"
    )
    print()
    print("LLM judge:")
    print("                 Human")
    print("             PASS     FAIL")
    print(f"Grader PASS  {llm_matrix['true_positive']:>4} {llm_matrix['false_positive']:>8}")
    print(f"Grader FAIL  {llm_matrix['false_negative']:>4} {llm_matrix['true_negative']:>8}")

    print()
    print("Criterion LLM:")
    print("                 Human")
    print("             PASS     FAIL")
    print(
        f"Grader PASS  {criterion_llm_matrix['true_positive']:>4} "
        f"{criterion_llm_matrix['false_positive']:>8}"
    )
    print(
        f"Grader FAIL  {criterion_llm_matrix['false_negative']:>4} "
        f"{criterion_llm_matrix['true_negative']:>8}"
    )

    print()
    print("Classification metrics:\n")
    print("Deterministic:")
    print(f"Precision: {deterministic_metrics['precision']:.2%}")
    print(f"Recall: {deterministic_metrics['recall']:.2%}")
    print()
    print("LLM judge:")
    print(f"Precision: {llm_metrics['precision']:.2%}")
    print(f"Recall: {llm_metrics['recall']:.2%}")
    print()
    print("Criterion LLM:")
    print(f"Precision: {criterion_llm_metrics['precision']:.2%}")
    print(f"Recall: {criterion_llm_metrics['recall']:.2%}")

    print()
    print("LLM judge stability:\n")

    header = f"{'ID':<36} {'Expected':<10}"

    for run_number in range(1, RUNS + 1):
        header += f"{f'LLM R{run_number}':<8}"

    header += f"{'Stable':<8} {'Match':<10}"

    for run_number in range(1, RUNS + 1):
        header += f"{f'Crit R{run_number}':<9}"

    header += f"{'Stable':<8} Match"

    print(header)

    for result in stability_results:
        expected = "PASS" if result["expected_passed"] else "FAIL"
        llm_stable = "YES" if result["llm_stable"] else "NO"
        criterion_llm_stable = "YES" if result["criterion_llm_stable"] else "NO"

        row = f"{result['id']:<36} {expected:<10}"

        for passed in result["llm_passed_values"]:
            verdict = "PASS" if passed else "FAIL"
            row += f"{verdict:<8}"

        row += f"{llm_stable:<8} {result['llm_match_rate']:<9.2%}"

        for passed in result["criterion_llm_passed_values"]:
            verdict = "PASS" if passed else "FAIL"
            row += f"{verdict:<9}"

        row += f"{criterion_llm_stable:<8} {result['criterion_llm_match_rate']:.2%}"

        print(row)

    print()
    print(f"Historical report: {historical_file}")
    print(f"Latest report: {latest_file}")


if __name__ == "__main__":
    main()
