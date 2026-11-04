import datetime
from argparse import ArgumentParser, Namespace

from prokat24_ai_lab.evals.retrieval_cases import load_retrieval_eval_cases
from prokat24_ai_lab.evals.retrieval_metrics import (
    RetrievalEvalSummary,
    build_grouped_retrieval_eval_summaries,
)
from prokat24_ai_lab.evals.search_evals import (
    SearchEvalResult,
    build_search_eval_report,
    run_search_evals,
    save_search_eval_report,
)
from prokat24_ai_lab.git_metadata import get_current_git_commit, is_git_worktree_dirty
from prokat24_ai_lab.knowledge.base import load_knowledge_base
from prokat24_ai_lab.project_paths import (
    KNOWLEDGE_PAGES_FILE,
    RETRIEVAL_EVAL_CASES_FILE,
    SEARCH_EVAL_REPORT_FILE,
)

DEV_MIN_ACCURACY = 0.875
DEV_MIN_TOP_3_ACCURACY = 1.0


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--split",
        choices=("all", "dev", "holdout"),
        default="all",
    )

    return parser.parse_args()


def print_failed_search_eval_results(results: list[SearchEvalResult]) -> None:
    failed_results = [result for result in results if not result["passed"]]

    if not failed_results:
        return

    print()
    print("Failed search evals:")
    print()

    for result in failed_results:
        print("Query:")
        print(result["query"])
        print()
        print("Expected document:")
        print(result["expected_document_id"])
        print()
        print("Top results:")

        for index, top_result in enumerate(result["top_results"], start=1):
            print(f"{index}. {top_result['url']}")
            print(f"   Title: {top_result['title']}")
            print(f"   Score: {top_result['score']}")

        print()


def print_grouped_summaries(
    title: str,
    summaries: dict[str, RetrievalEvalSummary],
) -> None:
    print(title)

    for group_name, summary in sorted(summaries.items()):
        print(
            f"{group_name}: "
            f"Top-1 {summary['passed_cases']}/{summary['total_cases']} "
            f"{summary['accuracy']:.2%} | "
            f"Top-3 {summary['top_3_passed_cases']}/{summary['total_cases']} "
            f"{summary['top_3_accuracy']:.2%}"
        )


def main() -> int:
    args = parse_args()

    pages = load_knowledge_base(KNOWLEDGE_PAGES_FILE)
    eval_cases = load_retrieval_eval_cases(RETRIEVAL_EVAL_CASES_FILE)

    if args.split != "all":
        eval_cases = [case for case in eval_cases if case["split"] == args.split]

    search_results = run_search_evals(eval_cases, pages)

    split_summaries = build_grouped_retrieval_eval_summaries(
        search_results,
        group_by="split",
    )

    category_summaries = build_grouped_retrieval_eval_summaries(
        search_results,
        group_by="category",
    )

    difficulty_summaries = build_grouped_retrieval_eval_summaries(
        search_results,
        group_by="difficulty",
    )

    print_grouped_summaries("By split", split_summaries)
    print_grouped_summaries("By category", category_summaries)
    print_grouped_summaries("By difficulty", difficulty_summaries)

    created_at = datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")
    git_commit = get_current_git_commit()
    git_dirty = is_git_worktree_dirty()

    report = build_search_eval_report(
        search_results,
        created_at=created_at,
        git_commit=git_commit,
        git_dirty=git_dirty,
    )

    print(f"Search evals: {report['passed_cases']}/{report['total_cases']} passed")

    save_search_eval_report(report, SEARCH_EVAL_REPORT_FILE)

    print(f"Saved report to {SEARCH_EVAL_REPORT_FILE}")

    failed_cases = report["total_cases"] - report["passed_cases"]

    if failed_cases > 0:
        print_failed_search_eval_results(report["results"])

    print(f"Accuracy: {report['accuracy']:.2%}")
    print(f"Top-3 accuracy: {report['top_3_accuracy']:.2%}")

    if args.split == "dev":
        if report["accuracy"] < DEV_MIN_ACCURACY:
            return 1

        if report["top_3_accuracy"] < DEV_MIN_TOP_3_ACCURACY:
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
