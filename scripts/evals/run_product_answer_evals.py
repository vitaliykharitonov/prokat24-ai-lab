from prokat24_ai_lab.eval_exit_codes import get_eval_exit_code
from prokat24_ai_lab.evals.product_answer_evals import (
    ProductAnswerEvalFailure,
    build_product_answer_eval_report_for_products,
    save_product_answer_eval_report,
)
from prokat24_ai_lab.git_metadata import get_current_git_commit, is_git_worktree_dirty
from prokat24_ai_lab.products.catalog import load_products
from prokat24_ai_lab.project_paths import PRODUCT_ANSWER_EVAL_REPORT_FILE, PRODUCTS_FILE


def format_product_answer_eval_summary(passed_count: int, total_count: int) -> str:
    return f"Product answer evals: {passed_count}/{total_count} passed"


def format_product_answer_eval_failure(failure: ProductAnswerEvalFailure) -> str:
    lines = [
        f"Query: {failure['query']}",
        "Missing fragments:",
    ]

    for fragment in failure["missing_fragments"]:
        lines.append(f"- {fragment}")

    lines.append("Answer:")
    lines.append(failure["answer"])

    return "\n".join(lines)


def main() -> int:
    loaded_products = load_products(PRODUCTS_FILE)
    git_commit = get_current_git_commit()
    git_dirty = is_git_worktree_dirty()
    report = build_product_answer_eval_report_for_products(
        loaded_products,
        git_commit=git_commit,
        git_dirty=git_dirty,
    )
    save_product_answer_eval_report(report, PRODUCT_ANSWER_EVAL_REPORT_FILE)

    print(
        format_product_answer_eval_summary(
            report["passed_count"],
            report["total_count"],
        )
    )

    if report["failures"]:
        print()
        print("FAILED:")

        for failure in report["failures"]:
            print(format_product_answer_eval_failure(failure))

    return get_eval_exit_code(report["failed_count"])


if __name__ == "__main__":
    raise SystemExit(main())
