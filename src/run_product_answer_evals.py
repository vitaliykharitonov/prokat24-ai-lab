from prokat24_ai_lab.product_answer_evals import (
    ProductAnswerEvalFailure,
    build_product_answer_eval_report_for_products,
)
from prokat24_ai_lab.product_catalog import load_products
from prokat24_ai_lab.project_paths import PRODUCTS_FILE


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


def get_product_answer_eval_exit_code(failure_count: int) -> int:
    if failure_count == 0:
        return 0

    return 1


def main() -> int:
    loaded_products = load_products(PRODUCTS_FILE)
    report = build_product_answer_eval_report_for_products(loaded_products)

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

    return get_product_answer_eval_exit_code(report["failed_count"])


if __name__ == "__main__":
    raise SystemExit(main())
