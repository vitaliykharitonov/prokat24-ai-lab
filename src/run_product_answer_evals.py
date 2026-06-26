from prokat24_ai_lab.product_answer_evals import (
    PRODUCT_ANSWER_EVAL_CASES,
    ProductAnswerEvalFailure,
    find_failed_product_answer_evals,
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


def main() -> None:
    loaded_products = load_products(PRODUCTS_FILE)
    failures = find_failed_product_answer_evals(loaded_products)
    total_count = len(PRODUCT_ANSWER_EVAL_CASES)
    passed_count = total_count - len(failures)

    print(format_product_answer_eval_summary(passed_count, total_count))

    if failures:
        print()
        print("FAILED:")

        for failure in failures:
            print(format_product_answer_eval_failure(failure))


if __name__ == "__main__":
    main()
