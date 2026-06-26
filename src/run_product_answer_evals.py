from prokat24_ai_lab.product_answer_evals import (
    PRODUCT_ANSWER_EVAL_CASES,
    count_passed_product_answer_evals,
)
from prokat24_ai_lab.product_catalog import load_products
from prokat24_ai_lab.project_paths import PRODUCTS_FILE


def format_product_answer_eval_summary(passed_count: int, total_count: int) -> str:
    return f"Product answer evals: {passed_count}/{total_count} passed"


def main() -> None:
    loaded_products = load_products(PRODUCTS_FILE)
    passed_count = count_passed_product_answer_evals(loaded_products)
    total_count = len(PRODUCT_ANSWER_EVAL_CASES)

    print(format_product_answer_eval_summary(passed_count, total_count))


if __name__ == "__main__":
    main()
