from run_product_answer_evals import (
    format_product_answer_eval_failure,
    format_product_answer_eval_summary,
    get_product_answer_eval_exit_code,
)


def test_format_product_answer_eval_summary_evaluates_correctly() -> None:
    passed_count = 4
    total_count = 5

    result = format_product_answer_eval_summary(passed_count, total_count)

    assert result == "Product answer evals: 4/5 passed"


def test_format_product_answer_eval_failure_formats_failure_details() -> None:
    failure = {
        "query": "яка застава за 2 місний намет",
        "answer": "Products not found",
        "missing_fragments": [
            "Прокат 2-місних наметів",
            "Deposit: 4000 грн",
        ],
    }

    result = format_product_answer_eval_failure(failure)

    assert result == (
        "Query: яка застава за 2 місний намет\n"
        "Missing fragments:\n"
        "- Прокат 2-місних наметів\n"
        "- Deposit: 4000 грн\n"
        "Answer:\n"
        "Products not found"
    )


def test_get_product_answer_eval_exit_code_returns_zero_without_failures() -> None:
    failure_count = 0

    result = get_product_answer_eval_exit_code(failure_count)

    assert result == 0


def test_get_product_answer_eval_exit_code_returns_one_with_failures() -> None:
    failure_count = 3

    result = get_product_answer_eval_exit_code(failure_count)

    assert result == 1
