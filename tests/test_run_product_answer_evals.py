from run_product_answer_evals import format_product_answer_eval_summary


def test_format_product_answer_eval_summary_evaluates_correctly() -> None:
    passed_count = 4
    total_count = 5

    result = format_product_answer_eval_summary(passed_count, total_count)

    assert result == "Product answer evals: 4/5 passed"
