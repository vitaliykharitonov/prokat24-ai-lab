from prokat24_ai_lab.eval_exit_codes import get_eval_exit_code


def test_get_product_answer_eval_exit_code_returns_zero_without_failures() -> None:
    failed_count = 0

    result = get_eval_exit_code(failed_count)

    assert result == 0


def test_get_product_answer_eval_exit_code_returns_one_with_failures() -> None:
    failed_count = 3

    result = get_eval_exit_code(failed_count)

    assert result == 1
