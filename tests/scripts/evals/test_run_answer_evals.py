import pytest
from scripts.evals.run_answer_evals import print_failed_answer_eval_results


def test_print_failed_answer_eval_results_prints_nothing_when_all_results_pass(
    capsys: pytest.CaptureFixture[str],
) -> None:
    results = [
        {
            "question": "як оформити прокат",
            "answer": "Для прокату потрібен документ.",
            "expected_answer_fragments": ["документ"],
            "missing_answer_fragments": [],
            "passed": True,
        }
    ]

    print_failed_answer_eval_results(results)

    captured = capsys.readouterr()

    assert captured.out == ""


def test_print_failed_answer_eval_results_prints_failed_result_details(
    capsys: pytest.CaptureFixture[str],
) -> None:
    results = [
        {
            "question": "яка застава за намет",
            "answer": "Зверніться до менеджера.",
            "expected_answer_fragments": ["Застава", "1000 грн"],
            "missing_answer_fragments": ["Застава", "1000 грн"],
            "passed": False,
        }
    ]

    print_failed_answer_eval_results(results)

    captured = capsys.readouterr()

    assert "Failed answer evals:" in captured.out
    assert "Question:" in captured.out
    assert "яка застава за намет" in captured.out
    assert "Missing fragments:" in captured.out
    assert "- Застава" in captured.out
    assert "- 1000 грн" in captured.out
    assert "Answer:" in captured.out
    assert "Зверніться до менеджера." in captured.out


def test_print_failed_answer_eval_results_does_not_print_passed_results(
    capsys: pytest.CaptureFixture[str],
) -> None:
    results = [
        {
            "question": "погане питання",
            "answer": "неповна відповідь",
            "expected_answer_fragments": ["важливий фрагмент"],
            "missing_answer_fragments": ["важливий фрагмент"],
            "passed": False,
        },
        {
            "question": "успішне питання",
            "answer": "правильна відповідь",
            "expected_answer_fragments": ["правильна"],
            "missing_answer_fragments": [],
            "passed": True,
        },
    ]

    print_failed_answer_eval_results(results)

    captured = capsys.readouterr()

    assert "погане питання" in captured.out
    assert "важливий фрагмент" in captured.out
    assert "неповна відповідь" in captured.out

    assert "успішне питання" not in captured.out
    assert "правильна відповідь" not in captured.out
