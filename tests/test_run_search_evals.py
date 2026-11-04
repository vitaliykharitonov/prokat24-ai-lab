import pytest
from scripts.evals.run_search_evals import print_failed_search_eval_results


def test_print_failed_search_eval_results_prints_nothing_when_all_results_pass(
    capsys: pytest.CaptureFixture[str],
) -> None:
    results = [
        {
            "query": "доставка",
            "expected_document_id": "https://example.com/delivery",
            "top_results": [
                "https://example.com/delivery",
                "https://example.com/payment",
            ],
            "passed": True,
        }
    ]

    print_failed_search_eval_results(results)
    captured = capsys.readouterr()

    assert captured.out == ""


def test_print_failed_search_eval_results_prints_failed_result_details(
    capsys: pytest.CaptureFixture[str],
) -> None:
    results = [
        {
            "query": "доставка",
            "expected_document_id": "https://example.com/delivery",
            "top_results": [
                {
                    "url": "https://example.com/payment",
                    "title": "Payment page",
                    "snippet": "Payment info",
                    "score": 3.0,
                    "score_details": {},
                    "matched_words": {},
                },
                {
                    "url": "https://example.com/contacts",
                    "title": "Contacts page",
                    "snippet": "Contacts info",
                    "score": 2.0,
                    "score_details": {},
                    "matched_words": {},
                },
            ],
            "passed": False,
        }
    ]

    print_failed_search_eval_results(results)

    captured = capsys.readouterr()

    assert "Failed search evals:" in captured.out
    assert "Query:" in captured.out
    assert "доставка" in captured.out
    assert "Expected document:" in captured.out
    assert "https://example.com/delivery" in captured.out
    assert "Top results:" in captured.out
    assert "1. https://example.com/payment" in captured.out
    assert "Title: Payment page" in captured.out
    assert "Score: 3.0" in captured.out
    assert "2. https://example.com/contacts" in captured.out
    assert "Title: Contacts page" in captured.out
    assert "Score: 2.0" in captured.out


def test_print_failed_search_eval_results_does_not_print_passed_results(
    capsys: pytest.CaptureFixture[str],
) -> None:
    results = [
        {
            "query": "поганий запит",
            "expected_document_id": "https://example.com/expected",
            "top_results": [
                {
                    "url": "https://example.com/wrong",
                    "title": "Wrong page",
                    "snippet": "Wrong result",
                    "score": 1.0,
                    "score_details": {},
                    "matched_words": {},
                },
            ],
            "passed": False,
        },
        {
            "query": "успішний запит",
            "expected_document_id": "https://example.com/correct",
            "top_results": [
                {
                    "url": "https://example.com/correct",
                    "title": "Correct page",
                    "snippet": "Correct result",
                    "score": 5.0,
                    "score_details": {},
                    "matched_words": {},
                },
            ],
            "passed": True,
        },
    ]

    print_failed_search_eval_results(results)

    captured = capsys.readouterr()

    assert "поганий запит" in captured.out
    assert "успішний запит" not in captured.out
