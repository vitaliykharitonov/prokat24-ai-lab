import pytest

import prokat24_ai_lab.product_answer_evals as product_answer_evals
from prokat24_ai_lab.product_catalog import Product


def test_find_missing_fragments_returns_missing_fragments() -> None:
    answer = """
        Прокат 2-місних наметів
        Тиждень: 1200 грн
    """

    expected_fragments = [
        "Прокат 2-місних наметів",
        "Тиждень: 1200 грн",
        "Deposit: 4000 грн",
    ]

    result = product_answer_evals.find_missing_fragments(answer, expected_fragments)

    assert result == [
        "Deposit: 4000 грн",
    ]


def test_find_missing_fragments_returns_empty_list_when_all_fragments_present() -> None:
    answer = """
            Прокат 2-місних наметів
            Тиждень: 1200 грн
        """

    expected_fragments = [
        "Прокат 2-місних наметів",
        "Тиждень: 1200 грн",
    ]

    result = product_answer_evals.find_missing_fragments(answer, expected_fragments)

    assert result == []


def test_build_product_answer_eval_report_returns_summary_and_results() -> None:
    passed_count = 23
    total_count = 23
    failures = []

    report = product_answer_evals.build_product_answer_eval_report(
        passed_count, total_count, failures
    )

    assert report["passed"]
    assert report["passed_count"] == 23
    assert report["total_count"] == 23
    assert report["failed_count"] == 0
    assert report["failures"] == []


def test_build_product_answer_eval_report_with_failures() -> None:
    passed_count = 22
    total_count = 23
    failures = [
        {
            "query": "test query",
            "missing_fragments": ["fragment"],
            "answer": "actual answer",
        },
    ]

    report = product_answer_evals.build_product_answer_eval_report(
        passed_count, total_count, failures
    )

    assert not report["passed"]
    assert report["passed_count"] == 22
    assert report["total_count"] == 23
    assert report["failed_count"] == 1
    assert report["failures"] == failures


def test_build_product_answer_eval_report_for_products_returns_passing_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    product: Product = {
        "name": "Прокат 2-місних наметів",
        "description": "Опис 2-місного намету.",
        "attributes": {
            "Вага": "3.31 кг",
        },
        "rental_prices": {
            "1-й день": "250 грн",
            "Тиждень": "1200 грн",
        },
        "deposit": "4000 грн або Закордонний паспорт/права",
        "url": "https://prokat24.com.ua/ua/test-tent/",
    }

    monkeypatch.setattr(
        product_answer_evals,
        "PRODUCT_ANSWER_EVAL_CASES",
        [
            {
                "query": "яка вага 2 місний намет",
                "expected_fragments": [
                    "Прокат 2-місних наметів",
                    "Вага: 3.31 кг",
                ],
            },
        ],
    )

    report = product_answer_evals.build_product_answer_eval_report_for_products(
        [product],
    )

    assert report["passed"]
    assert report["passed_count"] == 1
    assert report["total_count"] == 1
    assert report["failed_count"] == 0
    assert report["failures"] == []
