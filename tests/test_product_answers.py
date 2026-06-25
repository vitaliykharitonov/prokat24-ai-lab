import pytest

from prokat24_ai_lab.product_answers import (
    answer_product_query,
    build_product_search_query,
    detect_deposit_intent,
    detect_rental_period,
    format_product_deposit_answer,
    format_product_rental_price_answer,
    format_product_search_result,
    format_product_search_results,
    remove_word_sequence,
)

PRODUCT_1 = {
    "url": "https://prokat24.com.ua/ua/test-tent/",
    "name": "Прокат 2-місних наметів",
    "description": "Опис 2-місного намету.",
    "rental_prices": {
        "1-й день": "250 грн",
        "Тиждень": "1200 грн",
    },
    "deposit": "4000 грн або Закордонний паспорт/права",
    "attributes": {
        "К-сть осіб": "2 - 3",
    },
}

PRODUCT_2 = {
    "url": "https://prokat24.com.ua/ua/test-tent3/",
    "name": "Прокат 3-місних наметів",
    "description": "Опис 3-місного намету.",
    "rental_prices": {
        "1-й день": "300 грн",
        "Тиждень": "1500 грн",
    },
    "deposit": "4500 грн або закордонний паспорт/права + 1000 грн",
    "attributes": {
        "К-сть осіб": "3 - 4",
    },
}


def test_remove_word_sequence_removes_existing_sequence() -> None:
    words = ["прокат", "спальника", "на", "2", "тижні"]
    sequence = ["2", "тижні"]

    result = remove_word_sequence(words, sequence)

    assert result == ["прокат", "спальника", "на"]


def test_remove_word_sequence_keeps_product_number_when_sequence_is_later() -> None:
    words = ["скільки", "коштує", "2", "місний", "намет", "на", "2", "тижні"]
    sequence = ["2", "тижні"]

    result = remove_word_sequence(words, sequence)

    assert result == ["скільки", "коштує", "2", "місний", "намет", "на"]


def test_remove_word_sequence_returns_original_words_when_sequence_missing() -> None:
    words = ["прокат", "спальника"]
    sequence = ["2", "тижні"]

    result = remove_word_sequence(words, sequence)

    assert result == words


def test_build_product_search_query_removes_question_and_period_words() -> None:
    query = "скільки коштує 2 місний намет на тиждень"

    result = build_product_search_query(query)

    assert result == "2 місний намет"


def test_format_product_search_result_includes_product_fields():
    result = format_product_search_result(PRODUCT_1)

    assert "Прокат 2-місних наметів" in result
    assert "4000 грн" in result
    assert "1-й день: 250 грн" in result
    assert "Тиждень: 1200 грн" in result
    assert "https://prokat24.com.ua/ua/test-tent/" in result


def test_format_product_search_results_formats_multiple_products() -> None:
    product_list = [PRODUCT_1, PRODUCT_2]

    result = format_product_search_results(product_list)

    assert PRODUCT_1["name"] in result
    assert PRODUCT_2["name"] in result
    assert "\n\n" in result


def test_format_product_search_results_returns_message_when_empty() -> None:
    assert format_product_search_results([]) == "Products not found"


def test_answer_product_query_returns_formatted_search_results() -> None:
    product_list = [PRODUCT_1, PRODUCT_2]
    query = "2 місних"

    result = answer_product_query(product_list, query, limit=5)

    assert "Прокат 2-місних наметів" in result
    assert "Прокат 3-місних наметів" not in result


def test_answer_product_query_returns_not_found_message_when_no_products_match() -> None:
    product_list = [PRODUCT_1]
    query = "абракадабра"

    result = answer_product_query(product_list, query)

    assert result == "Products not found"


def test_answer_product_query_returns_specific_period_price_when_period_detected() -> None:
    product_list = [PRODUCT_1, PRODUCT_2]
    query = "скільки коштує 2 місний намет на тиждень"

    result = answer_product_query(product_list, query)

    assert "Прокат 2-місних наметів" in result
    assert "Тиждень: 1200 грн" in result
    assert "Deposit:" in result
    assert PRODUCT_1["url"] in result


def test_answer_product_query_returns_deposit_when_deposit_intent_detected() -> None:
    product_list = [PRODUCT_1, PRODUCT_2]
    query = "яка застава за 2 місний намет"

    result = answer_product_query(product_list, query)

    assert "Прокат 2-місних наметів" in result
    assert "Deposit: 4000 грн або Закордонний паспорт/права" in result
    assert PRODUCT_1["url"] in result


@pytest.mark.parametrize(
    ("query", "expected_period"),
    [
        ("яка вартість на тиждень", "Тиждень"),
        ("ціна за тиждень", "Тиждень"),
        ("на 2 тижні", "2 тижні"),
        ("на два тижні", "2 тижні"),
        ("на один день", "1-й день"),
        ("на 1 день", "1-й день"),
        ("скільки застава", None),
    ],
)
def test_detect_rental_period(query: str, expected_period: str | None) -> None:
    assert detect_rental_period(query) == expected_period


def test_format_product_rental_price_answer_includes_period_price() -> None:
    period = "Тиждень"

    result = format_product_rental_price_answer(PRODUCT_1, period)

    assert "Прокат 2-місних наметів" in result
    assert "Тиждень: 1200 грн" in result
    assert "4000 грн" in result
    assert PRODUCT_1["url"] in result


def test_format_product_rental_price_answer_returns_message_when_period_missing() -> None:
    period = "Місяць"

    result = format_product_rental_price_answer(PRODUCT_1, period)

    assert result == "Price for Місяць not found"


@pytest.mark.parametrize(
    ("query", "expected_result"),
    [
        ("яка застава за 2 місний намет", True),
        ("застава за чоботи", True),
        ("скільки застава за спальник", True),
        ("яка застава за човен", True),
        ("скільки коштує на тиждень", False),
        ("прокат спальника", False),
    ],
)
def test_detect_deposit_intent(query: str, expected_result: bool) -> None:
    assert detect_deposit_intent(query) is expected_result


def test_format_product_deposit_answer_includes_deposit() -> None:
    result = format_product_deposit_answer(PRODUCT_1)

    assert "Прокат 2-місних наметів" in result
    assert "Deposit: 4000 грн або Закордонний паспорт/права" in result
    assert PRODUCT_1["url"] in result
