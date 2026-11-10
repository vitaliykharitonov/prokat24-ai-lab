import pytest

from prokat24_ai_lab.products.query import (
    build_product_search_query,
    detect_availability_intent,
    detect_deposit_intent,
    detect_product_details_intent,
    detect_rental_day_count,
    detect_rental_period,
    detect_requested_attribute_label,
    remove_rental_day_count_words,
    remove_word_sequence,
)


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


def test_remove_rental_day_count_words() -> None:
    words = ["скільки", "коштує", "2", "місний", "намет", "на", "5", "днів"]
    day_count = 5

    result = remove_rental_day_count_words(words, day_count)

    assert result == ["скільки", "коштує", "2", "місний", "намет", "на"]


def test_remove_rental_day_count_words_ignores_unknown_day_count_phrase() -> None:
    words = ["багато", "днів", "5", "днів", "намет"]
    day_count = 5

    result = remove_rental_day_count_words(words, day_count)

    assert result == ["багато", "днів", "намет"]


@pytest.mark.parametrize(
    ("query", "expected_query"),
    [
        ("скільки коштує 2 місний намет на тиждень", "2 місний намет"),
        ("скільки коштує 2 місний намет на 5 днів", "2 місний намет"),
        ("скільки буде коштувати на 5 днів 2 місний намет", "2 місний намет"),
        ("чи є в наявності 2 місний намет", "2 місний намет"),
        ("є човен", "човен"),
        ("чи маєте байдарку", "байдарку"),
        ("у вас є намет", "намет"),
        ("які характеристики у 2 місний намет", "2 місний намет"),
        ("який опис у 2 місний намет", "2 місний намет"),
        ("що входить в комплект 2 місний намет", "2 місний намет"),
        ("комплектація 2 місний намет", "2 місний намет"),
        ("розмір 2 місний намет", "2 місний намет"),
        ("вага 2 місний намет", "2 місний намет"),
        ("яка вага 2 місного намету", "2 місного намету"),
    ],
)
def test_build_product_search_query_removes_non_product_words(
    query: str, expected_query: str
) -> None:
    result = build_product_search_query(query)

    assert result == expected_query


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


@pytest.mark.parametrize(
    ("query", "expected_count"),
    [
        ("скільки коштує на 1 день", 1),
        ("скільки коштує на один день", 1),
        ("скільки коштує на 2 дні", 2),
        ("скільки коштує на два дні", 2),
        ("скільки коштує на 5 днів", 5),
        ("скільки коштує на пʼять днів", 5),
        ("скільки коштує 2 місний намет", None),
        ("скільки коштує 2 місний намет на 5 днів", 5),
        ("скільки коштує на тиждень", None),
    ],
)
def test_detect_rental_day_count(query: str, expected_count: int | None) -> None:
    assert detect_rental_day_count(query) == expected_count


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


@pytest.mark.parametrize(
    ("query", "expected_result"),
    [
        ("чи є в наявності 2 місний намет", True),
        ("є човен?", True),
        ("чи маєте байдарку", True),
        ("у вас є намет?", True),
        ("наявність човна", True),
        ("скільки коштує 2 місний намет", False),
        ("яка застава за човен", False),
        ("ціна на тиждень байдарка", False),
        ("яка є ціна на намет", False),
        ("яка є застава за човен", False),
    ],
)
def test_detect_availability_intent(query: str, expected_result: bool) -> None:
    assert detect_availability_intent(query) is expected_result


@pytest.mark.parametrize(
    ("query", "expected_result"),
    [
        ("які характеристики у 2 місного намету", True),
        ("який опис у 2 місного намету", True),
        ("що входить в комплект 2 місного намету", True),
        ("комплектація 2 місного намету", True),
        ("розмір 2 місного намету", True),
        ("вага 2 місного намету", True),
        ("скільки коштує 2 місний намет", False),
        ("яка застава за човен", False),
        ("чи є в наявності байдарка", False),
        ("ціна на тиждень намет", False),
    ],
)
def test_detect_product_details_intent(query: str, expected_result: bool) -> None:
    assert detect_product_details_intent(query) is expected_result


@pytest.mark.parametrize(
    ("query", "expected_attribute"),
    [
        ("яка вага 2 місний намет", "Вага"),
        ("скільки важить 2 місний намет", "Вага"),
        ("якої ваги 2 місний намет", "Вага"),
        ("який розмір 2 місний намет", "Розмір"),
        ("якого розміру 2 місний намет", "Розмір"),
        ("2 місний намет комплектація", "Комплектація"),
        ("2 місний намет що у комплекті", "Комплектація"),
        ("які характеристики у 2 місний намет", None),
        ("який опис у 2 місний намет", None),
        ("скільки коштує 2 місний намет", None),
        ("яка застава за човен", None),
        ("чи є в наявності байдарка", None),
    ],
)
def test_detect_requested_attribute_label(query: str, expected_attribute: str | None) -> None:
    assert detect_requested_attribute_label(query) == expected_attribute
