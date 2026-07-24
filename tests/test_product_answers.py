import pytest

from prokat24_ai_lab.product_answers import (
    answer_product_query,
    build_product_search_query,
    calculate_rental_price_for_day_count,
    detect_availability_intent,
    detect_deposit_intent,
    detect_product_details_intent,
    detect_rental_day_count,
    detect_rental_period,
    detect_requested_attribute_label,
    format_day_count_label,
    format_product_attribute_answer,
    format_product_day_count_rental_price_answer,
    format_product_deposit_answer,
    format_product_details_answer,
    format_product_rental_price_answer,
    format_product_search_result,
    format_product_search_results,
    parse_price_amount,
    remove_rental_day_count_words,
    remove_word_sequence,
)
from prokat24_ai_lab.product_catalog import Product


@pytest.fixture
def two_person_tent() -> Product:
    return {
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


@pytest.fixture
def three_person_tent() -> Product:
    return {
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


@pytest.fixture
def full_product() -> Product:
    return {
        "url": "https://prokat24.com.ua/ua/turisticheskoe-snaryazhenie/prokat-palatok/prokat-palatki-2-mestnaya/",
        "name": "Прокат 2-місних наметів",
        "description": "Прокат легких двомісних наметів Terra Incognita Alfa2. "
        "Ідеально підходить для туристичних походів. "
        "Ефективна вентиляція, просте та швидке встановлення.",
        "rental_prices": {
            "1-й день": "250 грн",
            "2-й день": "220 грн",
            "3-й день": "190 грн",
            "з 4-го дня": "160 грн",
            "Тиждень": "1200 грн",
            "2 тижні": "2300 грн",
        },
        "deposit": "4000 грн або Закордонний паспорт/права",
        "attributes": {
            "Розмір": "205 х 240 см",
            "Висота": "115 см",
            "Вага": "3.31 кг",
            "К-сть осіб": "2 - 3",
            "Колір": "Хакі, пісочний",
            "Габарити упаковки": "15 х 15 х 45 см",
            "Комплектація": "Намет, зовнішній тент, набір кілочків (10 шт), дуги (2 шт), чохли",
        },
    }


@pytest.fixture
def product_without_extra_day_price() -> Product:
    return {
        "url": "https://prokat24.com.ua/ua/test-tent/",
        "name": "Прокат 2-місних наметів",
        "description": "Опис 2-місного намету.",
        "rental_prices": {
            "1-й день": "250 грн",
            "2-й день": "220 грн",
            "3-й день": "190 грн",
            "Тиждень": "1200 грн",
            "2 тижні": "2300 грн",
        },
        "deposit": "4000 грн або Закордонний паспорт/права",
        "attributes": {
            "К-сть осіб": "2 - 3",
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


def test_format_product_search_result_includes_product_fields(two_person_tent: Product):
    result = format_product_search_result(two_person_tent)

    assert "Прокат 2-місних наметів" in result
    assert "4000 грн" in result
    assert "1-й день: 250 грн" in result
    assert "Тиждень: 1200 грн" in result
    assert "https://prokat24.com.ua/ua/test-tent/" in result


def test_format_product_search_results_formats_multiple_products(
    two_person_tent: Product, three_person_tent: Product
) -> None:
    product_list = [two_person_tent, three_person_tent]

    result = format_product_search_results(product_list)

    assert two_person_tent["name"] in result
    assert three_person_tent["name"] in result
    assert "\n\n" in result


def test_format_product_search_results_returns_message_when_empty() -> None:
    assert format_product_search_results([]) == "Товари не знайдено"


def test_answer_product_query_returns_formatted_search_results(
    two_person_tent: Product, three_person_tent: Product
) -> None:
    product_list = [two_person_tent, three_person_tent]
    query = "2 місних"

    result = answer_product_query(product_list, query, limit=5)

    assert "Прокат 2-місних наметів" in result
    assert "Прокат 3-місних наметів" not in result


def test_answer_product_query_returns_not_found_message_when_no_products_match(
    two_person_tent: Product,
) -> None:
    product_list = [two_person_tent]
    query = "абракадабра"

    result = answer_product_query(product_list, query)

    assert result == "Товари не знайдено"


def test_answer_product_query_returns_specific_period_price_when_period_detected(
    two_person_tent: Product, three_person_tent: Product
) -> None:
    product_list = [two_person_tent, three_person_tent]
    query = "скільки коштує 2 місний намет на тиждень"

    result = answer_product_query(product_list, query)

    assert "Прокат 2-місних наметів" in result
    assert "Тиждень: 1200 грн" in result
    assert "Застава:" in result
    assert two_person_tent["url"] in result


def test_answer_product_query_returns_deposit_when_deposit_intent_detected(
    two_person_tent: Product, three_person_tent: Product
) -> None:
    product_list = [two_person_tent, three_person_tent]
    query = "яка застава за 2 місний намет"

    result = answer_product_query(product_list, query)

    assert "Прокат 2-місних наметів" in result
    assert "Застава: 4000 грн або Закордонний паспорт/права" in result
    assert two_person_tent["url"] in result


def test_answer_product_query_returns_product_details(full_product: Product) -> None:
    query = "які характеристики у 2 місний намет"

    result = answer_product_query([full_product], query)

    assert "Прокат 2-місних наметів" in result
    assert "Прокат легких двомісних наметів" in result
    assert "Характеристики:" in result
    assert "Розмір: 205 х 240 см" in result
    assert "Вага: 3.31 кг" in result
    assert "Комплектація: Намет, зовнішній тент" in result


def test_answer_product_query_returns_requested_attribute(full_product: Product) -> None:
    query = "яка вага 2 місний намет"

    result = answer_product_query([full_product], query)

    assert "Прокат 2-місних наметів" in result
    assert "Вага: 3.31 кг" in result
    assert "Характеристики:" not in result


def test_answer_product_query_returns_message_when_requested_attribute_missing(
    two_person_tent: Product,
) -> None:
    query = "яка вага 2 місний намет"

    result = answer_product_query([two_person_tent], query)

    assert "Прокат 2-місних наметів" in result
    assert "Характеристику Вага не знайдено" in result


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
    ("price", "expected_amount"),
    [
        ("250 грн", 250),
        ("1200 грн", 1200),
        ("2 300 грн", 2300),
        ("абв", None),
    ],
)
def test_parse_price_amount(price: str, expected_amount: int | None) -> None:
    assert parse_price_amount(price) == expected_amount


@pytest.mark.parametrize(
    ("day_count", "expected_total_price"),
    [
        (1, 250),
        (2, 470),
        (3, 660),
        (5, 980),
    ],
)
def test_calculate_rental_price_for_day_count(
    full_product: Product, day_count: int, expected_total_price: int | None
) -> None:
    result = calculate_rental_price_for_day_count(full_product, day_count)

    assert result == expected_total_price


def test_calculate_rental_price_for_day_count_when_extra_day_price_missing(
    product_without_extra_day_price: Product,
) -> None:
    day_count = 5

    result = calculate_rental_price_for_day_count(product_without_extra_day_price, day_count)

    assert result is None


def test_format_product_rental_price_answer_includes_period_price(two_person_tent: Product) -> None:
    period = "Тиждень"

    result = format_product_rental_price_answer(two_person_tent, period)

    assert "Прокат 2-місних наметів" in result
    assert "Тиждень: 1200 грн" in result
    assert "4000 грн" in result
    assert two_person_tent["url"] in result


def test_format_product_rental_price_answer_returns_message_when_period_missing(
    two_person_tent: Product,
) -> None:
    period = "Місяць"

    result = format_product_rental_price_answer(two_person_tent, period)

    assert result == "Ціну для періоду Місяць не знайдено"


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


def test_format_product_deposit_answer_includes_deposit(two_person_tent: Product) -> None:
    result = format_product_deposit_answer(two_person_tent)

    assert "Прокат 2-місних наметів" in result
    assert "Застава: 4000 грн або Закордонний паспорт/права" in result
    assert two_person_tent["url"] in result


@pytest.mark.parametrize(
    ("day_count", "expected_label"),
    [
        (1, "1 день"),
        (2, "2 дні"),
        (3, "3 дні"),
        (4, "4 дні"),
        (5, "5 днів"),
        (10, "10 днів"),
        (11, "11 днів"),
        (20, "20 днів"),
        (21, "21 день"),
        (22, "22 дні"),
        (23, "23 дні"),
        (24, "24 дні"),
        (25, "25 днів"),
        (30, "30 днів"),
    ],
)
def test_format_day_count_label(day_count: int, expected_label: str) -> None:
    assert format_day_count_label(day_count) == expected_label


def test_format_product_day_count_rental_price_answer_includes_total_price(
    full_product: Product,
) -> None:
    day_count = 5

    result = format_product_day_count_rental_price_answer(full_product, day_count)

    assert "Прокат 2-місних наметів" in result
    assert "5 днів: 980 грн" in result
    assert "Застава:" in result
    assert "URL:" in result


def test_format_product_day_count_rental_price_answer_returns_message_when_price_missing(
    product_without_extra_day_price: Product,
) -> None:
    day_count = 5

    result = format_product_day_count_rental_price_answer(
        product_without_extra_day_price, day_count
    )

    assert result == "Ціну на 5 днів не знайдено"


def test_format_product_details_answer(full_product: Product) -> None:
    result = format_product_details_answer(full_product)

    assert "Прокат 2-місних наметів" in result
    assert "Прокат легких двомісних наметів" in result
    assert "Характеристики:" in result
    assert "Розмір: 205 х 240 см" in result
    assert "Вага: 3.31 кг" in result
    assert "Комплектація: Намет, зовнішній тент" in result


@pytest.mark.parametrize(
    ("attribute_label", "expected_result"),
    [
        ("Вага", "Вага: 3.31 кг"),
        ("Розмір", "Розмір: 205 х 240 см"),
        ("Комплектація", "Комплектація: Намет, зовнішній тент"),
    ],
)
def test_format_product_attribute_answer(
    full_product: Product, attribute_label: str, expected_result: str
) -> None:
    result = format_product_attribute_answer(full_product, attribute_label)

    assert "Прокат 2-місних наметів" in result
    assert expected_result in result


def test_format_product_attribute_answer_returns_message_when_attribute_missing(
    two_person_tent: Product,
) -> None:
    attribute_label = "Вага"
    result = format_product_attribute_answer(two_person_tent, attribute_label)

    assert "Прокат 2-місних наметів" in result
    assert "Характеристику Вага не знайдено" in result
