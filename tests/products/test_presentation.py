import pytest

from prokat24_ai_lab.products.catalog import Product
from prokat24_ai_lab.products.presentation import (
    format_product_attribute_answer,
    format_product_day_count_rental_price_answer,
    format_product_deposit_answer,
    format_product_details_answer,
    format_product_rental_price_answer,
    format_product_search_result,
    format_product_search_results,
)


def test_format_product_search_result_includes_product_fields(two_person_tent: Product) -> None:
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


def test_format_product_deposit_answer_includes_deposit(two_person_tent: Product) -> None:
    result = format_product_deposit_answer(two_person_tent)

    assert "Прокат 2-місних наметів" in result
    assert "Застава: 4000 грн або Закордонний паспорт/права" in result
    assert two_person_tent["url"] in result


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
