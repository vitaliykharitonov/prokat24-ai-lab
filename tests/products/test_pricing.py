import pytest

from prokat24_ai_lab.products.catalog import Product
from prokat24_ai_lab.products.pricing import (
    calculate_rental_price_for_day_count,
    format_day_count_label,
    parse_price_amount,
)


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
