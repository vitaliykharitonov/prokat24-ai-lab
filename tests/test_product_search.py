import pytest

from prokat24_ai_lab.product_catalog import Product
from prokat24_ai_lab.product_search import (
    build_product_search_text,
    search_products,
)

PRODUCT = {
    "url": (
        "https://prokat24.com.ua/ua/turisticheskoe-snaryazhenie/"
        "prokat-palatok/prokat-palatki-2-mestnaya/"
    ),
    "name": "Прокат 2-місних наметів",
    "description": "Опис 2-місного намету.",
    "rental_prices": {
        "1-й день": "250 грн",
    },
    "deposit": "4000 грн або Закордонний паспорт/права",
    "attributes": {
        "К-сть осіб": "2 - 3",
    },
}

PRODUCT_2 = {
    "url": "https://prokat24.com.ua/ua/test-sleeping-bag/",
    "name": "Прокат літнього спальника",
    "description": "Опис літнього спальника.",
    "rental_prices": {
        "1-й день": "100 грн",
    },
    "deposit": "1000 грн",
    "attributes": {
        "Температура комфорту": "+10°C",
    },
}


def test_build_product_search_text_contains_product_fields() -> None:
    result = build_product_search_text(PRODUCT)

    assert "Прокат 2-місних наметів" in result
    assert "Опис 2-місного намету." in result
    assert "1-й день" in result
    assert "250 грн" in result
    assert "К-сть осіб" in result
    assert "4000 грн" in result
    assert "2 - 3" in result


@pytest.mark.parametrize(
    ("query", "expected_result"),
    [
        ("2 місних", [PRODUCT]),
        ("", []),
        ("   ", []),
    ],
)
def test_search_products_returns_products_matching_query_words(
    query: str,
    expected_result: list[Product],
) -> None:
    product_list = [PRODUCT, PRODUCT_2]

    result = search_products(product_list, query)

    assert result == expected_result


@pytest.mark.parametrize(
    ("limit", "expected_result"),
    [
        (1, [PRODUCT]),
        (0, []),
    ],
)
def test_search_products_respects_limit(
    limit: int,
    expected_result: list[Product],
) -> None:
    query = "прокат"
    product_list = [PRODUCT, PRODUCT_2]

    result = search_products(product_list, query, limit)

    assert result == expected_result


def test_search_products_ranks_name_matches_above_price_matches() -> None:
    three_person_boat: Product = {
        "url": "https://prokat24.com.ua/ua/test-boat/",
        "name": "Прокат 3-місних човнів",
        "description": "Опис човна.",
        "rental_prices": {
            "1-й день": "500 грн",
            "2-й день": "450 грн",
        },
        "deposit": "10000 грн",
        "attributes": {},
    }
    query = "2 місних"
    product_list = [three_person_boat, PRODUCT]

    result = search_products(product_list, query, limit=2)

    assert result == [PRODUCT, three_person_boat]
