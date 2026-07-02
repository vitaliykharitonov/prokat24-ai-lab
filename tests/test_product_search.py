import pytest

from prokat24_ai_lab.product_catalog import Product
from prokat24_ai_lab.product_search import (
    build_product_query_aliases,
    build_product_search_text,
    build_query_word_groups,
    contains_all_word_groups,
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

PRODUCT_3 = {
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

SLEEPING_BAG_PRODUCT: Product = {
    "url": "https://prokat24.com.ua/ua/test-sleeping-bag/",
    "name": "Прокат літніх спальників",
    "description": "Опис літнього спальника.",
    "rental_prices": {
        "1-й день": "100 грн",
    },
    "deposit": "1000 грн",
    "attributes": {},
}

SKI_BOOTS_PRODUCT: Product = {
    "url": "https://prokat24.com.ua/ua/test-ski-boots/",
    "name": "Прокат лижних черевиків",
    "description": "Опис лижних черевиків.",
    "rental_prices": {
        "1-й день": "150 грн",
    },
    "deposit": "1500 грн",
    "attributes": {},
}

BOAT_PRODUCT: Product = {
    "url": "https://prokat24.com.ua/ua/test-boat/",
    "name": "Прокат човнів",
    "description": "Опис човна.",
    "rental_prices": {
        "1-й день": "400 грн",
    },
    "deposit": "8000 грн",
    "attributes": {},
}


def test_build_product_query_aliases_builds_aliases() -> None:
    groups = (("намет", "намету", "наметів"),)

    results = build_product_query_aliases(groups)

    assert results == {
        "намет": {"намету", "наметів"},
        "намету": {"намет", "наметів"},
        "наметів": {"намет", "намету"},
    }


def test_build_query_word_groups_adds_aliases() -> None:
    query_words = ["прокат", "палатка"]

    result = build_query_word_groups(query_words)

    assert result == [
        {"прокат"},
        {"наметів", "намет", "палатці", "намети", "намету", "наметі", "палатки", "палатка"},
    ]


def test_build_product_search_text_contains_product_fields() -> None:
    result = build_product_search_text(PRODUCT)

    assert "Прокат 2-місних наметів" in result
    assert "Опис 2-місного намету." in result
    assert "1-й день" in result
    assert "250 грн" in result
    assert "К-сть осіб" in result
    assert "4000 грн" in result
    assert "2 - 3" in result


def test_contains_all_words_matches_alias_group() -> None:
    text = "Прокат 2-місних наметів"

    assert contains_all_word_groups(text, ["прокат", "палатка"])


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


def test_search_products_matches_product_by_query_alias() -> None:
    result = search_products([PRODUCT], "палатка")

    assert result == [PRODUCT]


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


def test_search_products_orders_results_by_score() -> None:
    two_person_boat: Product = {
        "url": "https://prokat24.com.ua/ua/test-boat/",
        "name": "Прокат 2-місних човнів",
        "description": "Опис човна.",
        "rental_prices": {
            "1-й день": "500 грн",
            "2-й день": "450 грн",
        },
        "deposit": "10000 грн",
        "attributes": {},
    }
    query = "2 місних"
    product_list = [two_person_boat, PRODUCT]

    result = search_products(product_list, query, limit=2)

    assert result == [PRODUCT, two_person_boat]


def test_search_products_does_not_match_query_words_split_across_fields() -> None:
    three_person_boat: Product = {
        "url": "https://prokat24.com.ua/ua/test-boat/",
        "name": "Прокат 3-місних човнів",
        "description": "Опис човна.",
        "rental_prices": {
            "1-й день": "500 грн",
            "2-й день": "450 грн",
        },
        "deposit": "10000 грн",
        "attributes": {
            "К-сть осіб": "2 - 3",
        },
    }

    query = "2 місних"

    result = search_products([three_person_boat], query)

    assert result == []


@pytest.mark.parametrize(
    ("query", "expected_result"),
    [
        ("палатка", [PRODUCT]),
        ("прокат спальника", [SLEEPING_BAG_PRODUCT]),
        ("лижні черевики", [SKI_BOOTS_PRODUCT]),
        ("черевики", [SKI_BOOTS_PRODUCT]),
        ("човен", [BOAT_PRODUCT]),
        ("2 місного намету", [PRODUCT]),
    ],
)
def test_search_products_matches_product_aliases(
    query: str,
    expected_result: list[Product],
) -> None:
    products = [
        PRODUCT,
        SLEEPING_BAG_PRODUCT,
        SKI_BOOTS_PRODUCT,
        BOAT_PRODUCT,
    ]

    result = search_products(products, query)

    assert result == expected_result


def test_search_products_ranks_alias_match_in_name_above_alias_match_in_description() -> None:
    ski_boots: Product = {
        "url": "https://prokat24.com.ua/ua/test-ski-boots/",
        "name": "Прокат лижних черевиків",
        "description": "Опис спорядження.",
        "rental_prices": {"1-й день": "150 грн"},
        "deposit": "1500 грн",
        "attributes": {},
    }
    description_match: Product = {
        "url": "https://prokat24.com.ua/ua/test-cover/",
        "name": "Прокат зимового спорядження",
        "description": "Підходить для черевиків.",
        "rental_prices": {"1-й день": "50 грн"},
        "deposit": "500 грн",
        "attributes": {},
    }

    query = "чоботи"
    product_list = [description_match, ski_boots]

    result = search_products(product_list, query, limit=2)

    assert result == [
        ski_boots,
        description_match,
    ]


def test_search_products_ranks_primary_name_match_above_secondary_name_match() -> None:
    boot_cover: Product = {
        "url": "https://prokat24.com.ua/ua/test-boot-cover/",
        "name": "Прокат чохлів для черевиків",
        "description": "Опис чохлів.",
        "rental_prices": {"1-й день": "50 грн"},
        "deposit": "500 грн",
        "attributes": {},
    }
    ski_boots: Product = {
        "url": "https://prokat24.com.ua/ua/test-ski-boots/",
        "name": "Прокат лижних черевиків",
        "description": "Опис лижних черевиків.",
        "rental_prices": {"1-й день": "150 грн"},
        "deposit": "1500 грн",
        "attributes": {},
    }
    query = "черевики"
    product_list = [boot_cover, ski_boots]

    result = search_products(product_list, query, limit=2)

    assert result == [
        ski_boots,
        boot_cover,
    ]
