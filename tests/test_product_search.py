import pytest

from prokat24_ai_lab.products.catalog import Product
from prokat24_ai_lab.products.catalog_search import (
    build_product_query_aliases,
    build_product_search_text,
    build_query_word_groups,
    contains_all_word_groups,
    search_products,
)


@pytest.fixture
def two_person_tent() -> Product:
    return {
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


@pytest.fixture
def summer_sleeping_bag() -> Product:
    return {
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


@pytest.fixture
def summer_sleeping_bags() -> Product:
    return {
        "url": "https://prokat24.com.ua/ua/test-sleeping-bag/",
        "name": "Прокат літніх спальників",
        "description": "Опис літнього спальника.",
        "rental_prices": {
            "1-й день": "100 грн",
        },
        "deposit": "1000 грн",
        "attributes": {},
    }


@pytest.fixture
def ski_boots() -> Product:
    return {
        "url": "https://prokat24.com.ua/ua/test-ski-boots/",
        "name": "Прокат лижних черевиків",
        "description": "Опис лижних черевиків.",
        "rental_prices": {
            "1-й день": "150 грн",
        },
        "deposit": "1500 грн",
        "attributes": {},
    }


@pytest.fixture
def boat() -> Product:
    return {
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


def test_build_product_search_text_contains_product_fields(two_person_tent: Product) -> None:
    result = build_product_search_text(two_person_tent)

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
    ("query", "expected_product_names"),
    [
        ("2 місних", ["Прокат 2-місних наметів"]),
        ("", []),
        ("   ", []),
    ],
)
def test_search_products_returns_products_matching_query_words(
    two_person_tent: Product,
    summer_sleeping_bag: Product,
    query: str,
    expected_product_names: list[str],
) -> None:
    product_list = [two_person_tent, summer_sleeping_bag]

    result = search_products(product_list, query)

    assert [product["name"] for product in result] == expected_product_names


def test_search_products_matches_product_by_query_alias(two_person_tent: Product) -> None:
    result = search_products([two_person_tent], "палатка")

    assert result == [two_person_tent]


@pytest.mark.parametrize(
    ("limit", "expected_product_names"),
    [
        (1, ["Прокат 2-місних наметів"]),
        (0, []),
    ],
)
def test_search_products_respects_limit(
    two_person_tent: Product,
    summer_sleeping_bag: Product,
    limit: int,
    expected_product_names: list[str],
) -> None:
    query = "прокат"
    product_list = [two_person_tent, summer_sleeping_bag]

    result = search_products(product_list, query, limit)

    assert [product["name"] for product in result] == expected_product_names


def test_search_products_orders_results_by_score(two_person_tent: Product) -> None:
    two_person_boat_wrong_description: Product = {
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
    product_list = [two_person_boat_wrong_description, two_person_tent]

    result = search_products(product_list, query, limit=2)

    assert result == [two_person_tent, two_person_boat_wrong_description]


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
    ("query", "expected_product_name"),
    [
        ("палатка", "Прокат 2-місних наметів"),
        ("прокат спальника", "Прокат літніх спальників"),
        ("лижні черевики", "Прокат лижних черевиків"),
        ("черевики", "Прокат лижних черевиків"),
        ("човен", "Прокат човнів"),
        ("2 місного намету", "Прокат 2-місних наметів"),
    ],
)
def test_search_products_matches_product_aliases(
    two_person_tent: Product,
    summer_sleeping_bags: Product,
    ski_boots: Product,
    boat: Product,
    query: str,
    expected_product_name: str,
) -> None:
    products = [
        two_person_tent,
        summer_sleeping_bags,
        ski_boots,
        boat,
    ]

    result = search_products(products, query)

    assert result
    assert result[0]["name"] == expected_product_name


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
