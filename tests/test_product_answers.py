from prokat24_ai_lab.product_answers import (
    format_product_search_result,
    format_product_search_results,
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
