from prokat24_ai_lab.product_answers import format_product_search_result
from prokat24_ai_lab.product_catalog import Product


def test_format_product_search_result():
    product: Product = {
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

    result = format_product_search_result(product)

    assert "Прокат 2-місних наметів" in result
    assert "4000 грн" in result
    assert "1-й день: 250 грн" in result
    assert "Тиждень: 1200 грн" in result
    assert "https://prokat24.com.ua/ua/test-tent/" in result
