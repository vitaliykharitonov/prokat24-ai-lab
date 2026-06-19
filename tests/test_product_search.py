from prokat24_ai_lab.product_search import build_product_search_text

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


def test_build_product_search_text_contains_product_fields() -> None:
    result = build_product_search_text(PRODUCT)

    assert "Прокат 2-місних наметів" in result
    assert "Опис 2-місного намету." in result
    assert "1-й день" in result
    assert "250 грн" in result
    assert "К-сть осіб" in result
    assert "4000 грн" in result
    assert "2 - 3" in result
