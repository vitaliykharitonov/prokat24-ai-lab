import pytest

from prokat24_ai_lab.product_catalog import (
    filter_product_category_urls,
    is_product_category_url,
)


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://prokat24.com.ua/ua/dostavka/", False),
        ("https://prokat24.com.ua/ua/contact-us/", False),
        ("https://prokat24.com.ua/ua/o-nas/", False),
        ("https://prokat24.com.ua/ua/login/", False),
        ("https://prokat24.com.ua/ua/shopnews/", False),
        ("https://prokat24.com.ua/ua/kak-vybrat-gornolyzhnyj-shlem/", False),
        ("https://prokat24.com.ua/ua/turisticheskoe-snaryazhenie/", True),
        ("https://prokat24.com.ua/ua/turisticheskoe-snaryazhenie/prokat-palatok/", True),
        ("https://prokat24.com.ua/ua/gornolyzhnoe-snaryazhenie/", True),
        ("https://prokat24.com.ua/ua/gornolyzhnoe-snaryazhenie/prokat-lyzh/", True),
        ("https://prokat24.com.ua/ua/prokat-mebeli/", True),
        ("https://prokat24.com.ua/ua/prokat-mebeli/prokat-stolov/", True),
    ],
)
def test_is_product_category_url_filters_correctly(url: str, expected: bool) -> None:
    assert is_product_category_url(url) == expected


def test_filter_product_category_urls_keeps_only_unique_sorted_category_urls() -> None:
    urls = [
        "https://prokat24.com.ua/ua/dostavka/",
        "https://prokat24.com.ua/ua/turisticheskoe-snaryazhenie/prokat-palatok/",
        "https://prokat24.com.ua/ua/turisticheskoe-snaryazhenie/prokat-palatok/",
        "https://prokat24.com.ua/ua/turisticheskoe-snaryazhenie/",
        "https://prokat24.com.ua/ua/turisticheskoe-snaryazhenie/",
    ]

    result = filter_product_category_urls(urls)

    assert result == [
        "https://prokat24.com.ua/ua/turisticheskoe-snaryazhenie/",
        "https://prokat24.com.ua/ua/turisticheskoe-snaryazhenie/prokat-palatok/",
    ]
