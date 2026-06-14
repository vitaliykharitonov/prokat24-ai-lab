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
        ("https://prokat24.com.ua/ua/palatki/", True),
        ("https://prokat24.com.ua/ua/spalniki/", True),
        ("https://example.com/ua/palatki/", False),
        ("https://prokat24.com.ua/ru/palatki/", False),
        ("https://prokat24.com.ua/ru/palatki", False),
    ],
)
def test_is_product_category_url_filters_correctly(url: str, expected: bool) -> None:
    assert is_product_category_url(url) == expected


def test_filter_product_category_urls_keeps_only_unique_sorted_category_urls() -> None:
    urls = [
        "https://prokat24.com.ua/ua/dostavka/",
        "https://prokat24.com.ua/ua/palatki/",
        "https://prokat24.com.ua/ua/palatki/",
        "https://example.com/ua/spalniki/",
        "https://prokat24.com.ua/ua/spalniki/",
    ]

    result = filter_product_category_urls(urls)

    assert result == [
        "https://prokat24.com.ua/ua/palatki/",
        "https://prokat24.com.ua/ua/spalniki/",
    ]
