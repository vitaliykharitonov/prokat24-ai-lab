import pytest

from prokat24_ai_lab.product_catalog import is_product_category_url


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
