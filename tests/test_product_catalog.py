import pytest

from prokat24_ai_lab.product_catalog import (
    build_product_from_html,
    extract_attributes_from_lines,
    extract_deposit_from_lines,
    extract_product_attributes,
    extract_product_deposit,
    extract_product_description,
    extract_product_info_lines,
    extract_product_name,
    extract_product_rental_prices,
    extract_rental_prices_from_lines,
    extract_section_lines,
    filter_product_category_urls,
    filter_product_urls,
    is_product_category_url,
    is_product_url,
    parse_key_value_lines,
)

CATEGORY_URL = "https://prokat24.com.ua/ua/turisticheskoe-snaryazhenie/prokat-palatok/"

PRODUCT_INFO_HTML = """
<div class="product-info__table-v2">
    <table>
        <tr><td>Вартість прокату</td></tr>
        <tr><td>1-й день</td><td>250 грн</td></tr>
        <tr><td>2-й день</td><td>220 грн</td></tr>
        <tr><td>Застава</td></tr>
        <tr><td>4000 грн або Закордонний паспорт/права</td></tr>
        <tr><td>Характеристики</td></tr>
        <tr><td>Розмір</td><td>205 х 240 см</td></tr>
        <tr><td>Висота</td><td>115 см</td></tr>
    </table>
</div>
"""

PRODUCT_INFO_LINES = [
    "Вартість прокату",
    "1-й день",
    "250 грн",
    "2-й день",
    "220 грн",
    "3-й день",
    "190 грн",
    "Застава",
    "4000 грн або Закордонний паспорт/права",
    "Характеристики",
    "Розмір",
    "205 х 240 см",
    "Висота",
    "115 см",
    "Вага",
    "3.31 кг",
]


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


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        (CATEGORY_URL, False),
        (CATEGORY_URL + "#popup-navigation", False),
        (CATEGORY_URL + "prokat-palatki-2-mestnaya/", True),
        (CATEGORY_URL + "prokat-palatki-3-mestnaya/", True),
        ("https://example.com/ua/turisticheskoe-snaryazhenie/prokat-palatok/x/", False),
        (
            "https://prokat24.com.ua/ua/turisticheskoe-snaryazhenie/"
            "prokat-spalnikov/prokat-spalnik/",
            False,
        ),
    ],
)
def test_is_product_url_filters_correctly(url: str, expected: bool) -> None:
    assert is_product_url(url, CATEGORY_URL) == expected


def test_filter_product_urls_keeps_only_unique_sorted_product_urls() -> None:
    urls = [
        CATEGORY_URL,
        CATEGORY_URL + "#popup-navigation",
        CATEGORY_URL + "prokat-palatki-2-mestnaya/",
        CATEGORY_URL + "prokat-palatki-2-mestnaya/",
        CATEGORY_URL + "prokat-palatki-3-mestnaya/",
        "https://prokat24.com.ua/ua/turisticheskoe-snaryazhenie/prokat-spalnikov/prokat-spalnik/",
        "https://example.com/ua/turisticheskoe-snaryazhenie/prokat-palatok/x/",
    ]

    result = filter_product_urls(CATEGORY_URL, urls)

    assert result == [
        CATEGORY_URL + "prokat-palatki-2-mestnaya/",
        CATEGORY_URL + "prokat-palatki-3-mestnaya/",
    ]


def test_extract_product_name_returns_product_name() -> None:
    html = """
    <span class="catalogue__product-name">
        Прокат 2-місних наметів
    </span>
    """

    result = extract_product_name(html)

    assert result == "Прокат 2-місних наметів"


def test_extract_product_name_returns_fallback_when_name_is_missing() -> None:
    html = """
    <span>
        Прокат 2-місних наметів
    </span>
    """

    result = extract_product_name(html)

    assert result == "No product name found"


def test_extract_product_name_returns_fallback_when_name_is_empty() -> None:
    html = """
    <span class="catalogue__product-name">
    </span>
    """

    result = extract_product_name(html)

    assert result == "No product name found"


def test_extract_product_description_returns_description() -> None:
    html = """
    <div class="short-descr">
        Прокат легких двомісних наметів Terra Incognita Alfa2.
    </div>
    """

    result = extract_product_description(html)

    assert result == "Прокат легких двомісних наметів Terra Incognita Alfa2."


def test_extract_product_description_returns_fallback_when_description_is_missing() -> None:
    html = """
    <div>
        Прокат легких двомісних наметів Terra Incognita Alfa2.
    </div>
    """

    result = extract_product_description(html)

    assert result == "No product description found"


def test_extract_product_description_returns_fallback_when_description_is_empty() -> None:
    html = """
    <div class="short-descr">
    </div>
    """

    result = extract_product_description(html)

    assert result == "No product description found"


def test_extract_product_info_lines_returns_table_lines() -> None:
    html = """
    <div class="product-info__table-v2">
    <table>
        <tr><td>Вартість прокату</td></tr>
        <tr><td>1-й день</td><td>250 грн</td></tr>
        <tr><td>Застава</td></tr>
        <tr><td>4000 грн або Закордонний паспорт/права</td></tr>
    </table>
    </div>
    """

    result = extract_product_info_lines(html)

    assert result == [
        "Вартість прокату",
        "1-й день",
        "250 грн",
        "Застава",
        "4000 грн або Закордонний паспорт/права",
    ]


def test_extract_product_info_lines_returns_empty_list_when_table_is_missing() -> None:
    html = """
        <div>
        <table>
            <tr><td>Вартість прокату</td></tr>
            <tr><td>1-й день</td><td>250 грн</td></tr>
            <tr><td>Застава</td></tr>
            <tr><td>4000 грн або Закордонний паспорт/права</td></tr>
        </table>
        </div>
        """

    result = extract_product_info_lines(html)

    assert result == []


def test_extract_section_lines_returns_lines_between_start_and_stop_markers() -> None:
    lines = [
        "Вартість прокату",
        "1-й день",
        "250 грн",
        "Застава",
        "4000 грн",
    ]

    result = extract_section_lines(
        lines,
        start_marker="Вартість прокату",
        stop_markers={"Застава"},
    )

    assert result == ["1-й день", "250 грн"]


def test_extract_section_lines_returns_lines_until_end_when_no_stop_marker_found() -> None:
    lines = [
        "Характеристики",
        "Розмір",
        "205 х 240 см",
    ]

    result = extract_section_lines(
        lines,
        start_marker="Характеристики",
        stop_markers=set(),
    )

    assert result == ["Розмір", "205 х 240 см"]


def test_extract_section_lines_returns_empty_list_when_start_marker_is_missing() -> None:
    lines = [
        "Вартість прокату",
        "1-й день",
        "250 грн",
    ]

    result = extract_section_lines(
        lines,
        start_marker="Характеристики",
        stop_markers=set(),
    )

    assert result == []


def test_parse_key_value_lines_returns_dict_from_pairs() -> None:
    lines = [
        "1-й день",
        "250 грн",
        "2-й день",
        "220 грн",
    ]

    result = parse_key_value_lines(lines)

    assert result == {
        "1-й день": "250 грн",
        "2-й день": "220 грн",
    }


def test_parse_key_value_lines_ignores_unpaired_last_key() -> None:
    lines = [
        "1-й день",
        "250 грн",
        "2-й день",
    ]

    result = parse_key_value_lines(lines)

    assert result == {
        "1-й день": "250 грн",
    }


def test_extract_rental_prices_from_lines_returns_rental_prices() -> None:
    lines = PRODUCT_INFO_LINES

    result = extract_rental_prices_from_lines(lines)

    assert result == {
        "1-й день": "250 грн",
        "2-й день": "220 грн",
        "3-й день": "190 грн",
    }


def test_extract_deposit_from_lines_returns_deposit() -> None:
    lines = PRODUCT_INFO_LINES

    result = extract_deposit_from_lines(lines)

    assert result == "4000 грн або Закордонний паспорт/права"


def test_extract_deposit_from_lines_returns_fallback_when_deposit_is_missing() -> None:
    lines = [
        "Вартість прокату",
        "1-й день",
        "250 грн",
    ]

    result = extract_deposit_from_lines(lines)

    assert result == "No product deposit found"


def test_extract_attributes_from_lines_returns_attributes() -> None:
    lines = PRODUCT_INFO_LINES

    result = extract_attributes_from_lines(lines)

    assert result == {
        "Розмір": "205 х 240 см",
        "Висота": "115 см",
        "Вага": "3.31 кг",
    }


def test_extract_product_rental_prices_returns_rental_prices_from_html() -> None:
    html = PRODUCT_INFO_HTML

    result = extract_product_rental_prices(html)

    assert result == {
        "1-й день": "250 грн",
        "2-й день": "220 грн",
    }


def test_extract_product_deposit_returns_deposit_from_html() -> None:
    html = PRODUCT_INFO_HTML

    result = extract_product_deposit(html)

    assert result == "4000 грн або Закордонний паспорт/права"


def test_extract_product_attributes_returns_attributes_from_html() -> None:
    html = PRODUCT_INFO_HTML

    result = extract_product_attributes(html)

    assert result == {
        "Розмір": "205 х 240 см",
        "Висота": "115 см",
    }


def test_build_product_from_html_returns_product() -> None:
    url = "https://prokat24.com.ua/ua/turisticheskoe-snaryazhenie/prokat-palatok/prokat-palatki-2-mestnaya/"
    html = """
    <html>
        <body>
            <h1 class="catalogue__product-name">
                Прокат 2-місних наметів
            </h1>
            <div class="short-descr">
                Прокат легких двомісних наметів Terra Incognita Alfa2.
            </div>
            <div class="product-info__table-v2">
                <table>
                    <tr>
                        <td>Вартість прокату</td>
                    </tr>
                    <tr>
                        <td>1-й день</td>
                        <td>250 грн</td>
                    </tr>
                    <tr>
                        <td>2-й день</td>
                        <td>220 грн</td>
                    </tr>
                    <tr>
                        <td>Застава</td>
                    </tr>
                    <tr>
                        <td>4000 грн або Закордонний паспорт/права</td>
                    </tr>
                    <tr>
                        <td>Характеристики</td>
                    </tr>
                    <tr>
                        <td>Розмір</td>
                        <td>205 х 240 см</td>
                    </tr>
                    <tr>
                        <td>Висота</td>
                        <td>115 см</td>
                    </tr>
                </table>
            </div>
        </body>
    </html>
    """

    result = build_product_from_html(url, html)

    assert result == {
        "url": url,
        "name": "Прокат 2-місних наметів",
        "description": "Прокат легких двомісних наметів Terra Incognita Alfa2.",
        "rental_prices": {
            "1-й день": "250 грн",
            "2-й день": "220 грн",
        },
        "deposit": "4000 грн або Закордонний паспорт/права",
        "attributes": {
            "Розмір": "205 х 240 см",
            "Висота": "115 см",
        },
    }
