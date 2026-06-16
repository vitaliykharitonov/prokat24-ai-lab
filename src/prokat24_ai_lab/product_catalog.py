from typing import TypedDict
from urllib.parse import urlparse

from bs4 import BeautifulSoup

from prokat24_ai_lab.prokat24_client import fetch_page

PRODUCT_CATEGORY_ROOT_PATHS = {
    "/ua/gornolyzhnoe-snaryazhenie/",
    "/ua/prokat-mebeli/",
    "/ua/prokat-razvlechenij/",
    "/ua/rybolovnoe-snaryazhenie/",
    "/ua/turisticheskoe-snaryazhenie/",
    "/ua/velosipedy-i-roliki/",
}


class Product(TypedDict):
    url: str
    name: str
    description: str
    rental_price: dict[str, str]
    deposit: str
    attributes: dict[str, str]


def is_product_category_url(url: str) -> bool:
    parsed = urlparse(url)

    if parsed.netloc != "prokat24.com.ua":
        return False

    if not parsed.path.endswith("/"):
        return False

    return any(parsed.path.startswith(root_path) for root_path in PRODUCT_CATEGORY_ROOT_PATHS)


def filter_product_category_urls(urls: list[str]) -> list[str]:
    filtered_urls = []

    for url in urls:
        if is_product_category_url(url):
            filtered_urls.append(url)

    return sorted(set(filtered_urls))


def is_product_url(url: str, category_url: str) -> bool:
    parsed = urlparse(url)

    if not url.startswith(category_url):
        return False

    if url == category_url:
        return False

    if not parsed.path.endswith("/"):
        return False

    return parsed.fragment == ""


def filter_product_urls(category_url: str, urls: list[str]) -> list[str]:
    filtered_urls = []
    for url in urls:
        if is_product_url(url, category_url):
            filtered_urls.append(url)

    return sorted(set(filtered_urls))


def extract_text_by_selector(html: str, selector: str, fallback: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    tag = soup.select_one(selector)

    if tag is None:
        return fallback

    text = tag.get_text(strip=True)

    if not text:
        return fallback

    return text


def extract_product_name(html: str) -> str:
    return extract_text_by_selector(
        html,
        selector=".catalogue__product-name",
        fallback="No product name found",
    )


def extract_product_description(html: str) -> str:
    return extract_text_by_selector(
        html,
        selector=".short-descr",
        fallback="No product description found",
    )


def extract_product_info_lines(html: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    table = soup.select_one(".product-info__table-v2")

    if table is None:
        return []

    text = table.get_text(separator="\n", strip=True)

    return text.splitlines()


def extract_section_lines(
    lines: list[str],
    start_marker: str,
    stop_markers: set[str],
) -> list[str]:
    section_lines = []
    is_collecting = False

    for line in lines:
        if line == start_marker:
            is_collecting = True
            continue

        if is_collecting and line in stop_markers:
            break

        if is_collecting:
            section_lines.append(line)

    return section_lines


def parse_key_value_lines(lines: list[str]) -> dict[str, str]:
    results = {}

    for index in range(0, len(lines), 2):
        key = lines[index]

        if index + 1 >= len(lines):
            break

        value = lines[index + 1]

        results[key] = value

    return results


def extract_rental_prices_from_lines(lines: list[str]) -> dict[str, str]:
    extracted_section = extract_section_lines(
        lines,
        start_marker="Вартість прокату",
        stop_markers={"Застава", "Характеристики"},
    )

    if not extracted_section:
        return "No product prices found"

    return parse_key_value_lines(extracted_section)


def extract_deposit_from_lines(lines: list[str]) -> str:
    extracted_section = extract_section_lines(
        lines,
        start_marker="Застава",
        stop_markers={"Характеристики"},
    )

    if not extracted_section:
        return "No product deposit found"

    return extracted_section[0]


def extract_attributes_from_lines(lines: list[str]) -> dict[str, str]:
    extracted_section = extract_section_lines(
        lines,
        start_marker="Характеристики",
        stop_markers=set(),
    )

    if not extracted_section:
        return "No product attributes found"

    return parse_key_value_lines(extracted_section)


def extract_product_rental_prices(html: str) -> dict[str, str]:
    lines = extract_product_info_lines(html)

    return extract_rental_prices_from_lines(lines)


def extract_product_deposit(html: str) -> str:
    lines = extract_product_info_lines(html)

    return extract_deposit_from_lines(lines)


def extract_product_attributes(html: str) -> dict[str, str]:
    lines = extract_product_info_lines(html)

    return extract_attributes_from_lines(lines)


def build_product_from_html(url: str, html: str) -> Product:
    name = extract_product_name(html)
    description = extract_product_description(html)
    rental_prices = extract_product_rental_prices(html)
    deposit = extract_product_deposit(html)
    attributes = extract_product_attributes(html)

    product: Product = {
        "url": url,
        "name": name,
        "description": description,
        "rental_prices": rental_prices,
        "deposit": deposit,
        "attributes": attributes,
    }

    return product


def build_product(url: str) -> Product:
    html = fetch_page(url)
    product = build_product_from_html(url, html)

    return product
