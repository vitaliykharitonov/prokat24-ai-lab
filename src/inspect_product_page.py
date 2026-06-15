from argparse import ArgumentParser, Namespace

from bs4 import BeautifulSoup

from prokat24_ai_lab.html_parsing import (
    extract_title,
)
from prokat24_ai_lab.product_catalog import (
    extract_product_description,
    extract_product_name,
)
from prokat24_ai_lab.prokat24_client import fetch_page


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Inspect html from a product page")
    parser.add_argument(
        "product_url",
        help="Product url",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    html = fetch_page(args.product_url)
    title = extract_title(html)
    product_name = extract_product_name(html)
    product_description = extract_product_description(html)

    soup = BeautifulSoup(html, "html.parser")
    tables = soup.select(".product-info__table-v2, .product-info__table")

    print(f"Title: {title}")
    print(f"Product name: {product_name}")
    print("Product description: ")
    print(f"{product_description}")

    print(f"Found {len(tables)} product info table blocks")

    for index, table in enumerate(tables, start=1):
        print(f"--- TABLE {index} ---")
        table_text = table.get_text(separator="\n", strip=True)
        print(table_text)


if __name__ == "__main__":
    main()
