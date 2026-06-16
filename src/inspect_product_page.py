from argparse import ArgumentParser, Namespace

from prokat24_ai_lab.html_parsing import (
    extract_title,
)
from prokat24_ai_lab.product_catalog import build_product_from_html
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

    product = build_product_from_html(args.product_url, html)

    print(f"Title: {title}")
    print(f"Product name: {product['name']}")
    print("Product description:")
    print(f"{product['description']}")
    print()
    print("Rental prices:")

    for label, price in product["rental_prices"].items():
        print(f"{label}: {price}")

    print()
    print("Deposit:")
    print(f"{product['deposit']}")
    print()
    print("Attributes:")

    for label, value in product["attributes"].items():
        print(f"{label}: {value}")


if __name__ == "__main__":
    main()
