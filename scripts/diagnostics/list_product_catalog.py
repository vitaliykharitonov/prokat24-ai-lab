from argparse import ArgumentParser, Namespace

from prokat24_ai_lab.html_parsing import extract_internal_links
from prokat24_ai_lab.products.catalog import (
    build_product_catalog,
    filter_leaf_product_category_urls,
    filter_product_category_urls,
)
from prokat24_ai_lab.prokat24_client import fetch_page
from prokat24_ai_lab.site_config import ALLOWED_DOMAIN, BASE_URL


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Category list limit")
    parser.add_argument(
        "--limit",
        type=int,
        default=2,
        help="Maximum number of categories to load",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    html = fetch_page(BASE_URL)
    links = extract_internal_links(html, BASE_URL, ALLOWED_DOMAIN)
    category_urls = filter_product_category_urls(links)
    leaf_category_urls = filter_leaf_product_category_urls(category_urls)
    selected_category_urls = leaf_category_urls[: args.limit]
    products = build_product_catalog(selected_category_urls)

    print(f"Found {len(selected_category_urls)} leaf categories")
    print(f"Found {len(products)} products")
    print()

    for index, product in enumerate(products, start=1):
        print(f"{index}. {product['name']}")
        print(f"URL: {product['url']}")
        print(f"Deposit: {product['deposit']}")
        print()


if __name__ == "__main__":
    main()
