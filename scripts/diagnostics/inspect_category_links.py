from argparse import ArgumentParser, Namespace

from prokat24_ai_lab.html_parsing import extract_internal_links
from prokat24_ai_lab.products.catalog import filter_product_urls
from prokat24_ai_lab.prokat24_client import fetch_page
from prokat24_ai_lab.site_config import ALLOWED_DOMAIN


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Inspect links from a product category page")
    parser.add_argument(
        "category_url",
        help="Category url",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    html = fetch_page(args.category_url)
    links = extract_internal_links(html, args.category_url, ALLOWED_DOMAIN)

    product_urls = filter_product_urls(args.category_url, links)

    print(f"Found {len(product_urls)} product urls")
    print()

    for link in product_urls:
        print(link)


if __name__ == "__main__":
    main()
