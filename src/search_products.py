from argparse import ArgumentParser, Namespace

from prokat24_ai_lab.product_answers import format_product_search_result
from prokat24_ai_lab.product_catalog import load_products
from prokat24_ai_lab.product_search import search_products
from prokat24_ai_lab.project_paths import PRODUCTS_FILE


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Search Prokat24 products")
    parser.add_argument("query", help="Search query")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum number of products to show",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    products = load_products(PRODUCTS_FILE)
    results = search_products(products, args.query, args.limit)

    for product in results:
        print(format_product_search_result(product))
        print()


if __name__ == "__main__":
    main()
