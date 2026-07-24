from argparse import ArgumentParser, Namespace

from prokat24_ai_lab.products.answers import (
    answer_product_query,
)
from prokat24_ai_lab.products.catalog import load_products
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

    answer = answer_product_query(products, args.query, limit=args.limit)

    print(answer)


if __name__ == "__main__":
    main()
