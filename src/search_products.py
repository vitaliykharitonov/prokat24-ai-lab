from argparse import ArgumentParser, Namespace

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
        print(f"Name: {product['name']}")
        print(f"URL: {product['url']}")
        print(f"Deposit: {product['deposit']}")
        print("Prices:")

        for lable, price in product["rental_prices"].items():
            print(f"{lable}: {price}")

        print()


if __name__ == "__main__":
    main()
