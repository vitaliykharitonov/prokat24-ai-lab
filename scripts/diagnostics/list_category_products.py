from argparse import ArgumentParser, Namespace

from prokat24_ai_lab.products.catalog import build_products_from_category


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Inspect category products")
    parser.add_argument(
        "category_url",
        help="Category url",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    products = build_products_from_category(args.category_url)

    print(f"Found {len(products)} products")
    print()
    for index, product in enumerate(products, start=1):
        print(f"{index}. {product['name']}")
        print(f"URL: {product['url']}")
        print(f"Deposit: {product['deposit']}")
        print("Rental prices:")

        for lable, price in product["rental_prices"].items():
            print(f"{lable}: {price}")

        print()


if __name__ == "__main__":
    main()
