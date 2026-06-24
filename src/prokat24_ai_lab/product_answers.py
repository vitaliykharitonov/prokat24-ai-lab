from prokat24_ai_lab.product_catalog import Product


def format_product_search_result(product: Product) -> str:
    lines = [
        product["name"],
        f"Deposit: {product['deposit']}",
        "Prices:",
    ]

    for label, price in product["rental_prices"].items():
        lines.append(f"- {label}: {price}")

    lines.append(f"URL: {product['url']}")

    return "\n".join(lines)


def format_product_search_results(products: list[Product]) -> str:
    if not products:
        return "Products not found"

    formatted_products = []

    for product in products:
        formatted_products.append(format_product_search_result(product))

    return "\n\n".join(formatted_products)
