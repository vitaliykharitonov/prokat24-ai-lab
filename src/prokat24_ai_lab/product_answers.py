from prokat24_ai_lab.product_catalog import Product


def format_product_search_result(product: Product) -> str:
    lines = [
        product["name"],
        f"Залог: {product['deposit']}",
        "Цены:",
    ]

    for label, price in product["rental_prices"].items():
        lines.append(f"- {label}: {price}")

    lines.append(f"URL: {product['url']}")

    return "\n".join(lines)
