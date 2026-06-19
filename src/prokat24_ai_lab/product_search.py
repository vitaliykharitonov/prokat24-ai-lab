from prokat24_ai_lab.product_catalog import Product


def build_product_search_text(product: Product) -> str:
    parts = [
        product["name"],
        product["description"],
        product["deposit"],
    ]

    for label, price in product["rental_prices"].items():
        parts.append(f"{label}: {price}")

    for label, value in product["attributes"].items():
        parts.append(f"{label}: {value}")

    return " ".join(parts)
