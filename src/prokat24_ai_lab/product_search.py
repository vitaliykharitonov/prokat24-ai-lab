from prokat24_ai_lab.product_catalog import Product
from prokat24_ai_lab.text_normalization import normalize_text


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


def search_products(
    products: list[Product],
    query: str,
    limit: int = 5,
) -> list[Product]:
    query_words = normalize_text(query).split()

    if not query_words or limit <= 0:
        return []

    results: list[Product] = []

    for product in products:
        product_text = normalize_text(build_product_search_text(product))
        product_words = set(product_text.split())

        if all(word in product_words for word in query_words):
            results.append(product)

        if len(results) >= limit:
            break

    return results
