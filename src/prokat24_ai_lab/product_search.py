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


def build_product_match_text(product: Product) -> str:
    parts = [
        product["name"],
        product["description"],
    ]

    for label, value in product["attributes"].items():
        parts.append(f"{label}: {value}")

    return " ".join(parts)


def score_product(product: Product, query_words: list[str]) -> int:
    name_text = normalize_text(product["name"])
    name_words = set(name_text.split())
    description_text = normalize_text(product["description"])
    description_words = set(description_text.split())
    attributes_parts = []

    for label, value in product["attributes"].items():
        attributes_parts.append(f"{label}: {value}")

    attributes_text = normalize_text(" ".join(attributes_parts))
    attributes_words = set(attributes_text.split())
    prices_parts = []

    for label, price in product["rental_prices"].items():
        prices_parts.append(f"{label}: {price}")

    prices_text = normalize_text(" ".join(prices_parts))
    prices_words = set(prices_text.split())
    deposit_text = normalize_text(product["deposit"])
    deposit_words = set(deposit_text.split())
    score = 0

    for word in query_words:
        if word in name_words:
            score += 10

        if word in description_words:
            score += 4

        if word in attributes_words:
            score += 3

        if word in prices_words:
            score += 1

        if word in deposit_words:
            score += 1

    return score


def search_products(
    products: list[Product],
    query: str,
    limit: int = 5,
) -> list[Product]:
    query_words = normalize_text(query).split()

    if not query_words or limit <= 0:
        return []

    scored_results: list[tuple[int, int, Product]] = []

    for index, product in enumerate(products, start=1):
        match_text = normalize_text(build_product_match_text(product))
        product_words = set(match_text.split())

        if all(word in product_words for word in query_words):
            score = score_product(product, query_words)
            scored_results.append((score, index, product))

    scored_results.sort(key=lambda item: (-item[0], item[1]))

    return [product for _score, _index, product in scored_results[:limit]]
