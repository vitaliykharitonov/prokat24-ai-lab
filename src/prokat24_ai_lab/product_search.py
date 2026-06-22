from prokat24_ai_lab.product_catalog import Product
from prokat24_ai_lab.text_normalization import normalize_text

PRODUCT_QUERY_ALIASES: dict[str, set[str]] = {
    "палатка": {"палатки", "намет", "наметів", "намети"},
    "спальника": {"спальник", "спальників", "спальники"},
    "черевики": {"черевиків", "ботинки", "ботинок"},
    "човен": {"човна", "човну", "човни"},
}


def build_query_word_groups(query_words: list[str]) -> list[set[str]]:
    query_word_groups = []

    for word in query_words:
        group = {word}
        aliases = PRODUCT_QUERY_ALIASES.get(word, set())
        group.update(aliases)
        query_word_groups.append(group)

    return query_word_groups


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


def contains_all_word_groups(text: str, query_words: list[str]) -> bool:
    words = set(normalize_text(text).split())
    groups = build_query_word_groups(query_words)

    return all(bool(group & words) for group in groups)


def product_matches_query(product: Product, query_words: list[str]) -> bool:
    if contains_all_word_groups(product["name"], query_words):
        return True

    if contains_all_word_groups(product["description"], query_words):
        return True

    attributes_parts = []

    for label, value in product["attributes"].items():
        attributes_parts.append(f"{label}: {value}")

    attributes_text = " ".join(attributes_parts)

    return contains_all_word_groups(attributes_text, query_words)


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
        if product_matches_query(product, query_words):
            score = score_product(product, query_words)
            scored_results.append((score, index, product))

    scored_results.sort(key=lambda item: (-item[0], item[1]))

    return [product for _score, _index, product in scored_results[:limit]]
