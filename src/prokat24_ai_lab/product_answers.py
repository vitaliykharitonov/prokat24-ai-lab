from prokat24_ai_lab.product_catalog import Product
from prokat24_ai_lab.product_search import search_products
from prokat24_ai_lab.text_normalization import normalize_text

RENTAL_PERIOD_ALIASES = {
    "2 тижні": [
        "2 тижні",
        "2 недели",
        "две недели",
    ],
    "Тиждень": [
        "тиждень",
        "неделю",
        "неделя",
    ],
    "1-й день": [
        "1 день",
        "первый день",
        "перший день",
    ],
}


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


def answer_product_query(products: list[Product], query: str, limit: int = 5) -> str:
    search_results = search_products(products, query, limit=limit)

    return format_product_search_results(search_results)


def detect_rental_period(query: str) -> str | None:
    normalized_query = normalize_text(query)

    for period, aliases in RENTAL_PERIOD_ALIASES.items():
        for alias in aliases:
            if normalize_text(alias) in normalized_query:
                return period
