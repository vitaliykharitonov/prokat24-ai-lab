from prokat24_ai_lab.product_catalog import Product
from prokat24_ai_lab.product_search import search_products
from prokat24_ai_lab.text_normalization import normalize_text

RENTAL_PERIOD_ALIASES = {
    "2 тижні": [
        "2 тижні",
        "два тижні",
    ],
    "Тиждень": [
        "на тиждень",
        "тиждень",
    ],
    "1-й день": [
        "1 день",
        "один день",
    ],
}

PRODUCT_QUERY_STOP_WORDS = {
    "скільки",
    "коштує",
    "вартість",
    "ціна",
    "за",
    "що",
    "на",
    "в",
    "оренда",
    "прокат",
    "тиждень",
    "тижні",
    "застава",
    "заставу",
    "який",
    "яка",
    "яку",
    "яке",
}

DEPOSIT_INTENT_WORDS = {
    "застава",
    "заставу",
}


def remove_word_sequence(words: list[str], sequence: list[str]) -> list[str]:
    sequence_length = len(sequence)

    for start_index in range(len(words)):
        end_index = start_index + sequence_length

        if words[start_index:end_index] == sequence:
            return words[:start_index] + words[end_index:]

    return words


def build_product_search_query(query: str) -> str:
    words = normalize_text(query).split()
    period = detect_rental_period(query)
    words = remove_rental_period_words(words, period)

    product_words = []

    for word in words:
        if word in PRODUCT_QUERY_STOP_WORDS:
            continue

        product_words.append(word)

    return " ".join(product_words)


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
    period = detect_rental_period(query)
    search_query = build_product_search_query(query)
    search_results = search_products(products, search_query, limit=limit)

    if not search_results:
        return format_product_search_results(search_results)

    if detect_deposit_intent(query):
        return format_product_deposit_answer(search_results[0])

    if period is not None:
        return format_product_rental_price_answer(search_results[0], period)

    return format_product_search_results(search_results)


def detect_rental_period(query: str) -> str | None:
    normalized_query = normalize_text(query)

    for period, aliases in RENTAL_PERIOD_ALIASES.items():
        for alias in aliases:
            if normalize_text(alias) in normalized_query:
                return period

    return None


def remove_rental_period_words(words: list[str], period: str | None) -> list[str]:
    if period is None:
        return words

    aliases = RENTAL_PERIOD_ALIASES[period]

    for alias in aliases:
        alias_words = normalize_text(alias).split()
        updated_words = remove_word_sequence(words, alias_words)

        if updated_words != words:
            return updated_words

    return words


def format_product_rental_price_answer(product: Product, period: str) -> str:
    price = product["rental_prices"].get(period)

    if price is None:
        return f"Price for {period} not found"

    lines = [
        product["name"],
        f"{period}: {price}",
        f"Deposit: {product['deposit']}",
        f"URL: {product['url']}",
    ]

    return "\n".join(lines)


def detect_deposit_intent(query: str) -> bool:
    query_words = set(normalize_text(query).split())

    return bool(query_words & DEPOSIT_INTENT_WORDS)


def format_product_deposit_answer(product: Product) -> str:
    lines = [
        product["name"],
        f"Deposit: {product['deposit']}",
        f"URL: {product['url']}",
    ]

    return "\n".join(lines)
