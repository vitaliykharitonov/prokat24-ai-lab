import re

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

RENTAL_DAY_WORDS = {
    "день",
    "дні",
    "дня",
    "днів",
}

UKRAINIAN_NUMBER_WORDS = {
    "один": 1,
    "два": 2,
    "три": 3,
    "чотири": 4,
    "пʼять": 5,
    "п'ять": 5,
    "шість": 6,
}

RENTAL_DAY_WORDS: set[str]


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
    day_count = detect_rental_day_count(query)

    words = remove_rental_period_words(words, period)
    words = remove_rental_day_count_words(words, day_count)

    product_words = []

    for word in words:
        if word in PRODUCT_QUERY_STOP_WORDS:
            continue

        product_words.append(word)

    return " ".join(product_words)


def format_product_search_result(product: Product) -> str:
    lines = [
        product["name"],
        f"Застава: {product['deposit']}",
        "Ціни:",
    ]

    for label, price in product["rental_prices"].items():
        lines.append(f"- {label}: {price}")

    lines.append(f"URL: {product['url']}")

    return "\n".join(lines)


def format_product_search_results(products: list[Product]) -> str:
    if not products:
        return "Товари не знайдено"

    formatted_products = []

    for product in products:
        formatted_products.append(format_product_search_result(product))

    return "\n\n".join(formatted_products)


def answer_product_query(products: list[Product], query: str, limit: int = 5) -> str:
    day_count = detect_rental_day_count(query)
    period = detect_rental_period(query)
    search_query = build_product_search_query(query)
    search_results = search_products(products, search_query, limit=limit)

    if not search_results:
        return format_product_search_results(search_results)

    if detect_deposit_intent(query):
        return format_product_deposit_answer(search_results[0])

    if day_count is not None:
        return format_product_day_count_rental_price_answer(search_results[0], day_count)

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


def detect_rental_day_count(query: str) -> int | None:
    query_words = normalize_text(query).split()

    for index in range(len(query_words) - 1):
        current_word = query_words[index]
        next_word = query_words[index + 1]

        if next_word not in RENTAL_DAY_WORDS:
            continue

        if current_word.isdigit():
            return int(current_word)

        if current_word in UKRAINIAN_NUMBER_WORDS:
            return UKRAINIAN_NUMBER_WORDS[current_word]

    return None


def parse_price_amount(price: str) -> int | None:
    match = re.search(r"\d[\d ]*", price)

    if match is None:
        return None

    price_amount = int(match.group().replace(" ", ""))

    return price_amount


def calculate_rental_price_for_day_count(product: Product, day_count: int) -> int | None:
    daily_price_labels = [
        "1-й день",
        "2-й день",
        "3-й день",
    ]
    total_price = 0

    if day_count <= 0:
        return None

    for label in daily_price_labels[:day_count]:
        price_str = product["rental_prices"].get(label)

        if price_str is None:
            return None

        price_amount = parse_price_amount(price_str)

        if price_amount is None:
            return None

        total_price += price_amount

    if day_count > 3:
        extra_price_str = product["rental_prices"].get("з 4-го дня")

        if extra_price_str is None:
            return None

        extra_price_amount = parse_price_amount(extra_price_str)

        if extra_price_amount is None:
            return None

        total_price += extra_price_amount * (day_count - 3)

    return total_price


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
        return f"Ціну для періоду {period} не знайдено"

    lines = [
        product["name"],
        f"{period}: {price}",
        f"Застава: {product['deposit']}",
        f"URL: {product['url']}",
    ]

    return "\n".join(lines)


def detect_deposit_intent(query: str) -> bool:
    query_words = set(normalize_text(query).split())

    return bool(query_words & DEPOSIT_INTENT_WORDS)


def format_product_deposit_answer(product: Product) -> str:
    lines = [
        product["name"],
        f"Застава: {product['deposit']}",
        f"URL: {product['url']}",
    ]

    return "\n".join(lines)


def format_day_count_label(day_count: int) -> str:
    if day_count == 1:
        return f"{day_count} день"

    if 1 < day_count <= 4:
        return f"{day_count} дні"

    return f"{day_count} днів"


def format_product_day_count_rental_price_answer(product: Product, day_count: int) -> str:
    price = calculate_rental_price_for_day_count(product, day_count)
    day_count_label = format_day_count_label(day_count)

    if price is None:
        return f"Ціну на {day_count_label} не знайдено"

    lines = [
        product["name"],
        f"{day_count_label}: {price} грн",
        f"Застава: {product['deposit']}",
        f"URL: {product['url']}",
    ]
    return "\n".join(lines)


def remove_rental_day_count_words(words: list[str], day_count: int | None) -> list[str]:
    if day_count is None:
        return words

    for index in range(len(words) - 1):
        current_word = words[index]
        next_word = words[index + 1]

        if next_word not in RENTAL_DAY_WORDS:
            continue

        if current_word.isdigit() and int(current_word) == day_count:
            return remove_word_sequence(words, [current_word, next_word])

        if current_word in UKRAINIAN_NUMBER_WORDS:
            current_number = UKRAINIAN_NUMBER_WORDS[current_word]

        if current_number == day_count:
            return remove_word_sequence(words, [current_word, next_word])

    return words
