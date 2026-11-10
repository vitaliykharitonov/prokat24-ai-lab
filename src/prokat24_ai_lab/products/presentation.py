from prokat24_ai_lab.products.catalog import Product
from prokat24_ai_lab.products.pricing import (
    calculate_rental_price_for_day_count,
    format_day_count_label,
)


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


def format_product_deposit_answer(product: Product) -> str:
    lines = [
        product["name"],
        f"Застава: {product['deposit']}",
        f"URL: {product['url']}",
    ]

    return "\n".join(lines)


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


def format_product_availability_answer(product: Product) -> str:
    lines = [product["name"], "Актуальну наявність краще уточнити у менеджера"]

    return "\n".join(lines)


def format_product_details_answer(product: Product) -> str:
    lines = [product["name"], product["description"], "Характеристики:"]

    for label, value in product["attributes"].items():
        lines.append(f"{label}: {value}")

    return "\n".join(lines)


def format_product_attribute_answer(product: Product, attribute_label: str) -> str:
    attribute_value = product["attributes"].get(attribute_label)

    if attribute_value is None:
        return f"{product['name']}\nХарактеристику {attribute_label} не знайдено"

    lines = [
        product["name"],
        f"{attribute_label}: {attribute_value}",
    ]

    return "\n".join(lines)
