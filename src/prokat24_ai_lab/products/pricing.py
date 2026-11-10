import re

from prokat24_ai_lab.products.catalog import Product


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


def format_day_count_label(day_count: int) -> str:
    if 11 <= day_count <= 14:
        return f"{day_count} днів"

    if day_count % 10 == 1:
        return f"{day_count} день"

    if day_count % 10 in (2, 3, 4):
        return f"{day_count} дні"

    return f"{day_count} днів"
