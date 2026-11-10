from prokat24_ai_lab.products.catalog import Product
from prokat24_ai_lab.products.catalog_search import search_products
from prokat24_ai_lab.products.presentation import (
    format_product_attribute_answer,
    format_product_availability_answer,
    format_product_day_count_rental_price_answer,
    format_product_deposit_answer,
    format_product_details_answer,
    format_product_rental_price_answer,
    format_product_search_results,
)
from prokat24_ai_lab.products.query import (
    build_product_search_query,
    detect_availability_intent,
    detect_deposit_intent,
    detect_product_details_intent,
    detect_rental_day_count,
    detect_rental_period,
    detect_requested_attribute_label,
)


def answer_product_query(products: list[Product], query: str, limit: int = 5) -> str:
    day_count = detect_rental_day_count(query)
    period = detect_rental_period(query)
    requested_attribute = detect_requested_attribute_label(query)
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

    if detect_availability_intent(query):
        return format_product_availability_answer(search_results[0])

    if requested_attribute is not None:
        return format_product_attribute_answer(search_results[0], requested_attribute)

    if detect_product_details_intent(query):
        return format_product_details_answer(search_results[0])

    return format_product_search_results(search_results)
