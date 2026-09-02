from prokat24_ai_lab.agents.types import ToolCall, ToolExecutor
from prokat24_ai_lab.products.answers import (
    format_product_day_count_rental_price_answer,
    format_product_search_results,
)
from prokat24_ai_lab.products.catalog import Product
from prokat24_ai_lab.products.catalog_search import search_products


def create_tool_executor(products: list[Product]) -> ToolExecutor:
    def execute_tool_call(tool_call: ToolCall) -> str:
        if tool_call["tool_name"] == "search_products":
            query = tool_call["arguments"]["query"]

            if not isinstance(query, str):
                raise ValueError("Query must be a string")

            search_result = search_products(products, query)
            formatted_products = format_product_search_results(search_result)

            return formatted_products

        if tool_call["tool_name"] == "get_product_price":
            product_url = tool_call["arguments"]["product_url"]
            days = tool_call["arguments"]["days"]

            if not isinstance(product_url, str):
                raise ValueError("Product URL must be a string")

            if not isinstance(days, int):
                raise ValueError("Days must be integer")

            product = next(
                (product for product in products if product["url"] == product_url),
                None,
            )

            if product is None:
                raise ValueError("Product not found")

            return format_product_day_count_rental_price_answer(product, days)

        raise ValueError(f"Unsupported tool: {tool_call['tool_name']}")

    return execute_tool_call
