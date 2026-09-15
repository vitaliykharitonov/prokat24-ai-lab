from prokat24_ai_lab.agents.types import ToolArguments, ToolDefinition, ToolHandler, ToolRegistry
from prokat24_ai_lab.knowledge.base import KnowledgePage
from prokat24_ai_lab.knowledge.keyword_search import search_knowledge_base
from prokat24_ai_lab.products.answers import (
    format_product_day_count_rental_price_answer,
    format_product_search_results,
)
from prokat24_ai_lab.products.catalog import Product
from prokat24_ai_lab.products.catalog_search import search_products

SEARCH_PRODUCTS_DEFINITION: ToolDefinition = {
    "name": "search_products",
    "description": (
        "Search the Prokat24 rental catalog and return matching products "
        "with rental prices, deposit information, and product URLs."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Product search query.",
            },
        },
        "required": ["query"],
    },
}

GET_PRODUCT_PRICE_DEFINITION: ToolDefinition = {
    "name": "get_product_price",
    "description": (
        "Calculate the rental price of a specific Prokat24 product for a specified number of days."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "product_url": {
                "type": "string",
                "description": "URL of the specific product.",
            },
            "days": {
                "type": "integer",
                "description": "Number of rental days.",
            },
        },
        "required": ["product_url", "days"],
    },
}

SEARCH_SERVICE_INFO_DEFINITION: ToolDefinition = {
    "name": "search_service_info",
    "description": (
        "Search Prokat24 service information such as delivery, booking, "
        "rental rules, contacts, addresses, and other customer service details."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Service information search query.",
            },
        },
        "required": ["query"],
    },
}


def create_search_products_handler(
    products: list[Product],
) -> ToolHandler:
    def handler(arguments: ToolArguments) -> str:
        query = arguments["query"]

        if not isinstance(query, str):
            raise ValueError("Query must be a string")

        search_result = search_products(products, query)
        return format_product_search_results(search_result)

    return handler


def create_get_product_price_handler(
    products: list[Product],
) -> ToolHandler:
    def handler(arguments: ToolArguments) -> str:
        product_url = arguments["product_url"]
        days = arguments["days"]

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

    return handler


def create_search_service_info_handler(pages: list[KnowledgePage]) -> ToolHandler:
    def handler(arguments: ToolArguments) -> str:
        query = arguments["query"]

        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string")

        search_results = search_knowledge_base(query, pages)

        if not search_results:
            return "No relevant service information found"

        top_result = search_results[0]

        page = next(
            (page for page in pages if page["url"] == top_result["url"]),
            None,
        )

        if page is None:
            raise ValueError("Knowledge page not found")

        lines: list[str] = []

        lines.append(f"Title: {page['title']}")
        lines.append(f"Information:\n{page['text']}")
        lines.append(f"URL: {page['url']}")

        return "\n".join(lines)

    return handler


def create_tool_registry(products: list[Product], pages: list[KnowledgePage]) -> ToolRegistry:
    return {
        "search_products": {
            "definition": SEARCH_PRODUCTS_DEFINITION,
            "handler": create_search_products_handler(products),
        },
        "get_product_price": {
            "definition": GET_PRODUCT_PRICE_DEFINITION,
            "handler": create_get_product_price_handler(products),
        },
        "search_service_info": {
            "definition": SEARCH_SERVICE_INFO_DEFINITION,
            "handler": create_search_service_info_handler(pages),
        },
    }
