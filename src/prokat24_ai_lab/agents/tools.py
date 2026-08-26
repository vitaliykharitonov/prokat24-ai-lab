from openai.types.responses import FunctionToolParam

GET_DELIVERY_INFO_TOOL: FunctionToolParam = {
    "type": "function",
    "name": "get_delivery_info",
    "description": "Get delivery information for a location",
    "strict": False,
    "parameters": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
            },
        },
        "required": ["location"],
    },
}

SEARCH_PRODUCTS_TOOL: FunctionToolParam = {
    "type": "function",
    "name": "search_products",
    "description": "Search product information",
    "strict": False,
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
            },
        },
        "required": ["query"],
    },
}

GET_PRODUCT_PRICE_TOOL: FunctionToolParam = {
    "type": "function",
    "name": "get_product_price",
    "description": "Get product prices",
    "strict": False,
    "parameters": {
        "type": "object",
        "properties": {
            "product": {
                "type": "string",
            },
            "days": {
                "type": "integer",
            },
        },
        "required": ["product", "days"],
    },
}

TOOLS: list[FunctionToolParam] = [
    GET_DELIVERY_INFO_TOOL,
    SEARCH_PRODUCTS_TOOL,
    GET_PRODUCT_PRICE_TOOL,
]
