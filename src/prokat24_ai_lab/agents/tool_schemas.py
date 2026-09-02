from openai.types.responses import FunctionToolParam

GET_DELIVERY_INFO_TOOL: FunctionToolParam = {
    "type": "function",
    "name": "get_delivery_info",
    "description": (
        "Get delivery information for a specified location. "
        "Use this tool when the user asks about delivery availability, conditions, or cost. "
        "Do not use it for product search or rental price calculation."
    ),
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
    "description": (
        "Search for products and retrieve product information. "
        "Use this tool when the user wants to find or inspect a product. "
        "Do not use it to calculate rental prices."
    ),
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
    "description": (
        "Get the rental price of a specific product for a specified number of days. "
        "Use this tool when the user asks how much a product costs to rent. "
        "Do not use it for general product search."
    ),
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
