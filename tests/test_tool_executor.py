import pytest

from prokat24_ai_lab.agents.tool_executor import create_tool_executor
from prokat24_ai_lab.agents.types import ToolCall
from prokat24_ai_lab.products.catalog import Product

TEST_PRODUCTS: list[Product] = [
    {
        "url": "https://example.com/tent/",
        "name": "Намет 2-місний",
        "description": "Туристичний двомісний намет",
        "rental_prices": {
            "1-й день": "300 грн",
            "2-й день": "250 грн",
            "3-й день": "200 грн",
        },
        "deposit": "1000 грн",
        "attributes": {},
    }
]


def test_tool_executor_executes_product_search() -> None:
    tool_call: ToolCall = {
        "tool_name": "search_products",
        "arguments": {"query": "намет"},
    }

    tool_executor = create_tool_executor(TEST_PRODUCTS)

    result = tool_executor(tool_call)

    assert "Намет 2-місний" in result


def test_tool_executor_raises_for_unsupported_tool() -> None:
    tool_call: ToolCall = {
        "tool_name": "unknown_tool",
        "arguments": {"query": "намет"},
    }

    tool_executor = create_tool_executor(TEST_PRODUCTS)

    with pytest.raises(ValueError, match="Unsupported tool: unknown_tool"):
        tool_executor(tool_call)


def test_tool_executor_raises_when_product_search_query_is_not_string() -> None:
    tool_call: ToolCall = {
        "tool_name": "search_products",
        "arguments": {"query": 123},
    }

    tool_executor = create_tool_executor(TEST_PRODUCTS)

    with pytest.raises(ValueError, match="Query must be a string"):
        tool_executor(tool_call)


def test_tool_executor_executes_product_price_lookup() -> None:
    tool_call: ToolCall = {
        "tool_name": "get_product_price",
        "arguments": {
            "product_url": "https://example.com/tent/",
            "days": 3,
        },
    }

    tool_executor = create_tool_executor(TEST_PRODUCTS)

    result = tool_executor(tool_call)

    assert "Намет 2-місний" in result
    assert "750 грн" in result


def test_tool_executor_raises_when_product_not_found() -> None:
    tool_call: ToolCall = {
        "tool_name": "get_product_price",
        "arguments": {
            "product_url": "https://example.com/missing/",
            "days": 3,
        },
    }

    tool_executor = create_tool_executor(TEST_PRODUCTS)

    with pytest.raises(ValueError, match="Product not found"):
        tool_executor(tool_call)


def test_tool_executor_raises_when_product_url_is_not_string() -> None:
    tool_call: ToolCall = {
        "tool_name": "get_product_price",
        "arguments": {
            "product_url": 123,
            "days": 3,
        },
    }

    tool_executor = create_tool_executor(TEST_PRODUCTS)

    with pytest.raises(ValueError, match="Product URL must be a string"):
        tool_executor(tool_call)


def test_tool_executor_raises_when_days_is_not_integer() -> None:
    tool_call: ToolCall = {
        "tool_name": "get_product_price",
        "arguments": {
            "product_url": "https://example.com/tent/",
            "days": "3",
        },
    }

    tool_executor = create_tool_executor(TEST_PRODUCTS)

    with pytest.raises(ValueError, match="Days must be integer"):
        tool_executor(tool_call)
