import pytest

from prokat24_ai_lab.agents.tool_registry import (
    create_search_service_info_handler,
    create_tool_registry,
)
from prokat24_ai_lab.agents.types import ToolArguments
from prokat24_ai_lab.knowledge.base import KnowledgePage
from prokat24_ai_lab.products.catalog import Product

DELIVERY_TEXT = (
    "Доставка по Києву можлива за попередньою домовленістю. "
    "Вартість доставки по Києву — від 200 грн. "
    "За межами Києва — від 200 грн + 10 грн/км."
)

CONTACTS_TEXT = (
    "Пункт прокату знаходиться у Києві. Для зв'язку з менеджером використовуйте телефон."
)

TEST_PAGES: list[KnowledgePage] = [
    {
        "url": "https://example.com/delivery/",
        "title": "Доставка",
        "text": DELIVERY_TEXT,
        "text_length": len(DELIVERY_TEXT),
    },
    {
        "url": "https://example.com/contacts/",
        "title": "Контакти",
        "text": CONTACTS_TEXT,
        "text_length": len(CONTACTS_TEXT),
    },
]

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


def test_search_service_info_handler_returns_relevant_information() -> None:
    handler = create_search_service_info_handler(TEST_PAGES)

    result = handler({"query": "доставка Київ"})

    assert "Доставка" in result
    assert "200 грн" in result


def test_search_service_info_handler_raises_when_query_is_not_string() -> None:
    handler = create_search_service_info_handler(TEST_PAGES)

    with pytest.raises(ValueError, match="Query must be a non-empty string"):
        handler({"query": 123})


def test_create_tool_registry_registers_available_tools() -> None:
    registry = create_tool_registry(TEST_PRODUCTS, TEST_PAGES)

    assert set(registry) == {
        "search_products",
        "get_product_price",
        "search_service_info",
    }

    for tool_name, registered_tool in registry.items():
        assert registered_tool["definition"]["name"] == tool_name


def test_search_service_info_handler_returns_all_knowledge_pages() -> None:
    pages: list[KnowledgePage] = [
        {
            "url": "https://prokat24.com.ua/ua/dostavka/",
            "title": "Доставка",
            "text": ("По Києву — від 200 грн. За межами Києва — від 200 грн + 10 грн/км."),
            "text_length": 67,
        },
        {
            "url": "https://prokat24.com.ua/ua/contact-us/",
            "title": "Контакти",
            "text": ("м. БЕРЕСТЕЙСЬКА, вул. Табірна, 42. Телефон: (095) 66 06 181."),
            "text_length": 71,
        },
    ]
    arguments: ToolArguments = {"query": "Скільки коштує доставка?"}

    handler = create_search_service_info_handler(pages)

    result = handler(arguments)

    expected = (
        "[Document 1]\n"
        "Title: Доставка\n"
        "Information:\n"
        "По Києву — від 200 грн. "
        "За межами Києва — від 200 грн + 10 грн/км.\n"
        "URL: https://prokat24.com.ua/ua/dostavka/\n"
        "\n"
        "[Document 2]\n"
        "Title: Контакти\n"
        "Information:\n"
        "м. БЕРЕСТЕЙСЬКА, вул. Табірна, 42. "
        "Телефон: (095) 66 06 181.\n"
        "URL: https://prokat24.com.ua/ua/contact-us/\n"
    )

    assert result == expected


def test_search_service_info_handler_raises_error_when_pages_are_empty() -> None:
    pages: list[KnowledgePage] = []
    arguments: ToolArguments = {
        "query": "Скільки коштує доставка?",
    }

    handler = create_search_service_info_handler(pages)

    with pytest.raises(ValueError, match="Knowledge pages not found"):
        handler(arguments)
