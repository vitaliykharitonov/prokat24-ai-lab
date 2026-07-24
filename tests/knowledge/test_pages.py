from prokat24_ai_lab.knowledge.pages import (
    KNOWLEDGE_PAGE_PATHS,
    build_knowledge_page_urls,
    get_knowledge_page_urls,
)
from prokat24_ai_lab.site_config import BASE_URL

EXPECTED_KNOWLEDGE_PAGE_URLS = [
    "https://prokat24.com.ua/ua/pravila-prokata/",
    "https://prokat24.com.ua/ua/dostavka/",
    "https://prokat24.com.ua/ua/contact-us/",
    "https://prokat24.com.ua/ua/o-nas/",
]


def test_build_knowledge_page_urls_returns_absolute_urls() -> None:
    result = build_knowledge_page_urls(
        base_url=BASE_URL,
        paths=KNOWLEDGE_PAGE_PATHS,
    )

    assert result == EXPECTED_KNOWLEDGE_PAGE_URLS


def test_get_knowledge_page_urls_returns_expected_urls() -> None:
    result = get_knowledge_page_urls()

    assert result == EXPECTED_KNOWLEDGE_PAGE_URLS
