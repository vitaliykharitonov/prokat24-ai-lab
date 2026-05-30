from prokat24_ai_lab.knowledge_pages import build_knowledge_page_urls, KNOWLEDGE_PAGE_PATHS
from prokat24_ai_lab.site_config import BASE_URL


def test_build_knowledge_page_urls() -> None:
    expected_urls = [
        "https://prokat24.com.ua/ua/pravila-prokata/",
        "https://prokat24.com.ua/ua/dostavka/",
        "https://prokat24.com.ua/ua/contact-us/",
        "https://prokat24.com.ua/ua/o-nas/",
    ]

    result = build_knowledge_page_urls(base_url=BASE_URL, paths=KNOWLEDGE_PAGE_PATHS)

    assert result == expected_urls