from urllib.parse import urljoin

from prokat24_ai_lab.site_config import BASE_URL

KNOWLEDGE_PAGE_PATHS = [
    "/ua/pravila-prokata/",
    "/ua/dostavka/",
    "/ua/contact-us/",
    "/ua/o-nas/",
]


def build_knowledge_page_urls(base_url: str, paths: list[str]) -> list[str]:
    knowledge_page_urls: list[str] = []

    for path in paths:
        absolute_url = urljoin(base_url, path)
        knowledge_page_urls.append(absolute_url)

    return knowledge_page_urls


def get_knowledge_page_urls() -> list[str]:
    return build_knowledge_page_urls(
        base_url=BASE_URL,
        paths=KNOWLEDGE_PAGE_PATHS,
    )
