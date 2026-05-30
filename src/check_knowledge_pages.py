from prokat24_ai_lab.html_parsing import extract_title
from prokat24_ai_lab.knowledge_pages import (
    build_knowledge_page_urls,
    KNOWLEDGE_PAGE_PATHS,
)
from prokat24_ai_lab.prokat24_client import fetch_page
from prokat24_ai_lab.site_config import BASE_URL


def main() -> None:
    knowledge_page_urls = build_knowledge_page_urls(
        base_url=BASE_URL,
        paths=KNOWLEDGE_PAGE_PATHS,
    )

    print("Checking knowledge base candidate pages:")

    for index, knowledge_page_url in enumerate(knowledge_page_urls, start=1):
        html = fetch_page(knowledge_page_url)
        title = extract_title(html)

        print(f"{index}. {knowledge_page_url}")
        print(f"Title: {title}")
        print(f"HTML Length: {len(html)}")
        print()


if __name__ == "__main__":
    main()
