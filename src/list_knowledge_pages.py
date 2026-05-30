from prokat24_ai_lab.site_config import BASE_URL
from prokat24_ai_lab.knowledge_pages import build_knowledge_page_urls, KNOWLEDGE_PAGE_PATHS


def main() -> None:
    knowledge_page_urls = build_knowledge_page_urls(base_url=BASE_URL, paths=KNOWLEDGE_PAGE_PATHS)

    print("Knowledge base candidate pages:")

    for index, knowledge_page_url in enumerate(knowledge_page_urls, start=1):
        print(f"{index}. {knowledge_page_url}")


if __name__ == "__main__":
    main()