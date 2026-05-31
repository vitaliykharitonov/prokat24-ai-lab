from prokat24_ai_lab.knowledge_pages import get_knowledge_page_urls


def main() -> None:
    knowledge_page_urls = get_knowledge_page_urls()

    print("Knowledge base candidate pages:")

    for index, knowledge_page_url in enumerate(knowledge_page_urls, start=1):
        print(f"{index}. {knowledge_page_url}")


if __name__ == "__main__":
    main()
