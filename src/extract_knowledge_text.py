from prokat24_ai_lab.html_parsing import extract_main_content_text, extract_title
from prokat24_ai_lab.knowledge_pages import get_knowledge_page_urls
from prokat24_ai_lab.prokat24_client import fetch_page


def main() -> None:
    urls = get_knowledge_page_urls()

    print("Extracting knowledge page text:")

    for index, url in enumerate(urls, start=1):
        html = fetch_page(url)
        title = extract_title(html)
        text = extract_main_content_text(html)
        preview = text[:500]

        print(f"{index}. {url}")
        print(f"Title: {title}")
        print(f"Text length: {len(text)} characters")
        print("Preview (500):")
        print(preview)
        print()


if __name__ == "__main__":
    main()
