import json
from pathlib import Path
from typing import TypedDict

from prokat24_ai_lab.html_parsing import extract_main_content_text, extract_title
from prokat24_ai_lab.knowledge_pages import get_knowledge_page_urls
from prokat24_ai_lab.prokat24_client import fetch_page


class KnowledgePage(TypedDict):
    url: str
    title: str
    text: str
    text_length: int


def build_knowledge_page(url: str) -> KnowledgePage:
    html = fetch_page(url)
    title = extract_title(html)
    text = extract_main_content_text(html)
    text_length = len(text)

    return {
        "url": url,
        "title": title,
        "text": text,
        "text_length": text_length,
    }


def build_knowledge_base() -> list[KnowledgePage]:
    urls = get_knowledge_page_urls()
    knowledge_pages: list[KnowledgePage] = []

    for url in urls:
        knowledge_page = build_knowledge_page(url)
        knowledge_pages.append(knowledge_page)

    return knowledge_pages


def save_knowledge_base(pages: list[KnowledgePage], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(pages, file, ensure_ascii=False, indent=2)