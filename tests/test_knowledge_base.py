import json
from pathlib import Path

from prokat24_ai_lab.knowledge.base import (
    KnowledgePage,
    build_knowledge_page_from_html,
    save_knowledge_base,
)


def test_build_knowledge_page_from_html_returns_page_data() -> None:
    url = "https://prokat24.com.ua/ua/o-nas/"
    html = """
    <html>
        <head>
            <title>Про нас — Прокат24</title>
        </head>
        <body>
            <main class="content">
                <div class="editor">
                    <p>Прокат24 — комплексний сервіс.</p>
                    <p>Ми надаємо повний спектр послуг.</p>
                </div>
            </main>
        </body>
    </html>
    """

    result = build_knowledge_page_from_html(url, html)

    assert result["url"] == url
    assert result["title"] == "Про нас — Прокат24"
    assert "Прокат24 — комплексний сервіс." in result["text"]
    assert "Ми надаємо повний спектр послуг." in result["text"]
    assert result["text_length"] == len(result["text"])


def test_save_knowledge_base_writes_json_file(tmp_path: Path) -> None:
    pages: list[KnowledgePage] = [
        {
            "url": "https://prokat24.com.ua/ua/o-nas/",
            "title": "Про нас — Прокат24",
            "text": "Прокат24 — комплексний сервіс.",
            "text_length": len("Прокат24 — комплексний сервіс."),
        }
    ]

    output_file = tmp_path / "knowledge_base.json"

    save_knowledge_base(pages, output_file)

    with open(output_file, encoding="utf-8") as file:
        loaded_pages = json.load(file)

    assert loaded_pages == pages
