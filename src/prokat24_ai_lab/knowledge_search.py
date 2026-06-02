import re
from typing import TypedDict

from prokat24_ai_lab.knowledge_base import KnowledgePage


class KnowledgeSearchResult(TypedDict):
    url: str
    title: str
    snippet: str
    score: int


def normalize_text(text: str) -> str:
    words = re.findall(r"[0-9a-zа-яіїєґʼ']+", text.lower())

    return " ".join(words)


def calculate_keyword_score(query: str, text: str) -> int:
    query_words = normalize_text(query).split()
    text_words = normalize_text(text).split()

    score = 0

    for word in query_words:
        score += text_words.count(word)

    return score


def make_snippet(text: str, query: str, max_length: int = 300) -> str:
    query_words = normalize_text(query)
    lower_text = text.lower()

    first_position: int | None = None

    for word in query_words:
        position = lower_text.find(word)

        if position == -1:
            continue

        if first_position is None or position < first_position:
            first_position = position

    if first_position is None:
        return text[:max_length]

    start = max(0, first_position - max_length // 2)
    end = start + max_length

    return text[start:end].strip()


def search_knowledge_base(query: str, pages: list[KnowledgePage]) -> list[KnowledgeSearchResult]:
    results: list[KnowledgeSearchResult] = []

    for page in pages:
        searchable_text = f"{page['title']} {page['url']} {page['text']}"
        score = calculate_keyword_score(query, searchable_text)

        if score == 0:
            continue

        result: KnowledgeSearchResult = {
            "url": page['url'],
            "title": page['title'],
            "snippet": make_snippet(page['text'], query),
            "score": score,

        }

        results.append(result)

    return sorted(results, key=lambda result: result["score"], reverse=True)


