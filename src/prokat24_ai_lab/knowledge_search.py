import re
from typing import TypedDict

from prokat24_ai_lab.knowledge_base import KnowledgePage

STOP_WORDS = {
    # --Stop words
    "чи",
    "є",
    "як",
    "яка",
    "який",
    "що",
    "в",
    "на",
    "при",
    "з",
    "за",
    # --Domain stop words
    "спорядження",
    "прокат",
    "прокату",
}

PAGE_SEARCH_KEYWORDS = {
    "https://prokat24.com.ua/ua/contact-us/": [
        "телефон",
        "телефони",
        "адреса",
        "адрес",
        "контакти",
        "viber",
        "email",
    ],
}

QUERY_ALIASES = {
    "привезти": ["доставка"],
    "знайти": ["адреса", "контакти"],
    "зв'язатись": ["контакти", "телефон"],
    "продовжити": ["прострочення", "строк", "термін"],
}


class KnowledgeSearchResult(TypedDict):
    url: str
    title: str
    snippet: str
    score: int


def normalize_text(text: str) -> str:
    words = re.findall(r"[0-9a-zа-яіїєґʼ']+", text.lower())

    return " ".join(words)


def expand_query_words(words: list[str]) -> list[str]:
    expanded_words = []

    for word in words:
        expanded_words.append(word)
        aliases = QUERY_ALIASES.get(word, [])
        expanded_words.extend(aliases)

    return expanded_words


def deduplicate_words(words: list[str]) -> list[str]:
    seen_words = set()
    unique_words = []

    for word in words:
        if word in seen_words:
            continue

        else:
            seen_words.add(word)
            unique_words.append(word)

    return unique_words


def get_query_words(query: str) -> list[str]:
    query_words = normalize_text(query).split()
    filtred_words = [word for word in query_words if word not in STOP_WORDS]
    expanded_query_words = expand_query_words(filtred_words)
    deduplicated_words = deduplicate_words(expanded_query_words)

    return deduplicated_words


def get_page_search_text(page: KnowledgePage) -> str:
    keywords = PAGE_SEARCH_KEYWORDS.get(page["url"], [])

    search_parts = [
        page["url"],
        page["title"],
        page["text"],
        *keywords,
    ]

    return " ".join(search_parts)


def calculate_keyword_score(query: str, text: str) -> int:
    query_words = get_query_words(query)
    text_words = normalize_text(text).split()

    score = 0

    for word in query_words:
        score += text_words.count(word)

    return score


def make_snippet(text: str, query: str, max_length: int = 300) -> str:
    query_words = normalize_text(query).split()
    lower_text = text.lower()

    first_position: int | None = None

    for word in query_words:
        position = lower_text.find(word)

        if position == -1:
            continue

        if first_position is None or position < first_position:
            first_position = position

    if first_position is None:
        snippet = text[:max_length].strip()

        if len(text) > max_length:
            return f"{snippet} ..."

        return snippet

    start = max(0, first_position - max_length // 2)
    end = min(len(text), start + max_length)

    snippet = text[start:end].strip()

    if start > 0:
        snippet = f"...{snippet}"

    if end < len(text):
        snippet = f"{snippet}..."

    return snippet


def search_knowledge_base(query: str, pages: list[KnowledgePage]) -> list[KnowledgeSearchResult]:
    results: list[KnowledgeSearchResult] = []

    for page in pages:
        searchable_text = get_page_search_text(page)
        score = calculate_keyword_score(query, searchable_text)

        if score == 0:
            continue

        result: KnowledgeSearchResult = {
            "url": page["url"],
            "title": page["title"],
            "snippet": make_snippet(page["text"], query),
            "score": score,
        }

        results.append(result)

    return sorted(results, key=lambda result: result["score"], reverse=True)
