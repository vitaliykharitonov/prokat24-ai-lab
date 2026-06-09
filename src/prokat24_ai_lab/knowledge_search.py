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


WORD_ALIASES = {
    "київ": ["києву", "києва", "києві"],
    "доставка": ["доставку", "доставки", "доставкою"],
    "застава": ["заставу", "застави", "заставою"],
}


WORD_WEIGHTS = {
    "можна": 0.5,
    "паспорт": 2.0,
    "застава": 2.0,
    "телефон": 2.0,
    "адреса": 2.0,
}


class ScoreDetail(TypedDict):
    count: int
    weight: float
    contribution: float


class KnowledgeSearchResult(TypedDict):
    url: str
    title: str
    snippet: str
    score: float
    score_details: dict[str, ScoreDetail]
    matched_words: dict[str, int]


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


def expand_word_forms(words: list[str]) -> list[str]:
    expanded_words = []

    for word in words:
        expanded_words.append(word)
        aliases = WORD_ALIASES.get(word, [])
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
    filtered_words = [word for word in query_words if word not in STOP_WORDS]
    expanded_query_words = expand_query_words(filtered_words)
    expanded_word_forms = expand_word_forms(expanded_query_words)
    deduplicated_words = deduplicate_words(expanded_word_forms)

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
    matches = count_query_word_matches(query, text)
    score = sum(matches.values())

    return score


def count_query_word_matches(query: str, text: str) -> dict[str, int]:
    query_words = get_query_words(query)
    text_words = normalize_text(text).split()
    matches = {}

    for word in query_words:
        count = text_words.count(word)
        if count > 0:
            matches[word] = count

    return matches


def calculate_weighted_score(matched_words: dict[str, int]) -> float:
    score = calculate_score_from_details(build_score_details(matched_words))

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


def build_score_details(matched_words: dict[str, int]) -> dict[str, ScoreDetail]:
    score_details = {}

    for word, count in matched_words.items():
        weight = WORD_WEIGHTS.get(word, 1.0)
        contribution = count * weight
        score_details[word] = {
            "count": count,
            "weight": weight,
            "contribution": contribution,
        }

    return score_details


def calculate_score_from_details(score_details: dict[str, ScoreDetail]) -> float:
    score = 0.0

    for detail in score_details.values():
        score += detail["contribution"]

    return score


def search_knowledge_base(query: str, pages: list[KnowledgePage]) -> list[KnowledgeSearchResult]:
    results: list[KnowledgeSearchResult] = []

    for page in pages:
        searchable_text = get_page_search_text(page)
        matched_words = count_query_word_matches(query, searchable_text)
        score_details = build_score_details(matched_words)
        score = calculate_score_from_details(score_details)

        if score == 0.0:
            continue

        result: KnowledgeSearchResult = {
            "url": page["url"],
            "title": page["title"],
            "snippet": make_snippet(page["text"], query),
            "score": score,
            "score_details": score_details,
            "matched_words": matched_words,
        }

        results.append(result)

    return sorted(results, key=lambda result: result["score"], reverse=True)
