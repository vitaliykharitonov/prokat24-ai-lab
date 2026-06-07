from prokat24_ai_lab.knowledge_search import (
    calculate_keyword_score,
    expand_query_words,
    get_page_search_text,
    get_query_words,
    make_snippet,
    normalize_text,
    search_knowledge_base,
)


def test_normalize_text_returns_lowercase_words() -> None:
    result = normalize_text("Доставка: Київ — 200 грн!")

    assert result == "доставка київ 200 грн"


def test_expand_query_words_adds_aliases_and_keeps_words_without_aliases() -> None:
    query_words = ["скільки", "коштує", "привезти"]
    expanded_query_words = expand_query_words(query_words)

    assert "доставка" in expanded_query_words

    query_words = ["де", "вас", "знайти"]
    expanded_query_words = expand_query_words(query_words)

    assert "адреса" in expanded_query_words
    assert "контакти" in expanded_query_words

    query_words = ["ремонт", "палатки"]
    expanded_query_words = expand_query_words(query_words)

    assert expanded_query_words == ["ремонт", "палатки"]


def test_get_query_words_removes_stop_words() -> None:
    result = get_query_words(
        query="чи є сервіс спорядження",
    )

    assert result == ["сервіс"]


def test_get_query_words_adds_aliases() -> None:
    query = "скільки коштує привезти"

    result = get_query_words(query)

    assert "доставка" in result


def test_get_page_search_text_adds_keywords() -> None:
    page = {
        "url": "https://prokat24.com.ua/ua/contact-us/",
        "title": "Зв’язатися з нами",
        "text": "м. БЕРЕСТЕЙСЬКА\nвул. Табірна, 42\n",
        "text_length": len("м. БЕРЕСТЕЙСЬКА\nвул. Табірна, 42\n"),
    }

    result = get_page_search_text(page)

    assert "телефон" in result
    assert "адреса" in result
    assert "viber" in result


def test_get_page_search_text_works_without_keywords() -> None:
    page = {
        "url": "https://prokat24.com.ua/ua/dostavka/",
        "title": "Доставка прокатного спорядження — Прокат24",
        "text": "Доставка:\nПослуга доставки можлива",
        "text_length": len("Доставка:\nПослуга доставки можлива"),
    }

    result = get_page_search_text(page)

    assert page["url"] in result
    assert page["title"] in result
    assert page["text"] in result


def test_calculate_keyword_score_ignores_stop_words() -> None:
    result = calculate_keyword_score(
        query="чи є сервіс",
        text="чи чи є є сервіс",
    )

    assert result == 1


def test_calculate_keyword_score_counts_whole_words() -> None:
    result = calculate_keyword_score(
        query="доставка київ",
        text="Доставка можлива. Доставка по Києву.",
    )

    assert result == 2


def test_make_snippet_returns_text_near_query_word() -> None:
    result = make_snippet(
        text="Початок тексту. Доставка можлива за попередньою домовленістю. Кінець тексту.",
        query="доставка",
        max_length=50,
    )

    assert "Доставка" in result


def test_make_snippet_adds_trailing_ellipsis_when_text_is_cut() -> None:
    text = (
        "Доставка можлива за попередньою домовленістю. "
        "По Києву — від 200 грн. "
        "За межами Києва — від 200 грн + 10 грн/км. "
        "Самовивіз доступний у пунктах Прокат24."
    )

    result = make_snippet(text, query="доставка", max_length=50)

    assert result.startswith("Доставка")
    assert result.endswith("...")


def test_search_knowledge_base_returns_best_match_first() -> None:
    pages = [
        {
            "url": "https://prokat24.com.ua/ua/o-nas/",
            "title": "Про нас — Прокат24",
            "text": "Прокат спорядження та сервіс.",
            "text_length": len("Прокат спорядження та сервіс."),
        },
        {
            "url": "https://prokat24.com.ua/ua/dostavka/",
            "title": "Доставка прокатного спорядження — Прокат24",
            "text": "Доставка можлива. По Києву — від 200 грн. Доставка великогабаритних товарів.",
            "text_length": len(
                "Доставка можлива. По Києву — від 200 грн. Доставка великогабаритних товарів."
            ),
        },
    ]

    result = search_knowledge_base(
        query="доставка київ",
        pages=pages,
    )

    assert len(result) == 1
    assert result[0]["title"] == "Доставка прокатного спорядження — Прокат24"
    assert result[0]["score"] == 3
    assert "Доставка" in result[0]["snippet"]
