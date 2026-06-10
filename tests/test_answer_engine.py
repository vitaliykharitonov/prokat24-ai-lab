from prokat24_ai_lab.answer_engine import answer_question


TEST_PAGES = [
    {
        "url": "https://prokat24.com.ua/ua/dostavka/",
        "title": "Доставка прокатного спорядження — Прокат24",
        "text": "Доставка можлива. По Києву — від 200 грн. Доставка великогабаритних товарів.",
        "text_length": len(
            "Доставка можлива. По Києву — від 200 грн. Доставка великогабаритних товарів."
        ),
    },
    {
        "url": "https://prokat24.com.ua/ua/o-nas/",
        "title": "Про нас — Прокат24",
        "text": "Прокат спорядження та сервіс.",
        "text_length": len("Прокат спорядження та сервіс."),
    },
]


def test_answer_question_returns_snippet_and_source_when_result_found() -> None:
    question = "доставка київ"
    answer = answer_question(question, TEST_PAGES)

    assert answer["question"] == question
    assert "Доставка" in answer["answer"]
    assert len(answer["sources"]) == 1
    assert answer["sources"][0]["title"] == TEST_PAGES[0]["title"]
    assert answer["sources"][0]["url"] == TEST_PAGES[0]["url"]
    assert answer["confidence"] == "medium"


def test_answer_question_returns_low_confidence_when_no_result_found() -> None:
    question = "абракадабра невідомеслово"
    answer = answer_question(question, TEST_PAGES)

    assert answer["question"] == question
    assert answer["answer"] == "Не знайшов відповідь у базі знань."
    assert answer["sources"] == []
    assert answer["confidence"] == "low"