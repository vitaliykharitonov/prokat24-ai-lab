from prokat24_ai_lab.knowledge.base import KnowledgePage
from prokat24_ai_lab.rag.answer import (
    answer_question,
    answer_question_from_context_documents,
    answer_question_from_ranked_documents,
)
from prokat24_ai_lab.rag.context import RetrievedContextDocument


def test_answer_question_from_ranked_documents_builds_context_and_generates_answer() -> None:
    question = "Як отримати спорядження?"
    pages: list[KnowledgePage] = [
        {
            "url": "delivery",
            "title": "Доставка",
            "text": "Спорядження можна отримати за допомогою доставки.",
            "text_length": 52,
        },
        {
            "url": "contacts",
            "title": "Контакти",
            "text": "Пункт прокату знаходиться біля метро Берестейська.",
            "text_length": 54,
        },
    ]
    ranked_documents = [
        ("delivery", 0.9),
        ("contacts", 0.8),
    ]

    received_arguments: list[tuple[str, str]] = []

    def fake_generate_text(instructions: str, prompt_input: str) -> str:
        received_arguments.append((instructions, prompt_input))
        return "Спорядження можна отримати за допомогою доставки."

    result = answer_question_from_ranked_documents(
        question=question,
        ranked_documents=ranked_documents,
        pages=pages,
        generate_text=fake_generate_text,
        top_k=1,
    )

    assert result == "Спорядження можна отримати за допомогою доставки."
    assert received_arguments == [
        (
            (
                "Answer the user's question using only the provided context.\n"
                "Answer in Ukrainian.\n"
                "Do not invent prices, availability, addresses, rental terms, or other facts.\n"
                "If the context does not contain enough information, say that the provided context "
                "is insufficient.\n"
                "Answer the user's exact intent directly and concisely.\n"
                "Do not include alternatives that conflict with the user's stated constraints.\n"
                "Do not offer follow-up help or ask additional questions.\n"
                "Cite supporting context documents using their labels, such as [Document 1].\n"
                "Every factual claim must include a citation to the document that supports it."
            ),
            (
                "Context:\n"
                "[Document 1]\n"
                "ID: delivery\n"
                "Title: Доставка\n"
                "Content:\n"
                "Спорядження можна отримати за допомогою доставки.\n\n"
                "Question:\n"
                "Як отримати спорядження?"
            ),
        )
    ]


def test_answer_question_ranks_documents_and_generates_answer() -> None:
    question = "Як отримати спорядження?"
    pages: list[KnowledgePage] = [
        {
            "url": "delivery",
            "title": "Доставка",
            "text": "Спорядження можна отримати за допомогою доставки.",
            "text_length": 52,
        },
        {
            "url": "contacts",
            "title": "Контакти",
            "text": "Пункт прокату знаходиться біля метро Берестейська.",
            "text_length": 54,
        },
    ]
    received_questions: list[str] = []

    def fake_rank_documents(query: str) -> list[tuple[str, float]]:
        received_questions.append(query)
        return [
            ("delivery", 0.9),
            ("contacts", 0.8),
        ]

    received_arguments: list[tuple[str, str]] = []

    def fake_generate_text(instructions: str, prompt_input: str) -> str:
        received_arguments.append((instructions, prompt_input))
        return "Спорядження можна отримати за допомогою доставки."

    result = answer_question(
        question=question,
        pages=pages,
        rank_documents=fake_rank_documents,
        generate_text=fake_generate_text,
        top_k=1,
    )

    assert received_questions == [question]
    assert result == "Спорядження можна отримати за допомогою доставки."
    assert len(received_arguments) == 1

    _instructions, prompt_input = received_arguments[0]

    assert "ID: delivery" in prompt_input
    assert "ID: contacts" not in prompt_input


def test_answer_question_from_context_documents_generates_answer() -> None:
    question = "Як отримати спорядження?"
    context_documents: list[RetrievedContextDocument] = [
        {
            "document_id": "delivery",
            "title": "Доставка",
            "text": "Спорядження можна отримати за допомогою доставки.",
            "score": 0.9,
        }
    ]

    received_arguments: list[tuple[str, str]] = []

    def fake_generate_text(instructions: str, prompt_input: str) -> str:
        received_arguments.append((instructions, prompt_input))
        return "Спорядження можна отримати за допомогою доставки."

    result = answer_question_from_context_documents(
        question=question,
        context_documents=context_documents,
        generate_text=fake_generate_text,
    )

    assert result == "Спорядження можна отримати за допомогою доставки."
    assert len(received_arguments) == 1

    _instructions, prompt_input = received_arguments[0]

    assert "ID: delivery" in prompt_input
    assert "Title: Доставка" in prompt_input
    assert "Спорядження можна отримати за допомогою доставки." in prompt_input
    assert "Як отримати спорядження?" in prompt_input
