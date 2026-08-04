import pytest

from prokat24_ai_lab.rag.prompt import (
    build_grounded_answer_prompt,
)


def test_build_grounded_answer_prompt_returns_instructions_and_input() -> None:
    question = "Як отримати спорядження?"
    context = (
        "[Document 1]\n"
        "ID: delivery\n"
        "Title: Доставка\n"
        "Content:\n"
        "Спорядження можна отримати за допомогою доставки."
    )

    result = build_grounded_answer_prompt(question, context)

    assert result["instructions"] == (
        "Answer the user's question using only the provided context.\n"
        "Answer in Ukrainian.\n"
        "Do not invent prices, availability, addresses, rental terms, or other facts.\n"
        "If the context does not contain enough information, say that the provided context "
        "is insufficient."
    )
    assert result["input"] == (
        "Context:\n"
        "[Document 1]\n"
        "ID: delivery\n"
        "Title: Доставка\n"
        "Content:\n"
        "Спорядження можна отримати за допомогою доставки.\n\n"
        "Question:\n"
        "Як отримати спорядження?"
    )


def test_build_grounded_answer_prompt_raises_for_empty_question() -> None:
    question = " "
    context = (
        "[Document 1]\n"
        "ID: delivery\n"
        "Title: Доставка\n"
        "Content:\n"
        "Спорядження можна отримати за допомогою доставки."
    )

    with pytest.raises(ValueError, match="Question must not be empty"):
        build_grounded_answer_prompt(question, context)


def test_build_grounded_answer_prompt_raises_for_empty_context() -> None:
    question = "Як отримати спорядження?"
    context = " "

    with pytest.raises(ValueError, match="Context must not be empty"):
        build_grounded_answer_prompt(question, context)
