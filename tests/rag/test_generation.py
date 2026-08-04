import pytest

from prokat24_ai_lab.rag.generation import generate_grounded_answer
from prokat24_ai_lab.rag.prompt import GroundedAnswerPrompt

PROMPT: GroundedAnswerPrompt = {
    "instructions": "Use only the provided context.",
    "input": (
        "Context:\n"
        "[Document 1]\n"
        "ID: delivery\n"
        "Title: Доставка\n"
        "Content:\n"
        "Спорядження можна отримати за допомогою доставки.\n\n"
        "Question:\n"
        "Як отримати спорядження?"
    ),
}


def test_generate_grounded_answer_calls_generator_with_prompt() -> None:
    received_arguments: list[tuple[str, str]] = []

    def generate_text(instructions: str, prompt_input: str) -> str:
        received_arguments.append((instructions, prompt_input))
        return "Спорядження можна отримати за допомогою доставки."

    result = generate_grounded_answer(PROMPT, generate_text)

    assert result == "Спорядження можна отримати за допомогою доставки."
    assert received_arguments == [
        (
            "Use only the provided context.",
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


def test_generate_grounded_answer_raises_for_empty_generated_answer() -> None:
    def generate_text(_instructions: str, _prompt_input: str) -> str:
        return "   "

    with pytest.raises(ValueError, match="Generated answer must not be empty"):
        generate_grounded_answer(PROMPT, generate_text)
