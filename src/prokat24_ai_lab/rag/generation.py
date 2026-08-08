from collections.abc import Callable

from prokat24_ai_lab.rag.prompt import GroundedAnswerPrompt

TextGenerator = Callable[[str, str], str]


def generate_grounded_answer(
    prompt: GroundedAnswerPrompt,
    generate_text: TextGenerator,
) -> str:
    answer = generate_text(prompt["instructions"], prompt["input"])

    if not answer.strip():
        raise ValueError("Generated answer must not be empty")

    return answer
