from typing import TypedDict


class GroundedAnswerPrompt(TypedDict):
    instructions: str
    input: str


INSTRUCTIONS: str = (
    "Answer the user's question using only the provided context.\n"
    "Answer in Ukrainian.\n"
    "Do not invent prices, availability, addresses, rental terms, or other facts.\n"
    "If the context does not contain enough information, say that the provided context "
    "is insufficient."
)


def build_grounded_answer_prompt(
    question: str,
    context: str,
) -> GroundedAnswerPrompt:
    if not question.strip():
        raise ValueError("Question must not be empty")

    if not context.strip():
        raise ValueError("Context must not be empty")

    prompt_input = "\n".join(
        (
            "Context:",
            context,
            "",
            "Question:",
            question,
        )
    )

    return {
        "instructions": INSTRUCTIONS,
        "input": prompt_input,
    }
