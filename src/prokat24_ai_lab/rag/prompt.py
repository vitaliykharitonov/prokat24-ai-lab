from typing import TypedDict


class GroundedAnswerPrompt(TypedDict):
    instructions: str
    input: str


INSTRUCTIONS: str = (
    "Answer the user's question using only the provided context.\n"
    "Answer in Ukrainian.\n"
    "Do not invent prices, availability, addresses, rental terms, or other facts.\n"
    "If the context does not contain enough information, say that the provided context "
    "is insufficient.\n"
    "Only say that the context is insufficient when information required "
    "to answer the user's direct question is missing.\n"
    "Answer the user's exact intent directly and concisely.\n"
    "When the user's question contains an explicit constraint, "
    "only answer with options that satisfy that constraint.\n"
    "Do not mention alternatives that violate it, even as additional information.\n"
    "Do not introduce unrelated prerequisites or procedures "
    "unless they are necessary to answer the user's question.\n"
    "Do not offer follow-up help or ask additional questions.\n"
    "Cite supporting context documents using their labels, such as [Document 1].\n"
    "Every factual claim must include a citation to the document that supports it."
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
