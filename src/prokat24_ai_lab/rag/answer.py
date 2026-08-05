from collections.abc import Callable

from prokat24_ai_lab.knowledge.base import KnowledgePage
from prokat24_ai_lab.rag.context import build_retrieved_context_documents, format_retrieved_context
from prokat24_ai_lab.rag.generation import TextGenerator, generate_grounded_answer
from prokat24_ai_lab.rag.prompt import build_grounded_answer_prompt

DocumentRanker = Callable[[str], list[tuple[str, float]]]


def answer_question_from_ranked_documents(
    question: str,
    ranked_documents: list[tuple[str, float]],
    pages: list[KnowledgePage],
    generate_text: TextGenerator,
    top_k: int,
) -> str:
    context_documents = build_retrieved_context_documents(ranked_documents, pages, top_k)
    formatted_context = format_retrieved_context(context_documents)

    prompt = build_grounded_answer_prompt(question, formatted_context)

    answer = generate_grounded_answer(prompt, generate_text)

    return answer


def answer_question(
    question: str,
    pages: list[KnowledgePage],
    rank_documents: DocumentRanker,
    generate_text: TextGenerator,
    top_k: int,
) -> str:
    ranked_documents = rank_documents(question)

    answer = answer_question_from_ranked_documents(
        question=question,
        ranked_documents=ranked_documents,
        pages=pages,
        generate_text=generate_text,
        top_k=top_k,
    )

    return answer
