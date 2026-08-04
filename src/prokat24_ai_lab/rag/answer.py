from prokat24_ai_lab.knowledge.base import KnowledgePage
from prokat24_ai_lab.rag.context import build_retrieved_context_documents, format_retrieved_context
from prokat24_ai_lab.rag.generation import TextGenerator, generate_grounded_answer
from prokat24_ai_lab.rag.prompt import build_grounded_answer_prompt


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
