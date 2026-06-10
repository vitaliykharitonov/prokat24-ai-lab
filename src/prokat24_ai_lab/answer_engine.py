from typing import TypedDict

from prokat24_ai_lab.knowledge_base import KnowledgePage
from prokat24_ai_lab.knowledge_search import search_knowledge_base


class AnswerSource(TypedDict):
    title: str
    url: str


class KnowledgeAnswer(TypedDict):
    question: str
    answer: str
    sources: list[AnswerSource]
    confidence: str


def answer_question(question: str, pages: list[KnowledgePage]) -> KnowledgeAnswer:
    search_results = search_knowledge_base(question, pages)

    if not search_results:
        answer = "Не знайшов відповідь у базі знань."
        return {
            "question": question,
            "answer": answer,
            "sources": [],
            "confidence": "low",
        }

    top_result = search_results[0]
    return {
        "question": question,
        "answer": top_result["snippet"],
        "sources": [{"title": top_result["title"], "url": top_result["url"]}],
        "confidence": "medium",
    }
