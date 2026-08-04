from typing import TypedDict

from prokat24_ai_lab.knowledge.base import KnowledgePage


class RetrievedContextDocument(TypedDict):
    document_id: str
    title: str
    text: str
    score: float


def build_retrieved_context_documents(
    ranked_documents: list[tuple[str, float]],
    pages: list[KnowledgePage],
    top_k: int,
) -> list[RetrievedContextDocument]:
    if top_k <= 0:
        raise ValueError("top_k must be > 0")

    context_documents: list[RetrievedContextDocument] = []
    pages_by_url = {page["url"]: page for page in pages}

    for document_id, score in ranked_documents[:top_k]:
        if document_id not in pages_by_url:
            raise ValueError(f"Knowledge page not found for document_id: {document_id}")

        page = pages_by_url[document_id]
        context_documents.append(
            RetrievedContextDocument(
                document_id=document_id,
                title=page["title"],
                text=page["text"],
                score=score,
            )
        )

    return context_documents
