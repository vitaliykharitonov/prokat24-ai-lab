import re


def extract_document_citations(answer: str) -> list[int]:
    document_citations = []
    citations = re.findall(r"\[Document (\d+)\]", answer)

    for citation in citations:
        document_citations.append(int(citation))

    return document_citations
