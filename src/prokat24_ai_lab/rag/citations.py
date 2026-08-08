import re


def extract_document_citations(answer: str) -> list[int]:
    document_citations = []
    citations = re.findall(r"\[Document (\d+)\]", answer)

    for citation in citations:
        document_citations.append(int(citation))

    return document_citations


def validate_document_citations(
    answer: str,
    document_count: int,
) -> None:
    if document_count <= 0:
        raise ValueError("Document count must be positive")

    document_citations = extract_document_citations(answer)

    if not document_citations:
        raise ValueError("No citations found")

    for citation in document_citations:
        if citation < 1 or citation > document_count:
            raise ValueError(
                f"Document citation is out of range: {citation}; expected 1 to {document_count}"
            )
