from prokat24_ai_lab.rag.citations import extract_document_citations


def test_extract_document_citations_returns_numbers_in_appearance_order() -> None:
    answer = "Доставка доступна [Document 2]. Умови уточнюються окремо [Document 1][Document 2]."

    result = extract_document_citations(answer)

    assert result == [2, 1, 2]


def test_extract_document_citations_returns_empty_list_when_no_citations() -> None:
    answer = "Доставка доступна за попередньою домовленістю."
    result = extract_document_citations(answer)

    assert result == []


def test_extract_document_citations_ignores_malformed_citations() -> None:
    answer = (
        "Доставка доступна [document 1] Document 3. "
        "Умови уточнюються окремо [Document X][Document 2."
    )

    result = extract_document_citations(answer)

    assert result == []
