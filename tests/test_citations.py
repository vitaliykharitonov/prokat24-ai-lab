import pytest

from prokat24_ai_lab.rag.citations import extract_document_citations, validate_document_citations


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


def test_validate_document_citations_accepts_valid_citations() -> None:
    answer = "Доставка доступна [Document 1][Document 3]."
    document_count = 3

    validate_document_citations(answer, document_count)


def test_validate_document_citations_raises_when_document_count_is_not_positive() -> None:
    answer = "Доставка доступна [Document 1][Document 3]."
    document_count = 0

    with pytest.raises(ValueError, match="Document count must be positive"):
        validate_document_citations(answer, document_count)


def test_validate_document_citations_raises_when_answer_has_no_citations() -> None:
    answer = "Доставка доступна."
    document_count = 3

    with pytest.raises(ValueError, match="No citations found"):
        validate_document_citations(answer, document_count)


def test_validate_document_citations_raises_when_citation_is_greater_than_document_count() -> None:
    answer = "Доставка доступна [Document 4]."
    document_count = 3

    with pytest.raises(ValueError, match="Document citation is out of range: 4; expected 1 to 3"):
        validate_document_citations(answer, document_count)


def test_validate_document_citations_raises_when_citation_is_zero() -> None:
    answer = "Доставка доступна [Document 0]."
    document_count = 3

    with pytest.raises(ValueError, match="Document citation is out of range: 0; expected 1 to 3"):
        validate_document_citations(answer, document_count)
