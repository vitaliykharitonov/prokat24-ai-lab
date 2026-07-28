import pytest

from prokat24_ai_lab.retrieval.rrf import reciprocal_rank_fusion


def test_reciprocal_rank_fusion_combines_document_rankings() -> None:
    rankings = [
        ["dostavka", "contact-us", "o-nas"],
        ["contact-us", "o-nas", "dostavka"],
    ]

    result = reciprocal_rank_fusion(rankings)

    assert [document_id for document_id, _score in result] == [
        "contact-us",
        "dostavka",
        "o-nas",
    ]

    assert [score for _document_id, score in result] == pytest.approx(
        [
            0.032522,
            0.032266,
            0.032002,
        ],
        abs=0.000001,
    )
