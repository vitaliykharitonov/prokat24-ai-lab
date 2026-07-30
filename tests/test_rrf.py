import pytest

from prokat24_ai_lab.retrieval.rrf import reciprocal_rank_fusion, weighted_reciprocal_rank_fusion


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


def test_weighted_reciprocal_rank_fusion_applies_ranking_weights() -> None:
    rankings = [
        ["dostavka", "contact-us", "o-nas"],
        ["contact-us", "o-nas", "dostavka"],
    ]
    weights = [1, 2]

    result = weighted_reciprocal_rank_fusion(rankings, weights)

    assert [document_id for document_id, _score in result] == [
        "contact-us",
        "dostavka",
        "o-nas",
    ]

    assert [score for _document_id, score in result] == pytest.approx(
        [
            0.048916,
            0.048139,
            0.048131,
        ],
        abs=0.000001,
    )


def test_weighted_reciprocal_rank_fusion_rejects_mismatched_lengths() -> None:
    rankings = [
        ["dostavka", "contact-us", "o-nas"],
        ["contact-us", "o-nas", "dostavka"],
    ]
    weights = [1, 2, 3]

    with pytest.raises(ValueError, match="Weights and rankings must have same length"):
        weighted_reciprocal_rank_fusion(rankings, weights)


def test_weighted_reciprocal_rank_fusion_rejects_negative_weight() -> None:
    rankings = [
        ["dostavka", "contact-us", "o-nas"],
        ["contact-us", "o-nas", "dostavka"],
    ]
    weights = [1, -2]

    with pytest.raises(ValueError, match="Weight cannot be negative"):
        weighted_reciprocal_rank_fusion(rankings, weights)


def test_weighted_reciprocal_rank_fusion_ignores_zero_weight_ranking() -> None:
    rankings = [
        ["keyword-a", "keyword-b"],
        ["embedding-a", "embedding-b"],
    ]

    weights = [1.0, 0.0]

    result = weighted_reciprocal_rank_fusion(rankings, weights)

    assert [document_id for document_id, _score in result] == [
        "keyword-a",
        "keyword-b",
    ]
