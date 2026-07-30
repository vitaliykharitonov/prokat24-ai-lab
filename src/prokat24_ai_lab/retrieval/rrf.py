def reciprocal_rank_fusion(
    rankings: list[list[str]],
    rank_constant: int = 60,
) -> list[tuple[str, float]]:
    document_scores: dict[str, float] = {}

    for ranking in rankings:
        for index, document_id in enumerate(ranking, start=1):
            rrf_score = 1 / (rank_constant + index)
            document_scores[document_id] = document_scores.get(document_id, 0.0) + rrf_score

    return sorted(document_scores.items(), key=lambda item: (-item[1], item[0]))


def weighted_reciprocal_rank_fusion(
    rankings: list[list[str]],
    weights: list[float],
    rank_constant: int = 60,
) -> list[tuple[str, float]]:
    if len(weights) != len(rankings):
        raise ValueError("Weights and rankings must have same length")

    for weight in weights:
        if weight < 0:
            raise ValueError("Weight cannot be negative")

    document_scores: dict[str, float] = {}

    for ranking, weight in zip(rankings, weights, strict=True):
        for index, document_id in enumerate(ranking, start=1):
            rrf_score = weight / (rank_constant + index)
            document_scores[document_id] = document_scores.get(document_id, 0.0) + rrf_score

    return sorted(document_scores.items(), key=lambda item: (-item[1], item[0]))
