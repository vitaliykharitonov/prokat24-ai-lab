import math

import pytest

from prokat24_ai_lab.vector_math import cosine_similarity, dot_product, vector_magnitude


def test_dot_product_calculates_correctly() -> None:
    result = dot_product([1.0, 2.0], [3.0, 4.0])

    assert result == 11.0


def test_dot_product_raises_value_error_for_vectors_of_different_lengths() -> None:
    with pytest.raises(ValueError):
        dot_product([1.0], [3.0, 4.0])


def test_vector_magnitude_calculates_correctly() -> None:
    result = vector_magnitude([3.0, 4.0])

    assert result == 5.0


def test_cosine_similarity_returns_one_for_identical_vectors() -> None:
    result = cosine_similarity([1.0, 0.0], [1.0, 0.0])

    assert result == 1.0


def test_cosine_similarity_returns_zero_for_perpendicular_vectors() -> None:
    result = cosine_similarity([1.0, 0.0], [0.0, 1.0])

    assert result == 0.0


def test_cosine_similarity_returns_minus_one_for_opposite_vectors() -> None:
    result = cosine_similarity([1.0, 0.0], [-1.0, 0.0])

    assert result == -1.0


def test_cosine_similarity_raises_value_error_for_zero_vector() -> None:
    with pytest.raises(ValueError):
        cosine_similarity([0.0, 0.0], [1.0, 0.0])


def test_cosine_similarity_calculates_fractional_result() -> None:
    result = cosine_similarity([1.0, 1.0], [1.0, 0.0])
    expected_value = 1 / math.sqrt(2)

    assert result == pytest.approx(expected_value)
