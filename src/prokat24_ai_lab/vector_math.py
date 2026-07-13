import math


def dot_product(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError("Vectors must have the same length")

    return sum(a[i] * b[i] for i in range(len(a)))


def vector_magnitude(vector: list[float]) -> float:
    return math.sqrt(sum(value**2 for value in vector))


def cosine_similarity(a: list[float], b: list[float]) -> float:
    a_magnitude = vector_magnitude(a)
    b_magnitude = vector_magnitude(b)

    if a_magnitude == 0 or b_magnitude == 0:
        raise ValueError("Vectors must have non-zero magnitudes")

    return dot_product(a, b) / (a_magnitude * b_magnitude)
