from prokat24_ai_lab.text_normalization import normalize_text


def text_to_binary_vector(
    text: str,
    vocabulary: list[str],
    word_forms: dict[str, list[str]] | None = None,
) -> list[float]:
    normalized_words = set(normalize_text(text).split())
    text_vector = []

    for word in vocabulary:
        candidates = [word]

        if word_forms is not None:
            candidates.extend(word_forms.get(word, []))

        if set(candidates) & normalized_words:
            text_vector.append(1.0)

        else:
            text_vector.append(0.0)

    return text_vector
