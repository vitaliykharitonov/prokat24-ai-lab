from prokat24_ai_lab.text_vectorizer import text_to_binary_vector


def test_text_to_binary_vector_marks_present_words() -> None:
    text = "Яка застава та чи можлива доставка?"
    vocabulary = ["доставка", "адреса", "застава"]

    result = text_to_binary_vector(text, vocabulary)

    assert result == [1.0, 0.0, 1.0]


def test_text_to_binary_vector_does_not_match_part_of_word() -> None:
    text = "Потрібна застава"
    vocabulary = ["став", "застава"]

    result = text_to_binary_vector(text, vocabulary)

    assert result == [0.0, 1.0]


def test_text_to_binary_vector_returns_zero_vector_for_unknown_words() -> None:
    text = "Хочу взяти лижі"
    vocabulary = ["доставка", "адреса", "застава"]

    result = text_to_binary_vector(text, vocabulary)

    assert result == [0.0, 0.0, 0.0]


def test_text_to_binary_vector_matches_configured_word_forms() -> None:
    text = "Інформація про доставку та заставу"
    vocabulary = ["доставка", "застава"]

    word_forms = {
        "доставка": ["доставку"],
        "застава": ["заставу"],
    }

    result = text_to_binary_vector(text, vocabulary, word_forms)

    assert result == [1.0, 1.0]
