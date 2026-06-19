from prokat24_ai_lab.text_normalization import normalize_text


def test_normalize_text_returns_lowercase_words() -> None:
    result = normalize_text("Доставка: Київ — 200 грн!")

    assert result == "доставка київ 200 грн"
