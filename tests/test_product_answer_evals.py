from prokat24_ai_lab.product_answer_evals import find_missing_fragments


def test_find_missing_fragments_returns_missing_fragments() -> None:
    answer = """
        Прокат 2-місних наметів
        Тиждень: 1200 грн
    """

    expected_fragments = [
        "Прокат 2-місних наметів",
        "Тиждень: 1200 грн",
        "Deposit: 4000 грн",
    ]

    result = find_missing_fragments(answer, expected_fragments)

    assert result == [
        "Deposit: 4000 грн",
    ]


def test_find_missing_fragments_returns_empty_list_when_all_fragments_present() -> None:
    answer = """
            Прокат 2-місних наметів
            Тиждень: 1200 грн
        """

    expected_fragments = [
        "Прокат 2-місних наметів",
        "Тиждень: 1200 грн",
    ]

    result = find_missing_fragments(answer, expected_fragments)

    assert result == []
