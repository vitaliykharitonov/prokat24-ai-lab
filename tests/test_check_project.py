from prokat24_ai_lab.project_checks import get_quality_checks


def test_get_quality_check_commands_returns_project_checks() -> None:
    results = get_quality_checks()

    check_names = [check_name for check_name, _command in results]
    commands = [command for _check_name, command in results]

    assert len(results) == 9
    assert check_names == [
        "PYTEST",
        "RUFF CHECK",
        "RUFF FORMAT",
        "MYPY",
        "KEYWORD SEARCH EVALS",
        "KNOWLEDGE ANSWER EVALS",
        "PRODUCT ANSWER EVALS",
        "EMBEDDING SEARCH EVALS",
        "HYBRID SEARCH EVALS",
    ]

    assert commands[0][1:] == ["-m", "pytest", "-q"]
    assert commands[1][1:] == ["-m", "ruff", "check", "."]
    assert commands[2][1:] == ["-m", "ruff", "format", "--check", "."]
    assert commands[3][1:] == ["-m", "mypy", "src/prokat24_ai_lab"]
    assert commands[4][1:] == ["scripts/evals/run_search_evals.py"]
    assert commands[5][1:] == ["scripts/evals/run_answer_evals.py"]
    assert commands[6][1:] == ["scripts/evals/run_product_answer_evals.py"]
    assert commands[7][1:] == ["scripts/evals/run_offline_embedding_search_evals.py"]
    assert commands[8][1:] == ["scripts/evals/run_hybrid_search_evals.py"]
