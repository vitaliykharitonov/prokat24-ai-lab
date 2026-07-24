from prokat24_ai_lab.project_checks import get_quality_check_commands


def test_get_quality_check_commands_returns_project_checks() -> None:
    results = get_quality_check_commands()

    assert len(results) == 8
    assert results[0][1:] == ["-m", "pytest"]
    assert results[1][1:] == ["-m", "ruff", "check", "."]
    assert results[2][1:] == ["-m", "ruff", "format", "--check", "."]
    assert results[3][1:] == ["-m", "mypy", "src/prokat24_ai_lab"]
    assert results[4][1:] == ["scripts/evals/run_search_evals.py"]
    assert results[5][1:] == ["scripts/evals/run_answer_evals.py"]
    assert results[6][1:] == ["scripts/evals/run_product_answer_evals.py"]
    assert results[7][1:] == ["scripts/evals/run_offline_embedding_search_evals.py"]
