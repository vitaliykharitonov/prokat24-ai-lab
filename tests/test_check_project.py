from check_project import get_quality_check_commands


def test_get_quality_check_commands_returns_project_checks() -> None:
    results = get_quality_check_commands()

    assert len(results) == 4
    assert results[0][1:] == ["-m", "pytest"]
    assert results[1][1:] == ["-m", "ruff", "check", "."]
    assert results[2][1:] == ["-m", "ruff", "format", "--check", "."]
    assert results[3][1:] == ["src/run_product_answer_evals.py"]
