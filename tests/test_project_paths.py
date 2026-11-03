from pathlib import Path

import pytest

from prokat24_ai_lab.project_paths import find_project_root


def test_find_project_root_from_nested_directory(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    nested_directory = project_root / "src" / "prokat24_ai_lab"

    nested_directory.mkdir(parents=True)
    (project_root / "pyproject.toml").write_text("", encoding="utf-8")

    result = find_project_root(nested_directory)

    assert result == project_root


def test_find_project_root_from_nested_file(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    nested_directory = project_root / "src" / "prokat24_ai_lab"
    nested_file = nested_directory / "module.py"

    nested_directory.mkdir(parents=True)
    nested_file.write_text("", encoding="utf-8")
    (project_root / "pyproject.toml").write_text("", encoding="utf-8")

    result = find_project_root(nested_file)

    assert result == project_root


def test_find_project_root_raises_when_pyproject_is_missing(tmp_path: Path) -> None:
    nested_directory = tmp_path / "some" / "nested" / "directory"
    nested_directory.mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="Project root was not found"):
        find_project_root(nested_directory)
