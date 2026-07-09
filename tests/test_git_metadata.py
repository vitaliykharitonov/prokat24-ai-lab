import subprocess
from pathlib import Path

import pytest

import prokat24_ai_lab.git_metadata as git_metadata
from prokat24_ai_lab.project_paths import PROJECT_ROOT


def test_get_current_git_commit_returns_short_commit_hash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def fake_subprocess_run(
        command: list[str],
        capture_output: bool,
        text: bool,
        check: bool,
        cwd: Path,
    ) -> subprocess.CompletedProcess[str]:
        calls.append(
            {
                "command": command,
                "capture_output": capture_output,
                "text": text,
                "check": check,
                "cwd": cwd,
            }
        )
        return subprocess.CompletedProcess(
            args=command,
            returncode=0,
            stdout="abc1234\n",
            stderr="",
        )

    monkeypatch.setattr(
        git_metadata.subprocess,
        "run",
        fake_subprocess_run,
    )

    result = git_metadata.get_current_git_commit()

    assert result == "abc1234"
    assert calls == [
        {
            "command": ["git", "rev-parse", "--short", "HEAD"],
            "capture_output": True,
            "text": True,
            "check": True,
            "cwd": PROJECT_ROOT,
        }
    ]


def test_get_current_git_commit_returns_none_when_git_command_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_subprocess_run(
        command: list[str],
        capture_output: bool,
        text: bool,
        check: bool,
        cwd: Path,
    ) -> subprocess.CompletedProcess[str]:
        raise subprocess.CalledProcessError(
            returncode=128,
            cmd=command,
            stderr="fatal: not a git repository",
        )

    monkeypatch.setattr(
        git_metadata.subprocess,
        "run",
        fake_subprocess_run,
    )

    result = git_metadata.get_current_git_commit()

    assert result is None


def test_get_current_git_commit_returns_none_when_git_is_not_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_subprocess_run(
        command: list[str],
        capture_output: bool,
        text: bool,
        check: bool,
        cwd: Path,
    ) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError

    monkeypatch.setattr(
        git_metadata.subprocess,
        "run",
        fake_subprocess_run,
    )

    result = git_metadata.get_current_git_commit()

    assert result is None


def test_get_current_git_commit_returns_none_when_stdout_is_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_subprocess_run(
        command: list[str],
        capture_output: bool,
        text: bool,
        check: bool,
        cwd: Path,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=command,
            returncode=0,
            stdout="\n",
            stderr="",
        )

    monkeypatch.setattr(git_metadata.subprocess, "run", fake_subprocess_run)

    result = git_metadata.get_current_git_commit()

    assert result is None


def test_is_git_worktree_dirty_returns_false_when_status_is_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_subprocess_run(
        command: list[str],
        capture_output: bool,
        text: bool,
        check: bool,
        cwd: Path,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=command,
            returncode=0,
            stdout="",
            stderr="",
        )

    monkeypatch.setattr(
        git_metadata.subprocess,
        "run",
        fake_subprocess_run,
    )

    result = git_metadata.is_git_worktree_dirty()

    assert result is False


def test_is_git_worktree_dirty_returns_true_when_status_has_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_subprocess_run(
        command: list[str],
        capture_output: bool,
        text: bool,
        check: bool,
        cwd: Path,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=command,
            returncode=0,
            stdout=" M src/file.py\n?? tests/test_file.py\n",
            stderr="",
        )

    monkeypatch.setattr(git_metadata.subprocess, "run", fake_subprocess_run)

    result = git_metadata.is_git_worktree_dirty()

    assert result is True


def test_is_git_worktree_dirty_returns_none_when_git_command_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_subprocess_run(
        command: list[str],
        capture_output: bool,
        text: bool,
        check: bool,
        cwd: Path,
    ) -> subprocess.CompletedProcess[str]:
        raise subprocess.CalledProcessError(
            returncode=128,
            cmd=command,
            stderr="fatal: not a git repository",
        )

    monkeypatch.setattr(git_metadata.subprocess, "run", fake_subprocess_run)

    result = git_metadata.is_git_worktree_dirty()

    assert result is None


def test_is_git_worktree_dirty_returns_none_when_git_is_not_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_subprocess_run(
        command: list[str],
        capture_output: bool,
        text: bool,
        check: bool,
        cwd: Path,
    ) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError

    monkeypatch.setattr(git_metadata.subprocess, "run", fake_subprocess_run)

    result = git_metadata.is_git_worktree_dirty()

    assert result is None
