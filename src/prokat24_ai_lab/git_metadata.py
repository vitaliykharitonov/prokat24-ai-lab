import subprocess

from prokat24_ai_lab.project_paths import PROJECT_ROOT


def run_git_command(arguments: list[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git", *arguments],
            capture_output=True,
            text=True,
            check=True,
            cwd=PROJECT_ROOT,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None

    return result.stdout.strip()


def get_current_git_commit() -> str | None:
    commit = run_git_command(["rev-parse", "--short", "HEAD"])

    if not commit:
        return None

    return commit


def is_git_worktree_dirty() -> bool | None:
    status = run_git_command(["status", "--porcelain"])

    if status is None:
        return None

    return bool(status)
