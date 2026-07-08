import subprocess

from prokat24_ai_lab.project_paths import PROJECT_ROOT


def get_current_git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=PROJECT_ROOT,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None

    commit = result.stdout.strip()

    if not commit:
        return None

    return commit
