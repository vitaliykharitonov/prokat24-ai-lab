import subprocess
import sys


def get_quality_check_commands() -> list[list[str]]:
    return [
        [sys.executable, "-m", "pytest"],
        [sys.executable, "-m", "ruff", "check", "."],
        [sys.executable, "-m", "ruff", "format", "--check", "."],
        [sys.executable, "src/run_search_evals.py"],
        [sys.executable, "src/run_answer_evals.py"],
        [sys.executable, "src/run_product_answer_evals.py"],
        [sys.executable, "src/run_offline_embedding_search_evals.py"],
    ]


def run_quality_check_command(command: list[str]) -> int:
    print(f"Running: {' '.join(command)}")
    completed_process = subprocess.run(command, check=False)
    return completed_process.returncode


def run_quality_checks(commands: list[list[str]]) -> int:
    for command in commands:
        return_code = run_quality_check_command(command)

        if return_code != 0:
            return return_code

    return 0


def main() -> int:
    commands = get_quality_check_commands()
    return run_quality_checks(commands)


if __name__ == "__main__":
    sys.exit(main())
