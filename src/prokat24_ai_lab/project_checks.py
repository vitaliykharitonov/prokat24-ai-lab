import subprocess
import sys


def get_quality_checks() -> list[tuple[str, list[str]]]:
    return [
        (
            "PYTEST",
            [sys.executable, "-m", "pytest", "-q"],
        ),
        (
            "RUFF CHECK",
            [sys.executable, "-m", "ruff", "check", "."],
        ),
        (
            "RUFF FORMAT",
            [sys.executable, "-m", "ruff", "format", "--check", "."],
        ),
        (
            "MYPY",
            [sys.executable, "-m", "mypy", "src/prokat24_ai_lab"],
        ),
        (
            "KEYWORD SEARCH EVALS",
            [sys.executable, "scripts/evals/run_search_evals.py"],
        ),
        (
            "KNOWLEDGE ANSWER EVALS",
            [sys.executable, "scripts/evals/run_answer_evals.py"],
        ),
        (
            "PRODUCT ANSWER EVALS",
            [sys.executable, "scripts/evals/run_product_answer_evals.py"],
        ),
        (
            "EMBEDDING SEARCH EVALS",
            [
                sys.executable,
                "scripts/evals/run_offline_embedding_search_evals.py",
                "--split",
                "dev",
            ],
        ),
        (
            "HYBRID SEARCH EVALS",
            [sys.executable, "scripts/evals/run_hybrid_search_evals.py", "--split", "dev"],
        ),
    ]


def run_quality_check_command(check_name: str, command: list[str]) -> int:
    print()
    print("-" * 80)
    print(check_name)
    print("-" * 80)
    print(f"Running: {' '.join(command)}", flush=True)
    completed_process = subprocess.run(command, check=False)
    return completed_process.returncode


def run_quality_checks(checks: list[tuple[str, list[str]]]) -> int:
    for check_name, command in checks:
        return_code = run_quality_check_command(check_name, command)

        if return_code != 0:
            print()
            print(f"FAILED: {check_name} (exit code {return_code})")
            return return_code

    print()
    print("=" * 80)
    print("ALL PROJECT CHECKS PASSED")
    print("=" * 80)

    return 0


def main() -> int:
    checks = get_quality_checks()
    return run_quality_checks(checks)
