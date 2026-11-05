import os
from pathlib import Path

from prokat24_ai_lab.evals.agent.outcome import (
    build_agent_eval_comparison_markdown_summary,
    compare_agent_eval_reports,
    load_agent_eval_report,
)
from prokat24_ai_lab.project_paths import AGENT_EVAL_REPORTS_DIR


def main() -> None:
    previous_file = Path("previous_agent_eval") / "latest.json"
    current_file = AGENT_EVAL_REPORTS_DIR / "latest.json"

    previous_report = load_agent_eval_report(previous_file)
    current_report = load_agent_eval_report(current_file)

    comparison = compare_agent_eval_reports(previous_report, current_report)

    comparison_markdown_summary = build_agent_eval_comparison_markdown_summary(comparison)

    summary_file = Path(os.environ["GITHUB_STEP_SUMMARY"])

    with open(summary_file, "a", encoding="utf-8") as file:
        file.write("\n" + comparison_markdown_summary + "\n")


if __name__ == "__main__":
    main()
