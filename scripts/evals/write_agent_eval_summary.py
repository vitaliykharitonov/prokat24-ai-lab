import os
from pathlib import Path

from prokat24_ai_lab.evals.agent_evals import (
    build_agent_eval_markdown_summary,
    load_agent_eval_report,
)
from prokat24_ai_lab.project_paths import AGENT_EVAL_REPORTS_DIR


def main() -> None:
    input_file = AGENT_EVAL_REPORTS_DIR / "latest.json"

    report = load_agent_eval_report(input_file)
    markdown_summary = build_agent_eval_markdown_summary(report)

    summary_file = Path(os.environ["GITHUB_STEP_SUMMARY"])

    with open(summary_file, "a", encoding="utf-8") as file:
        file.write(markdown_summary + "\n")


if __name__ == "__main__":
    main()
