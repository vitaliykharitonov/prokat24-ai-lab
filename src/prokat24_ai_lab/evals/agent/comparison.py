from pathlib import Path
from typing import Literal, TypedDict

from prokat24_ai_lab.evals.agent.report import AgentEvalReport, load_agent_eval_report


class AgentMetricComparison(TypedDict):
    previous_pass_rate: float
    current_pass_rate: float
    pass_rate_delta: float

    improved_case_ids: list[str]
    worsened_case_ids: list[str]
    unchanged_case_ids: list[str]

    added_case_ids: list[str]
    removed_case_ids: list[str]


class AgentEvalComparison(TypedDict):
    previous_run_id: str
    current_run_id: str

    outcome: AgentMetricComparison
    capability: AgentMetricComparison
    grounding: AgentMetricComparison


AgentMetricName = Literal["outcome", "capability", "grounding"]


def compare_agent_metric_results(
    previous_result: AgentEvalReport,
    current_result: AgentEvalReport,
    metric_name: AgentMetricName,
) -> AgentMetricComparison:
    previous_passed_by_id: dict[str, bool] = {}
    current_passed_by_id: dict[str, bool] = {}

    for result in previous_result["results"]:
        case_id = result["id"]
        passed = result[metric_name]["passed"]

        previous_passed_by_id[case_id] = passed

    for result in current_result["results"]:
        case_id = result["id"]
        passed = result[metric_name]["passed"]

        current_passed_by_id[case_id] = passed

    previous_case_ids = set(previous_passed_by_id)
    current_case_ids = set(current_passed_by_id)

    common_case_ids = sorted(previous_case_ids & current_case_ids)
    added_case_ids = sorted(current_case_ids - previous_case_ids)
    removed_case_ids = sorted(previous_case_ids - current_case_ids)

    improved_case_ids: list[str] = []
    worsened_case_ids: list[str] = []
    unchanged_case_ids: list[str] = []

    for case in common_case_ids:
        previous_passed = previous_passed_by_id[case]
        current_passed = current_passed_by_id[case]

        if not previous_passed and current_passed:
            improved_case_ids.append(case)

        elif previous_passed and not current_passed:
            worsened_case_ids.append(case)

        else:
            unchanged_case_ids.append(case)

    previous_summary = previous_result["summary"][metric_name]
    current_summary = current_result["summary"][metric_name]

    previous_pass_rate = previous_summary["pass_rate"]
    current_pass_rate = current_summary["pass_rate"]
    pass_rate_delta = current_pass_rate - previous_pass_rate

    return AgentMetricComparison(
        previous_pass_rate=previous_pass_rate,
        current_pass_rate=current_pass_rate,
        pass_rate_delta=pass_rate_delta,
        improved_case_ids=improved_case_ids,
        worsened_case_ids=worsened_case_ids,
        unchanged_case_ids=unchanged_case_ids,
        added_case_ids=added_case_ids,
        removed_case_ids=removed_case_ids,
    )


def compare_agent_eval_reports(
    previous_report: AgentEvalReport,
    current_report: AgentEvalReport,
) -> AgentEvalComparison:
    outcome_comparison = compare_agent_metric_results(
        previous_result=previous_report,
        current_result=current_report,
        metric_name="outcome",
    )
    capability_comparison = compare_agent_metric_results(
        previous_result=previous_report,
        current_result=current_report,
        metric_name="capability",
    )
    grounding_comparison = compare_agent_metric_results(
        previous_result=previous_report,
        current_result=current_report,
        metric_name="grounding",
    )

    return AgentEvalComparison(
        previous_run_id=previous_report["run_id"],
        current_run_id=current_report["run_id"],
        outcome=outcome_comparison,
        capability=capability_comparison,
        grounding=grounding_comparison,
    )


def list_agent_eval_report_files(
    reports_dir: Path,
) -> list[Path]:
    runs_dir = reports_dir / "runs"
    reports = sorted(runs_dir.glob("*.json"))

    return reports


def get_latest_agent_eval_report_files(
    reports_dir: Path,
) -> tuple[Path, Path]:
    report_files = list_agent_eval_report_files(reports_dir)

    if len(report_files) < 2:
        raise ValueError("At least two agent eval reports are required for comparison")

    return report_files[-2], report_files[-1]


def compare_latest_agent_eval_reports(
    reports_dir: Path,
) -> AgentEvalComparison:
    previous_file, current_file = get_latest_agent_eval_report_files(reports_dir)

    previous_report = load_agent_eval_report(previous_file)
    current_report = load_agent_eval_report(current_file)

    return compare_agent_eval_reports(
        previous_report=previous_report,
        current_report=current_report,
    )


def build_agent_eval_comparison_markdown_summary(
    comparison: AgentEvalComparison,
) -> str:
    lines: list[str] = []

    lines.append("## Comparison with previous run")
    lines.append("")
    lines.append(f"Previous: `{comparison['previous_run_id']}`")
    lines.append(f"Current: `{comparison['current_run_id']}`")

    lines.append("")
    lines.append("| Metric | Previous | Current | Delta |")
    lines.append("|---|---:|---:|---:|")

    lines.append(
        f"| Outcome | "
        f"{comparison['outcome']['previous_pass_rate']:.2%} | "
        f"{comparison['outcome']['current_pass_rate']:.2%} | "
        f"{comparison['outcome']['pass_rate_delta']:+.2%} |"
    )
    lines.append(
        f"| Capability | "
        f"{comparison['capability']['previous_pass_rate']:.2%} | "
        f"{comparison['capability']['current_pass_rate']:.2%} | "
        f"{comparison['capability']['pass_rate_delta']:+.2%} |"
    )
    lines.append(
        f"| Grounding | "
        f"{comparison['grounding']['previous_pass_rate']:.2%} | "
        f"{comparison['grounding']['current_pass_rate']:.2%} | "
        f"{comparison['grounding']['pass_rate_delta']:+.2%} |"
    )
    lines.append("")

    def append_case_changes(
        lines: list[str],
        metric_name: AgentMetricName,
        metric_comparison: AgentMetricComparison,
    ) -> None:
        if (
            not metric_comparison["worsened_case_ids"]
            and not metric_comparison["improved_case_ids"]
        ):
            return

        lines.append(f"### {metric_name.upper()} changes")
        lines.append("")

        if metric_comparison["improved_case_ids"]:
            lines.append("Improved:")
            for case_id in metric_comparison["improved_case_ids"]:
                lines.append(f"- {case_id}")

            lines.append("")

        if metric_comparison["worsened_case_ids"]:
            lines.append("Worsened:")
            for case_id in metric_comparison["worsened_case_ids"]:
                lines.append(f"- {case_id}")

            lines.append("")

    append_case_changes(lines, "outcome", comparison["outcome"])
    append_case_changes(lines, "capability", comparison["capability"])
    append_case_changes(lines, "grounding", comparison["grounding"])

    return "\n".join(lines)
