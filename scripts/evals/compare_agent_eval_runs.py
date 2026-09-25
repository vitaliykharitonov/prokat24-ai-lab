from prokat24_ai_lab.evals.agent_evals import (
    AgentMetricComparison,
    AgentMetricName,
    compare_latest_agent_eval_reports,
)
from prokat24_ai_lab.project_paths import AGENT_EVAL_REPORTS_DIR


def print_agent_metric_comparison(
    metric_name: AgentMetricName,
    comparison: AgentMetricComparison,
) -> None:
    print(f"\n{metric_name.upper()}:")
    print(
        f"Pass rate: {comparison['previous_pass_rate']:.2%} -> "
        f"{comparison['current_pass_rate']:.2%} ({comparison['pass_rate_delta']:+.2%})"
    )
    print(f"Improved: {len(comparison['improved_case_ids'])}")
    print(f"Worsened: {len(comparison['worsened_case_ids'])}")
    print(f"Unchanged: {len(comparison['unchanged_case_ids'])}")
    print(f"Added: {len(comparison['added_case_ids'])}")
    print(f"Removed: {len(comparison['removed_case_ids'])}")

    if comparison["improved_case_ids"]:
        print("\nImproved cases:")
        for case_id in comparison["improved_case_ids"]:
            print(f"- {case_id}")

    if comparison["worsened_case_ids"]:
        print("\nWorsened cases:")
        for case_id in comparison["worsened_case_ids"]:
            print(f"- {case_id}")

    if comparison["added_case_ids"]:
        print("\nAdded cases:")
        for case_id in comparison["added_case_ids"]:
            print(f"- {case_id}")

    if comparison["removed_case_ids"]:
        print("\nRemoved cases:")
        for case_id in comparison["removed_case_ids"]:
            print(f"- {case_id}")


def main() -> None:
    comparisons = compare_latest_agent_eval_reports(AGENT_EVAL_REPORTS_DIR)

    print("\nAGENT EVAL RUN COMPARISON\n")
    print(f"Previous: {comparisons['previous_run_id']}")
    print(f"Current: {comparisons['current_run_id']}")

    print_agent_metric_comparison("outcome", comparisons["outcome"])
    print_agent_metric_comparison("capability", comparisons["capability"])
    print_agent_metric_comparison("grounding", comparisons["grounding"])


if __name__ == "__main__":
    main()
