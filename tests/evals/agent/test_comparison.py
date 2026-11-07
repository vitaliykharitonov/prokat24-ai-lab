from pathlib import Path

import pytest

from prokat24_ai_lab.evals.agent.comparison import (
    AgentEvalComparison,
    AgentMetricComparison,
    build_agent_eval_comparison_markdown_summary,
    compare_agent_eval_reports,
    compare_agent_metric_results,
    compare_latest_agent_eval_reports,
    get_latest_agent_eval_report_files,
    list_agent_eval_report_files,
)
from prokat24_ai_lab.evals.agent.report import (
    AgentCriterionCaseEvaluation,
    AgentEvalCaseResult,
    AgentEvalConfiguration,
    AgentEvalReport,
    AgentEvalSummary,
    AgentMetricSummary,
    AgentOutcomeCaseEvaluation,
    save_agent_eval_report,
)


def test_compare_agent_metric_results_classifies_case_changes() -> None:
    previous_report = AgentEvalReport(
        report_type="agent_eval_report",
        schema_version=1,
        created_at="2026-09-23T18:00:00+00:00",
        git_commit="abc1234",
        git_dirty=False,
        run_id="previous_run",
        configuration=AgentEvalConfiguration(
            agent_model="gpt-5-mini",
            judge_model="gpt-5.4-mini-2026-03-17",
            max_agent_steps=5,
            eval_cases_file="data/agent_outcome_eval_cases.json",
        ),
        summary=AgentEvalSummary(
            outcome=AgentMetricSummary(
                total_cases=4,
                passed_cases=2,
                pass_rate=0.5,
            ),
            capability=AgentMetricSummary(
                total_cases=0,
                passed_cases=0,
                pass_rate=0.0,
            ),
            grounding=AgentMetricSummary(
                total_cases=0,
                passed_cases=0,
                pass_rate=0.0,
            ),
        ),
        results=[
            {
                "id": "improved_case",
                "outcome": {"passed": False},
            },
            {
                "id": "worsened_case",
                "outcome": {"passed": True},
            },
            {
                "id": "unchanged_case",
                "outcome": {"passed": True},
            },
            {
                "id": "removed_case",
                "outcome": {"passed": False},
            },
        ],
    )
    current_report = AgentEvalReport(
        report_type="agent_eval_report",
        schema_version=1,
        created_at="2026-09-24T18:00:00+00:00",
        git_commit="def5678",
        git_dirty=False,
        run_id="current_run",
        configuration=AgentEvalConfiguration(
            agent_model="gpt-5-mini",
            judge_model="gpt-5.4-mini-2026-03-17",
            max_agent_steps=5,
            eval_cases_file="data/agent_outcome_eval_cases.json",
        ),
        summary=AgentEvalSummary(
            outcome=AgentMetricSummary(
                total_cases=4,
                passed_cases=3,
                pass_rate=0.75,
            ),
            capability=AgentMetricSummary(
                total_cases=0,
                passed_cases=0,
                pass_rate=0.0,
            ),
            grounding=AgentMetricSummary(
                total_cases=0,
                passed_cases=0,
                pass_rate=0.0,
            ),
        ),
        results=[
            {
                "id": "improved_case",
                "outcome": {"passed": True},
            },
            {
                "id": "worsened_case",
                "outcome": {"passed": False},
            },
            {
                "id": "unchanged_case",
                "outcome": {"passed": True},
            },
            {
                "id": "added_case",
                "outcome": {"passed": True},
            },
        ],
    )

    comparison = compare_agent_metric_results(previous_report, current_report, "outcome")

    assert comparison["previous_pass_rate"] == 0.5
    assert comparison["current_pass_rate"] == 0.75
    assert comparison["pass_rate_delta"] == 0.25

    assert comparison["improved_case_ids"] == ["improved_case"]
    assert comparison["worsened_case_ids"] == ["worsened_case"]
    assert comparison["unchanged_case_ids"] == ["unchanged_case"]
    assert comparison["added_case_ids"] == ["added_case"]
    assert comparison["removed_case_ids"] == ["removed_case"]


def test_compare_agent_eval_reports_compares_all_metrics() -> None:
    previous_report = AgentEvalReport(
        report_type="agent_eval_report",
        schema_version=1,
        created_at="2026-09-23T18:00:00+00:00",
        git_commit="abc1234",
        git_dirty=False,
        run_id="previous_run",
        configuration=AgentEvalConfiguration(
            agent_model="gpt-5-mini",
            judge_model="gpt-5.4-mini-2026-03-17",
            max_agent_steps=5,
            eval_cases_file="data/agent_outcome_eval_cases.json",
        ),
        summary=AgentEvalSummary(
            outcome=AgentMetricSummary(
                total_cases=1,
                passed_cases=0,
                pass_rate=0.0,
            ),
            capability=AgentMetricSummary(
                total_cases=1,
                passed_cases=1,
                pass_rate=1.0,
            ),
            grounding=AgentMetricSummary(
                total_cases=1,
                passed_cases=1,
                pass_rate=1.0,
            ),
        ),
        results=[
            AgentEvalCaseResult(
                id="test_case",
                user_input="Test question",
                final_answer="Test answer",
                tool_calls=[],
                tool_executions=[],
                outcome=AgentOutcomeCaseEvaluation(
                    passed=False,
                    missing_fragments=[],
                    unexpected_fragments=[],
                    missing_tool_names=[],
                ),
                capability=AgentCriterionCaseEvaluation(
                    passed=True,
                    failed_criteria=[],
                    criteria=[],
                ),
                grounding=AgentCriterionCaseEvaluation(
                    passed=True,
                    failed_criteria=[],
                    criteria=[],
                ),
            ),
        ],
    )
    current_report = AgentEvalReport(
        report_type="agent_eval_report",
        schema_version=1,
        created_at="2026-09-24T18:00:00+00:00",
        git_commit="def5678",
        git_dirty=False,
        run_id="current_run",
        configuration=AgentEvalConfiguration(
            agent_model="gpt-5-mini",
            judge_model="gpt-5.4-mini-2026-03-17",
            max_agent_steps=5,
            eval_cases_file="data/agent_outcome_eval_cases.json",
        ),
        summary=AgentEvalSummary(
            outcome=AgentMetricSummary(
                total_cases=1,
                passed_cases=1,
                pass_rate=1.0,
            ),
            capability=AgentMetricSummary(
                total_cases=1,
                passed_cases=0,
                pass_rate=0.0,
            ),
            grounding=AgentMetricSummary(
                total_cases=1,
                passed_cases=1,
                pass_rate=1.0,
            ),
        ),
        results=[
            AgentEvalCaseResult(
                id="test_case",
                user_input="Test question",
                final_answer="Test answer",
                tool_calls=[],
                tool_executions=[],
                outcome=AgentOutcomeCaseEvaluation(
                    passed=True,
                    missing_fragments=[],
                    unexpected_fragments=[],
                    missing_tool_names=[],
                ),
                capability=AgentCriterionCaseEvaluation(
                    passed=False,
                    failed_criteria=["supported_actions_only"],
                    criteria=[],
                ),
                grounding=AgentCriterionCaseEvaluation(
                    passed=True,
                    failed_criteria=[],
                    criteria=[],
                ),
            ),
        ],
    )

    reports_comparison = compare_agent_eval_reports(previous_report, current_report)

    assert reports_comparison["previous_run_id"] == "previous_run"
    assert reports_comparison["current_run_id"] == "current_run"

    assert reports_comparison["outcome"]["improved_case_ids"] == ["test_case"]
    assert reports_comparison["outcome"]["pass_rate_delta"] == 1.0

    assert reports_comparison["capability"]["worsened_case_ids"] == ["test_case"]
    assert reports_comparison["capability"]["pass_rate_delta"] == -1.0

    assert reports_comparison["grounding"]["unchanged_case_ids"] == ["test_case"]
    assert reports_comparison["grounding"]["pass_rate_delta"] == 0.0


def test_list_agent_eval_report_files_returns_sorted_json_files(tmp_path: Path) -> None:
    reports_dir = tmp_path / "agent"
    runs_dir = reports_dir / "runs"
    runs_dir.mkdir(parents=True)

    first_file = runs_dir / "20260923T120000Z_aaa1111.json"
    second_file = runs_dir / "20260924T120000Z_bbb2222.json"
    third_file = runs_dir / "20260925T120000Z_ccc3333.json"
    ignored_file = runs_dir / "notes.txt"

    third_file.write_text("{}", encoding="utf-8")
    first_file.write_text("{}", encoding="utf-8")
    ignored_file.write_text("ignore", encoding="utf-8")
    second_file.write_text("{}", encoding="utf-8")

    report_files = list_agent_eval_report_files(reports_dir)

    assert report_files == [
        first_file,
        second_file,
        third_file,
    ]


def test_get_latest_agent_eval_report_files_returns_previous_and_current(
    tmp_path: Path,
) -> None:
    reports_dir = tmp_path / "agent"
    runs_dir = reports_dir / "runs"
    runs_dir.mkdir(parents=True)

    first_file = runs_dir / "20260923T120000Z_aaa1111.json"
    second_file = runs_dir / "20260924T120000Z_bbb2222.json"
    third_file = runs_dir / "20260925T120000Z_ccc3333.json"

    first_file.write_text("{}", encoding="utf-8")
    third_file.write_text("{}", encoding="utf-8")
    second_file.write_text("{}", encoding="utf-8")

    previous_file, current_file = get_latest_agent_eval_report_files(reports_dir)

    assert previous_file == second_file
    assert current_file == third_file


def test_get_latest_agent_eval_report_files_raises_for_less_than_two_reports(
    tmp_path: Path,
) -> None:
    reports_dir = tmp_path / "agent"
    runs_dir = reports_dir / "runs"
    runs_dir.mkdir(parents=True)

    only_file = runs_dir / "20260925T120000Z_ccc3333.json"
    only_file.write_text("{}", encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="At least two agent eval reports are required for comparison",
    ):
        get_latest_agent_eval_report_files(reports_dir)


def test_compare_latest_agent_eval_reports_compares_two_latest_runs(
    tmp_path: Path,
) -> None:
    reports_dir = tmp_path / "agent"

    first_report = AgentEvalReport(
        report_type="agent_eval_report",
        schema_version=1,
        created_at="2026-09-23T12:00:00+00:00",
        git_commit="aaa1111",
        git_dirty=False,
        run_id="20260923T120000Z_aaa1111",
        configuration=AgentEvalConfiguration(
            agent_model="gpt-5-mini",
            judge_model="gpt-5.4-mini-2026-03-17",
            max_agent_steps=5,
            eval_cases_file="data/agent_outcome_eval_cases.json",
        ),
        summary=AgentEvalSummary(
            outcome=AgentMetricSummary(total_cases=0, passed_cases=0, pass_rate=0.0),
            capability=AgentMetricSummary(total_cases=0, passed_cases=0, pass_rate=0.0),
            grounding=AgentMetricSummary(total_cases=0, passed_cases=0, pass_rate=0.0),
        ),
        results=[],
    )

    second_report = first_report.copy()
    second_report["created_at"] = "2026-09-24T12:00:00+00:00"
    second_report["git_commit"] = "bbb2222"
    second_report["run_id"] = "20260924T120000Z_bbb2222"

    third_report = first_report.copy()
    third_report["created_at"] = "2026-09-25T12:00:00+00:00"
    third_report["git_commit"] = "ccc3333"
    third_report["run_id"] = "20260925T120000Z_ccc3333"

    save_agent_eval_report(first_report, reports_dir)
    save_agent_eval_report(third_report, reports_dir)
    save_agent_eval_report(second_report, reports_dir)

    comparison = compare_latest_agent_eval_reports(reports_dir)

    assert comparison["previous_run_id"] == "20260924T120000Z_bbb2222"
    assert comparison["current_run_id"] == "20260925T120000Z_ccc3333"


def test_build_agent_eval_comparison_markdown_summary_formats_changes() -> None:
    comparison = AgentEvalComparison(
        previous_run_id="previous_run",
        current_run_id="current_run",
        outcome=AgentMetricComparison(
            previous_pass_rate=0.8,
            current_pass_rate=0.9,
            pass_rate_delta=0.1,
            improved_case_ids=["equipment_purchase"],
            worsened_case_ids=[],
            unchanged_case_ids=["case_1"],
            added_case_ids=[],
            removed_case_ids=[],
        ),
        capability=AgentMetricComparison(
            previous_pass_rate=1.0,
            current_pass_rate=0.9,
            pass_rate_delta=-0.1,
            improved_case_ids=[],
            worsened_case_ids=["booking_rules"],
            unchanged_case_ids=["case_1"],
            added_case_ids=[],
            removed_case_ids=[],
        ),
        grounding=AgentMetricComparison(
            previous_pass_rate=0.8,
            current_pass_rate=0.8,
            pass_rate_delta=0.0,
            improved_case_ids=[],
            worsened_case_ids=[],
            unchanged_case_ids=["case_1"],
            added_case_ids=[],
            removed_case_ids=[],
        ),
    )

    expected = (
        "## Comparison with previous run\n"
        "\n"
        "Previous: `previous_run`\n"
        "Current: `current_run`\n"
        "\n"
        "| Metric | Previous | Current | Delta |\n"
        "|---|---:|---:|---:|\n"
        "| Outcome | 80.00% | 90.00% | +10.00% |\n"
        "| Capability | 100.00% | 90.00% | -10.00% |\n"
        "| Grounding | 80.00% | 80.00% | +0.00% |\n"
        "\n"
        "### OUTCOME changes\n"
        "\n"
        "Improved:\n"
        "- equipment_purchase\n"
        "\n"
        "### CAPABILITY changes\n"
        "\n"
        "Worsened:\n"
        "- booking_rules\n"
    )

    markdown = build_agent_eval_comparison_markdown_summary(comparison)

    assert markdown == expected
