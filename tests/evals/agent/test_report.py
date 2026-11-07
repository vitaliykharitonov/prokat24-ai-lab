import json
from pathlib import Path

from prokat24_ai_lab.evals.agent.grounding import AgentGroundingEvalResult
from prokat24_ai_lab.evals.agent.report import (
    AgentEvalConfiguration,
    AgentEvalReport,
    AgentEvalSummary,
    AgentMetricSummary,
    build_agent_eval_markdown_summary,
    build_agent_eval_report,
    load_agent_eval_report,
    save_agent_eval_report,
)
from prokat24_ai_lab.evals.agent.types import AgentCriterionEvalResult, AgentOutcomeEvalResult


def test_build_agent_eval_report() -> None:
    outcome_results: list[AgentOutcomeEvalResult] = [
        AgentOutcomeEvalResult(
            id="kyiv_delivery",
            trace_id="trace-123",
            user_input="Скільки коштує доставка по Києву?",
            final_answer="Доставка по Києву — від 200 грн.",
            missing_fragments=[],
            unexpected_fragments=[],
            tool_calls=[
                {
                    "tool_name": "search_service_info",
                    "arguments": {"query": "доставка по Києву"},
                }
            ],
            missing_tool_names=[],
            tool_executions=[
                {
                    "tool_call": {
                        "tool_name": "search_service_info",
                        "arguments": {"query": "доставка по Києву"},
                    },
                    "output": "Доставка по Києву — від 200 грн.",
                }
            ],
            passed=True,
        )
    ]

    capability_results: list[AgentCriterionEvalResult] = [
        AgentCriterionEvalResult(
            user_input="Скільки коштує доставка по Києву?",
            final_answer="Доставка по Києву — від 200 грн.",
            passed=True,
            failed_criteria=[],
            criteria=[
                {
                    "name": "supported_actions_only",
                    "passed": True,
                    "reason": "The answer stays within supported capabilities.",
                }
            ],
        )
    ]

    grounding_results: list[AgentGroundingEvalResult] = [
        AgentGroundingEvalResult(
            user_input="Скільки коштує доставка по Києву?",
            final_answer="Доставка по Києву — від 200 грн.",
            passed=True,
            failed_criteria=[],
            criteria=[
                {
                    "name": "grounded_facts_only",
                    "passed": True,
                    "reason": "The delivery price is supported by the tool output.",
                }
            ],
        )
    ]

    configuration = AgentEvalConfiguration(
        agent_model="gpt-5-mini",
        judge_model="gpt-5.4-mini-2026-03-17",
        max_agent_steps=5,
        eval_cases_file="data/agent_outcome_eval_cases.json",
    )

    report = build_agent_eval_report(
        outcome_results=outcome_results,
        capability_results=capability_results,
        grounding_results=grounding_results,
        configuration=configuration,
        run_id="20260923T120000Z_abc1234",
        created_at="2026-09-23T12:00:00+00:00",
        git_commit="abc1234",
        git_dirty=False,
    )

    assert report["results"][0]["trace_id"] == "trace-123"
    assert report["report_type"] == "agent_eval_report"
    assert report["schema_version"] == 1
    assert report["created_at"] == "2026-09-23T12:00:00+00:00"
    assert report["git_commit"] == "abc1234"
    assert report["git_dirty"] is False
    assert report["run_id"] == "20260923T120000Z_abc1234"

    assert report["configuration"] == configuration

    assert report["summary"] == {
        "outcome": {
            "total_cases": 1,
            "passed_cases": 1,
            "pass_rate": 1.0,
        },
        "capability": {
            "total_cases": 1,
            "passed_cases": 1,
            "pass_rate": 1.0,
        },
        "grounding": {
            "total_cases": 1,
            "passed_cases": 1,
            "pass_rate": 1.0,
        },
    }

    assert len(report["results"]) == 1

    case_result = report["results"][0]

    assert case_result["id"] == "kyiv_delivery"
    assert case_result["outcome"]["passed"] is True
    assert case_result["capability"]["passed"] is True
    assert case_result["grounding"]["passed"] is True


def test_save_agent_eval_report_saves_historical_and_latest_files(tmp_path: Path) -> None:
    report = AgentEvalReport(
        report_type="agent_eval_report",
        schema_version=1,
        created_at="2026-09-23T18:00:00+00:00",
        git_commit="abc1234",
        git_dirty=False,
        run_id="20260923T180000Z_abc1234",
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
                passed_cases=1,
                pass_rate=1.0,
            ),
            grounding=AgentMetricSummary(
                total_cases=1,
                passed_cases=1,
                pass_rate=1.0,
            ),
        ),
        results=[],
    )
    output_dir = tmp_path / "agent"
    historical_file = output_dir / "runs" / "20260923T180000Z_abc1234.json"
    latest_file = output_dir / "latest.json"

    saved_historical_file, saved_latest_file = save_agent_eval_report(report, output_dir)

    with open(historical_file, encoding="utf-8") as file:
        historical_report = json.load(file)

    with open(latest_file, encoding="utf-8") as file:
        latest_report = json.load(file)

    assert historical_report == latest_report == report
    assert saved_historical_file == historical_file
    assert saved_latest_file == latest_file


def test_load_agent_eval_report_reads_json_file(tmp_path: Path) -> None:
    report = AgentEvalReport(
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
        results=[],
    )

    input_file = tmp_path / "agent_eval_report.json"

    with open(input_file, "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    loaded_data = load_agent_eval_report(input_file)

    assert loaded_data == report


def test_build_agent_eval_markdown_summary_formats_report() -> None:
    report = AgentEvalReport(
        report_type="agent_eval_report",
        schema_version=1,
        created_at="2026-09-26T18:00:00+00:00",
        git_commit="abc1234",
        git_dirty=False,
        run_id="20260926T180000Z_abc1234",
        configuration=AgentEvalConfiguration(
            agent_model="gpt-5-mini",
            judge_model="gpt-5.4-mini-2026-03-17",
            max_agent_steps=5,
            eval_cases_file="data/agent_outcome_eval_cases.json",
        ),
        summary=AgentEvalSummary(
            outcome=AgentMetricSummary(
                total_cases=10,
                passed_cases=9,
                pass_rate=0.9,
            ),
            capability=AgentMetricSummary(
                total_cases=10,
                passed_cases=10,
                pass_rate=1.0,
            ),
            grounding=AgentMetricSummary(
                total_cases=10,
                passed_cases=8,
                pass_rate=0.8,
            ),
        ),
        results=[],
    )
    expected = (
        "## Agent Eval Summary\n"
        "\n"
        "| Metric | Passed | Total | Pass rate |\n"
        "|---|---:|---:|---:|\n"
        "| Outcome | 9 | 10 | 90.00% |\n"
        "| Capability | 10 | 10 | 100.00% |\n"
        "| Grounding | 8 | 10 | 80.00% |\n"
        "\n"
        "Run ID: `20260926T180000Z_abc1234`\n"
        "Git commit: `abc1234`"
    )

    markdown_summary = build_agent_eval_markdown_summary(report)

    assert markdown_summary == expected
