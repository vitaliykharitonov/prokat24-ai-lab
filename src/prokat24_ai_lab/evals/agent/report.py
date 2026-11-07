import json
from collections.abc import Sequence
from pathlib import Path
from typing import TypedDict, cast

from prokat24_ai_lab.agents.types import ToolCall, ToolExecution
from prokat24_ai_lab.evals.agent.grounding import AgentGroundingEvalResult
from prokat24_ai_lab.evals.agent.types import (
    AgentCriterionEvalResult,
    AgentCriterionResult,
    AgentOutcomeEvalResult,
)
from prokat24_ai_lab.evals.report_metadata import build_eval_report_metadata


class AgentEvalConfiguration(TypedDict):
    agent_model: str
    judge_model: str
    max_agent_steps: int
    eval_cases_file: str


class AgentMetricSummary(TypedDict):
    total_cases: int
    passed_cases: int
    pass_rate: float


class AgentEvalSummary(TypedDict):
    outcome: AgentMetricSummary
    capability: AgentMetricSummary
    grounding: AgentMetricSummary


class AgentOutcomeCaseEvaluation(TypedDict):
    passed: bool
    missing_fragments: list[str]
    unexpected_fragments: list[str]
    missing_tool_names: list[str]


class AgentCriterionCaseEvaluation(TypedDict):
    passed: bool
    failed_criteria: list[str]
    criteria: list[AgentCriterionResult]


class AgentEvalCaseResult(TypedDict):
    id: str
    trace_id: str | None
    user_input: str
    final_answer: str
    tool_calls: list[ToolCall]
    tool_executions: list[ToolExecution]
    outcome: AgentOutcomeCaseEvaluation
    capability: AgentCriterionCaseEvaluation
    grounding: AgentCriterionCaseEvaluation


class AgentEvalReport(TypedDict):
    report_type: str
    schema_version: int
    created_at: str
    git_commit: str | None
    git_dirty: bool | None
    run_id: str
    configuration: AgentEvalConfiguration
    summary: AgentEvalSummary
    results: list[AgentEvalCaseResult]


class AgentPassFailResult(TypedDict):
    passed: bool


def build_agent_eval_case_results(
    results: list[AgentOutcomeEvalResult],
    capability_results: list[AgentCriterionEvalResult],
    grounding_results: list[AgentGroundingEvalResult],
) -> list[AgentEvalCaseResult]:
    eval_results: list[AgentEvalCaseResult] = []

    for result, capability_result, grounding_result in zip(
        results,
        capability_results,
        grounding_results,
        strict=True,
    ):
        eval_results.append(
            AgentEvalCaseResult(
                id=result["id"],
                trace_id=result["trace_id"],
                user_input=result["user_input"],
                final_answer=result["final_answer"],
                tool_calls=result["tool_calls"],
                tool_executions=result["tool_executions"],
                outcome=AgentOutcomeCaseEvaluation(
                    passed=result["passed"],
                    missing_fragments=result["missing_fragments"],
                    unexpected_fragments=result["unexpected_fragments"],
                    missing_tool_names=result["missing_tool_names"],
                ),
                capability=AgentCriterionCaseEvaluation(
                    passed=capability_result["passed"],
                    failed_criteria=capability_result["failed_criteria"],
                    criteria=capability_result["criteria"],
                ),
                grounding=AgentCriterionCaseEvaluation(
                    passed=grounding_result["passed"],
                    failed_criteria=grounding_result["failed_criteria"],
                    criteria=grounding_result["criteria"],
                ),
            )
        )

    return eval_results


def summarize_agent_metric_results(
    results: Sequence[AgentPassFailResult],
) -> AgentMetricSummary:
    total_cases = len(results)

    if total_cases == 0:
        return AgentMetricSummary(
            total_cases=0,
            passed_cases=0,
            pass_rate=0.0,
        )

    passed_cases = 0

    for result in results:
        if result["passed"]:
            passed_cases += 1

    pass_rate = passed_cases / total_cases

    return AgentMetricSummary(
        total_cases=total_cases,
        passed_cases=passed_cases,
        pass_rate=pass_rate,
    )


def build_agent_eval_summary(
    outcome_results: Sequence[AgentOutcomeEvalResult],
    capability_results: Sequence[AgentCriterionEvalResult],
    grounding_results: Sequence[AgentGroundingEvalResult],
) -> AgentEvalSummary:
    outcome = summarize_agent_metric_results(outcome_results)
    capability = summarize_agent_metric_results(capability_results)
    grounding = summarize_agent_metric_results(grounding_results)

    return AgentEvalSummary(
        outcome=outcome,
        capability=capability,
        grounding=grounding,
    )


def build_agent_eval_report(
    outcome_results: list[AgentOutcomeEvalResult],
    capability_results: list[AgentCriterionEvalResult],
    grounding_results: list[AgentGroundingEvalResult],
    configuration: AgentEvalConfiguration,
    run_id: str,
    created_at: str,
    git_commit: str | None = None,
    git_dirty: bool | None = None,
) -> AgentEvalReport:
    metadata = build_eval_report_metadata(
        report_type="agent_eval_report",
        created_at=created_at,
        git_commit=git_commit,
        git_dirty=git_dirty,
    )
    summary = build_agent_eval_summary(
        outcome_results=outcome_results,
        capability_results=capability_results,
        grounding_results=grounding_results,
    )
    results = build_agent_eval_case_results(
        results=outcome_results,
        capability_results=capability_results,
        grounding_results=grounding_results,
    )

    return AgentEvalReport(
        **metadata,
        run_id=run_id,
        configuration=configuration,
        summary=summary,
        results=results,
    )


def save_agent_eval_report(
    report: AgentEvalReport,
    output_dir: Path,
) -> tuple[Path, Path]:
    historical_path = output_dir / "runs"
    historical_file = historical_path / f"{report['run_id']}.json"
    latest_file = output_dir / "latest.json"

    historical_path.mkdir(parents=True, exist_ok=True)

    with open(historical_file, "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    with open(latest_file, "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    return historical_file, latest_file


def load_agent_eval_report(
    report_file: Path,
) -> AgentEvalReport:
    with open(report_file, encoding="utf-8") as file:
        report = json.load(file)

    return cast(AgentEvalReport, report)


def build_agent_eval_markdown_summary(report: AgentEvalReport) -> str:
    lines: list[str] = []

    lines.append("## Agent Eval Summary")
    lines.append("")

    lines.append("| Metric | Passed | Total | Pass rate |")
    lines.append("|---|---:|---:|---:|")

    summary = report["summary"]

    lines.append(
        f"| Outcome | "
        f"{summary['outcome']['passed_cases']} | "
        f"{summary['outcome']['total_cases']} | "
        f"{summary['outcome']['pass_rate']:.2%} |"
    )
    lines.append(
        f"| Capability | "
        f"{summary['capability']['passed_cases']} | "
        f"{summary['capability']['total_cases']} | "
        f"{summary['capability']['pass_rate']:.2%} |"
    )
    lines.append(
        f"| Grounding | "
        f"{summary['grounding']['passed_cases']} | "
        f"{summary['grounding']['total_cases']} | "
        f"{summary['grounding']['pass_rate']:.2%} |"
    )

    lines.append("")
    lines.append(f"Run ID: `{report['run_id']}`")

    git_commit = report["git_commit"] or "N/A"

    lines.append(f"Git commit: `{git_commit}`")

    return "\n".join(lines)
