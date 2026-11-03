import json
import re
import unicodedata
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, TypedDict, cast

from pydantic import BaseModel

from prokat24_ai_lab.agents.types import AgentRunner, AgentRunResult, ToolCall, ToolExecution
from prokat24_ai_lab.evals.report_metadata import build_eval_report_metadata
from prokat24_ai_lab.evals.semantic_grading import (
    CriterionJudge,
    JudgeInput,
    Rubric,
    summarize_criterion_judge_result,
)
from prokat24_ai_lab.observability.context import current_eval_case_id


class AgentOutcomeEvalCase(TypedDict):
    id: str
    user_input: str
    expected_answer_fragments: list[str]
    forbidden_answer_fragments: list[str]
    required_tool_names: list[str]


class AgentOutcomeEvalResult(TypedDict):
    id: str
    trace_id: str | None
    user_input: str
    final_answer: str
    missing_fragments: list[str]
    unexpected_fragments: list[str]
    tool_calls: list[ToolCall]
    missing_tool_names: list[str]
    tool_executions: list[ToolExecution]
    passed: bool


class AgentCriterionResult(TypedDict):
    name: str
    passed: bool
    reason: str


class AgentCriterionEvalResult(TypedDict):
    user_input: str
    final_answer: str
    passed: bool
    failed_criteria: list[str]
    criteria: list[AgentCriterionResult]


class AgentCapabilityJudgeCase(TypedDict):
    id: str
    category: str
    question: str
    answer: str
    expected_passed: bool


class AgentCapabilityJudgeCaseModel(BaseModel):
    id: str
    category: str
    question: str
    answer: str
    expected_passed: bool


class AgentCapabilityJudgeResult(TypedDict):
    id: str
    category: str
    expected_passed: bool
    actual_passed: bool
    matched: bool
    failed_criteria: list[str]
    criteria: list[AgentCriterionResult]


class AgentCapabilityJudgeSummary(TypedDict):
    total_cases: int
    matched_cases: int
    accuracy: float


class AgentGroundingEvalResult(TypedDict):
    user_input: str
    final_answer: str
    passed: bool
    failed_criteria: list[str]
    criteria: list[AgentCriterionResult]


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

AGENT_CAPABILITY_RUBRIC: Rubric = {
    "name": "Agent capability boundaries",
    "criteria": [
        {
            "name": "supported_actions_only",
            "description": (
                "Evaluate only whether the assistant claims, implies, or offers "
                "that it itself can perform an action that is not supported by "
                "the available capabilities. "
                "\n\n"
                "Do not evaluate factual correctness, factual groundedness, "
                "or whether evidence for a factual statement is included in the "
                "context. Those are evaluated separately. A factual statement "
                "must not fail this criterion merely because supporting tool "
                "output is not shown here. "
                "\n\n"
                "Distinguish who performs the action. Instructions for the user "
                "to contact Prokat24, statements about actions performed by "
                "Prokat24 staff, links to external pages, and suggestions to "
                "confirm information with Prokat24 are allowed. Helping the user "
                "prepare questions, checklists, or wording for contacting "
                "Prokat24 is also allowed. Mentioning an unavailable action as "
                "something the user or Prokat24 should do does not mean that the "
                "assistant claims it can perform that action. "
                "\n\n"
                "Fail only when the assistant itself claims, implies, or offers "
                "to perform an unavailable action, such as making or modifying "
                "a booking, checking live availability, calculating an exact "
                "delivery quote for an address, checking current purchase "
                "inventory or purchase prices, determining which phone number "
                "serves a particular area, or retrieving product-specific "
                "service details that are not available to it."
            ),
        },
    ],
}

AGENT_CAPABILITY_CONTEXT = (
    "Available assistant capabilities:\n"
    "- Search the Prokat24 rental catalog.\n"
    "- Read rental product information such as rental prices and deposits.\n"
    "- Read published Prokat24 service and general business information, "
    "such as delivery rules, booking rules, contacts, addresses, rental "
    "conditions, and published information about services or sales.\n"
    "- Explain retrieved information, perform calculations based on available "
    "rental data, and help the user prepare questions or checklists.\n"
    "- Recommend that the user contact Prokat24 when an action must be "
    "completed or confirmed externally.\n"
    "\n"
    "Unavailable capabilities:\n"
    "- Make or modify a booking.\n"
    "- Check live product availability.\n"
    "- Calculate an exact delivery quote for an address.\n"
    "- Check current purchase inventory or purchase prices.\n"
    "- Determine which phone number should be used for a particular area.\n"
    "- Retrieve product-specific service information unless that information "
    "is already available in published tool output.\n"
    "\n"
    "This evaluation is only about action capability boundaries. "
    "It does not evaluate factual groundedness."
)

AGENT_GROUNDING_RUBRIC: Rubric = {
    "name": "Agent answer grounding",
    "criteria": [
        {
            "name": "grounded_facts_only",
            "description": (
                "Every factual claim about Prokat24 products, prices, deposits, "
                "delivery, booking rules, contacts, addresses, services, or other "
                "business-specific information in the assistant answer must be "
                "supported by the provided tool executions. "
                "Claims may be paraphrased and do not need to copy the tool output "
                "word for word. Calculations and conclusions directly derivable "
                "from the tool outputs are allowed. "
                "Do not fail merely because the answer omits information from the "
                "tool outputs. "
                "Do not evaluate whether the answer is relevant, complete, or "
                "within the assistant's capabilities; those are evaluated separately. "
                "Fail if the answer introduces a business-specific factual claim "
                "that cannot be supported or derived from the provided tool outputs."
            ),
        },
    ],
}


def normalize_eval_text(text: str) -> str:
    # Normalize compatible Unicode characters, e.g. ① -> 1.
    normalized_text = unicodedata.normalize("NFKC", text)

    # Ignore letter case differences during eval comparison.
    normalized_text = normalized_text.casefold()

    # Normalize different dash/minus characters to a regular hyphen.
    for dash in ("‐", "‒", "–", "—", "−"):
        normalized_text = normalized_text.replace(dash, "-")

    # Remove space used as thousands separator between digits.
    normalized_text = re.sub(
        r"(?<=\d)\s+(?=\d)",
        "",
        normalized_text,
    )

    # Collapse spaces, tabs, and line breaks into a single space.
    normalized_text = re.sub(r"\s+", " ", normalized_text)

    return normalized_text.strip()


def evaluate_agent_outcome_case(
    case: AgentOutcomeEvalCase,
    agent_result: AgentRunResult,
) -> AgentOutcomeEvalResult:
    user_input = case["user_input"]
    final_answer = agent_result["final_answer"]
    normalized_answer = normalize_eval_text(final_answer)

    missing_fragments = []
    unexpected_fragments = []
    missing_tool_names = []

    for fragment in case["expected_answer_fragments"]:
        normalized_fragment = normalize_eval_text(fragment)

        if normalized_fragment not in normalized_answer:
            missing_fragments.append(fragment)

    for fragment in case["forbidden_answer_fragments"]:
        normalized_fragment = normalize_eval_text(fragment)
        if normalized_fragment in normalized_answer:
            unexpected_fragments.append(fragment)

    actual_tool_names = [tool_call["tool_name"] for tool_call in agent_result["tool_calls"]]

    for tool_name in case["required_tool_names"]:
        if tool_name not in actual_tool_names:
            missing_tool_names.append(tool_name)

    passed = not missing_fragments and not unexpected_fragments and not missing_tool_names

    return AgentOutcomeEvalResult(
        id=case["id"],
        trace_id=None,
        user_input=user_input,
        final_answer=final_answer,
        missing_fragments=missing_fragments,
        unexpected_fragments=unexpected_fragments,
        tool_calls=agent_result["tool_calls"],
        missing_tool_names=missing_tool_names,
        tool_executions=agent_result["tool_executions"],
        passed=passed,
    )


def run_agent_outcome_evals(
    cases: list[AgentOutcomeEvalCase],
    agent_runner: AgentRunner,
) -> list[AgentOutcomeEvalResult]:
    agent_results = []

    for case in cases:
        case_id_token = current_eval_case_id.set(case["id"])

        try:
            agent_result = agent_runner(case["user_input"])
            agent_results.append(evaluate_agent_outcome_case(case, agent_result))

        finally:
            current_eval_case_id.reset(case_id_token)

    return agent_results


def load_agent_outcome_eval_cases(input_file: Path) -> list[AgentOutcomeEvalCase]:
    with open(input_file, encoding="utf-8") as file:
        cases: list[AgentOutcomeEvalCase] = json.load(file)

    return cases


def evaluate_capability_answer(
    question: str,
    answer: str,
    judge: CriterionJudge,
) -> AgentCriterionEvalResult:
    judge_input: JudgeInput = {
        "question": question,
        "context": AGENT_CAPABILITY_CONTEXT,
        "answer": answer,
    }

    criterion_result = judge(
        judge_input,
        AGENT_CAPABILITY_RUBRIC,
    )

    summary = summarize_criterion_judge_result(criterion_result)

    criteria: list[AgentCriterionResult] = [
        {
            "name": criterion.name,
            "passed": criterion.passed,
            "reason": criterion.reason,
        }
        for criterion in criterion_result.criteria
    ]

    return AgentCriterionEvalResult(
        user_input=question,
        final_answer=answer,
        passed=summary["passed"],
        failed_criteria=summary["failed_criteria"],
        criteria=criteria,
    )


def evaluate_agent_capability(
    result: AgentOutcomeEvalResult,
    judge: CriterionJudge,
) -> AgentCriterionEvalResult:
    return evaluate_capability_answer(
        question=result["user_input"],
        answer=result["final_answer"],
        judge=judge,
    )


def load_agent_capability_judge_cases(
    input_file: Path,
) -> list[AgentCapabilityJudgeCase]:
    with open(input_file, encoding="utf-8") as file:
        cases: list[AgentCapabilityJudgeCase] = json.load(file)

    return cases


def evaluate_agent_capability_judge_case(
    case: AgentCapabilityJudgeCase,
    judge: CriterionJudge,
) -> AgentCapabilityJudgeResult:
    question = case["question"]
    answer = case["answer"]

    capability_result = evaluate_capability_answer(
        question=question,
        answer=answer,
        judge=judge,
    )

    expected_passed = case["expected_passed"]
    actual_passed = capability_result["passed"]
    matched = expected_passed == actual_passed

    return {
        "id": case["id"],
        "category": case["category"],
        "expected_passed": expected_passed,
        "actual_passed": actual_passed,
        "matched": matched,
        "failed_criteria": capability_result["failed_criteria"],
        "criteria": capability_result["criteria"],
    }


def run_agent_capability_judge_evals(
    cases: list[AgentCapabilityJudgeCase],
    judge: CriterionJudge,
) -> list[AgentCapabilityJudgeResult]:
    results: list[AgentCapabilityJudgeResult] = []

    for case in cases:
        result = evaluate_agent_capability_judge_case(
            case=case,
            judge=judge,
        )
        results.append(result)

    return results


def summarize_agent_capability_judge_results(
    results: list[AgentCapabilityJudgeResult],
) -> AgentCapabilityJudgeSummary:
    if not results:
        return AgentCapabilityJudgeSummary(
            total_cases=0,
            matched_cases=0,
            accuracy=0.0,
        )

    total_cases = len(results)
    matched_cases = 0

    for result in results:
        if result["matched"]:
            matched_cases += 1

    accuracy = matched_cases / total_cases

    return AgentCapabilityJudgeSummary(
        total_cases=total_cases,
        matched_cases=matched_cases,
        accuracy=accuracy,
    )


def build_agent_grounding_context(
    tool_executions: list[ToolExecution],
) -> str:
    lines: list[str] = []

    for index, execution in enumerate(tool_executions, start=1):
        lines.append(f"[Tool execution {index}]")
        lines.append(f"Tool: {execution['tool_call']['tool_name']}")
        lines.append(f"Arguments: {execution['tool_call']['arguments']}")
        lines.append("Output:")
        lines.append(execution["output"])
        lines.append("")

    return "\n".join(lines)


def evaluate_agent_grounding(
    result: AgentOutcomeEvalResult,
    judge: CriterionJudge,
) -> AgentGroundingEvalResult:
    context = build_agent_grounding_context(result["tool_executions"])

    judge_input: JudgeInput = {
        "question": result["user_input"],
        "context": context,
        "answer": result["final_answer"],
    }

    criterion_result = judge(
        judge_input,
        AGENT_GROUNDING_RUBRIC,
    )

    summary = summarize_criterion_judge_result(criterion_result)

    criteria: list[AgentCriterionResult] = [
        {
            "name": criterion.name,
            "passed": criterion.passed,
            "reason": criterion.reason,
        }
        for criterion in criterion_result.criteria
    ]

    return AgentGroundingEvalResult(
        user_input=result["user_input"],
        final_answer=result["final_answer"],
        passed=summary["passed"],
        failed_criteria=summary["failed_criteria"],
        criteria=criteria,
    )


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


def load_agent_eval_report(
    report_file: Path,
) -> AgentEvalReport:
    with open(report_file, encoding="utf-8") as file:
        report = json.load(file)

    return cast(AgentEvalReport, report)


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
