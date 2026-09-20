import json
import re
import unicodedata
from pathlib import Path
from typing import TypedDict

from openai import BaseModel

from prokat24_ai_lab.agents.types import AgentRunner, AgentRunResult, ToolCall, ToolExecution
from prokat24_ai_lab.evals.semantic_grading import (
    CriterionJudge,
    JudgeInput,
    Rubric,
    summarize_criterion_judge_result,
)


class AgentOutcomeEvalCase(TypedDict):
    user_input: str
    expected_answer_fragments: list[str]
    forbidden_answer_fragments: list[str]
    required_tool_names: list[str]


class AgentOutcomeEvalResult(TypedDict):
    user_input: str
    final_answer: str
    missing_fragments: list[str]
    unexpected_fragments: list[str]
    tool_calls: list[ToolCall]
    missing_tool_names: list[str]
    tool_executions: list[ToolExecution]
    passed: bool


class AgentOutcomeEvalSummary(TypedDict):
    total_cases: int
    passed_cases: int
    pass_rate: float


class AgentCapabilityCriterionResult(TypedDict):
    name: str
    passed: bool
    reason: str


class AgentCriterionEvalResult(TypedDict):
    user_input: str
    final_answer: str
    passed: bool
    failed_criteria: list[str]
    criteria: list[AgentCapabilityCriterionResult]


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
    criteria: list[AgentCapabilityCriterionResult]


class AgentCapabilityJudgeSummary(TypedDict):
    total_cases: int
    matched_cases: int
    accuracy: float


class AgentGroundingEvalResult(TypedDict):
    user_input: str
    final_answer: str
    passed: bool
    failed_criteria: list[str]
    criteria: list[AgentCapabilityCriterionResult]


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
        agent_result = agent_runner(case["user_input"])
        agent_results.append(evaluate_agent_outcome_case(case, agent_result))

    return agent_results


def summarize_agent_outcome_eval_results(
    results: list[AgentOutcomeEvalResult],
) -> AgentOutcomeEvalSummary:
    total_cases = len(results)

    if total_cases == 0:
        return AgentOutcomeEvalSummary(
            total_cases=0,
            passed_cases=0,
            pass_rate=0.0,
        )

    passed_cases = 0

    for result in results:
        if result["passed"]:
            passed_cases += 1

    pass_rate = passed_cases / total_cases

    return AgentOutcomeEvalSummary(
        total_cases=total_cases,
        passed_cases=passed_cases,
        pass_rate=pass_rate,
    )


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

    criteria: list[AgentCapabilityCriterionResult] = [
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

    criteria: list[AgentCapabilityCriterionResult] = [
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
