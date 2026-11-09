import json
from pathlib import Path
from typing import TypedDict

from pydantic import BaseModel

from prokat24_ai_lab.evals.agent.types import (
    AgentCriterionEvalResult,
    AgentCriterionResult,
    AgentOutcomeEvalResult,
)
from prokat24_ai_lab.evals.grading.judge import (
    CriterionJudge,
    JudgeInput,
    Rubric,
    summarize_criterion_judge_result,
)


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
