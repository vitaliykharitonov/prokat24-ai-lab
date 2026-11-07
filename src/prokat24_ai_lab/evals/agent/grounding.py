from typing import TypedDict

from prokat24_ai_lab.agents.types import ToolExecution
from prokat24_ai_lab.evals.agent.types import AgentCriterionResult, AgentOutcomeEvalResult
from prokat24_ai_lab.evals.grading.semantic import (
    CriterionJudge,
    JudgeInput,
    Rubric,
    summarize_criterion_judge_result,
)


class AgentGroundingEvalResult(TypedDict):
    user_input: str
    final_answer: str
    passed: bool
    failed_criteria: list[str]
    criteria: list[AgentCriterionResult]


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
