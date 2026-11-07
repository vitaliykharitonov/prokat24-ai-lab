import json
import re
import unicodedata
from pathlib import Path
from typing import TypedDict

from prokat24_ai_lab.agents.types import AgentRunner, AgentRunResult
from prokat24_ai_lab.evals.agent.types import AgentOutcomeEvalResult
from prokat24_ai_lab.observability.context import current_eval_case_id


class AgentOutcomeEvalCase(TypedDict):
    id: str
    user_input: str
    expected_answer_fragments: list[str]
    forbidden_answer_fragments: list[str]
    required_tool_names: list[str]


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
