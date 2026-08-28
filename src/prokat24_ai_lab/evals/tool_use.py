import json
from collections.abc import Callable
from pathlib import Path
from typing import TypedDict


class ToolCall(TypedDict):
    tool_name: str
    arguments: dict[str, str | int | bool]


class ToolUseEvalCase(TypedDict):
    user_input: str
    expected_tool_call: ToolCall | None


class ToolUseEvalResult(TypedDict):
    user_input: str
    expected_tool_call: ToolCall | None
    actual_tool_call: ToolCall | None
    tool_call_decision_match: bool
    tool_name_match: bool
    arguments_match: bool
    passed: bool


class ToolUseEvalSummary(TypedDict):
    total_cases: int
    tool_call_cases: int
    passed_cases: int
    pass_rate: float
    tool_name_matches: int
    tool_name_match_rate: float
    arguments_matches: int
    arguments_match_rate: float
    tool_call_decision_matches: int
    tool_call_decision_match_rate: float


ToolCallGenerator = Callable[[str], ToolCall | None]


def load_tool_use_eval_cases(input_file: Path) -> list[ToolUseEvalCase]:
    with open(input_file, encoding="utf-8") as file:
        cases: list[ToolUseEvalCase] = json.load(file)

    return cases


def evaluate_tool_use_case(
    case: ToolUseEvalCase,
    actual_tool_call: ToolCall | None,
) -> ToolUseEvalResult:
    expected_tool_call = case["expected_tool_call"]

    if expected_tool_call is None:
        tool_name_matches = actual_tool_call is None
        arguments_match = actual_tool_call is None

    elif actual_tool_call is None:
        tool_name_matches = False
        arguments_match = False

    else:
        tool_name_matches = actual_tool_call["tool_name"] == expected_tool_call["tool_name"]
        arguments_match = actual_tool_call["arguments"] == expected_tool_call["arguments"]

    passed = tool_name_matches and arguments_match
    tool_call_decision_match = (expected_tool_call is None) == (actual_tool_call is None)

    return ToolUseEvalResult(
        user_input=case["user_input"],
        expected_tool_call=case["expected_tool_call"],
        actual_tool_call=actual_tool_call,
        tool_name_match=tool_name_matches,
        arguments_match=arguments_match,
        passed=passed,
        tool_call_decision_match=tool_call_decision_match,
    )


def run_tool_use_evals(
    cases: list[ToolUseEvalCase],
    generate_tool_call: ToolCallGenerator,
) -> list[ToolUseEvalResult]:
    results: list[ToolUseEvalResult] = []

    for case in cases:
        actual_tool_call = generate_tool_call(case["user_input"])

        results.append(evaluate_tool_use_case(case, actual_tool_call))

    return results


def summarize_tool_use_eval_results(
    results: list[ToolUseEvalResult],
) -> ToolUseEvalSummary:
    if not results:
        return ToolUseEvalSummary(
            total_cases=0,
            tool_call_cases=0,
            passed_cases=0,
            pass_rate=0.0,
            tool_name_matches=0,
            tool_name_match_rate=0.0,
            arguments_matches=0,
            arguments_match_rate=0.0,
            tool_call_decision_matches=0,
            tool_call_decision_match_rate=0.0,
        )

    total_cases = len(results)
    tool_call_cases = 0
    passed_cases = 0
    tool_name_matches = 0
    arguments_matches = 0
    tool_name_match_rate = 0.0
    arguments_match_rate = 0.0
    tool_call_decision_matches = 0

    for result in results:
        if result["expected_tool_call"] is not None:
            tool_call_cases += 1

            if result["tool_name_match"]:
                tool_name_matches += 1

            if result["arguments_match"]:
                arguments_matches += 1

        if result["passed"]:
            passed_cases += 1

        if result["tool_call_decision_match"]:
            tool_call_decision_matches += 1

    pass_rate = passed_cases / total_cases

    if tool_call_cases:
        tool_name_match_rate = tool_name_matches / tool_call_cases
        arguments_match_rate = arguments_matches / tool_call_cases

    tool_call_decision_match_rate = tool_call_decision_matches / total_cases

    return ToolUseEvalSummary(
        total_cases=total_cases,
        tool_call_cases=tool_call_cases,
        passed_cases=passed_cases,
        pass_rate=pass_rate,
        tool_name_matches=tool_name_matches,
        tool_name_match_rate=tool_name_match_rate,
        arguments_matches=arguments_matches,
        arguments_match_rate=arguments_match_rate,
        tool_call_decision_matches=tool_call_decision_matches,
        tool_call_decision_match_rate=tool_call_decision_match_rate,
    )
