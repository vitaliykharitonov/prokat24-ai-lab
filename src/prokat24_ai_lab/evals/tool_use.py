import json
from collections.abc import Callable
from pathlib import Path
from typing import TypedDict


class ToolCall(TypedDict):
    tool_name: str
    arguments: dict[str, str | int | bool]


class ToolUseEvalCase(TypedDict):
    user_input: str
    expected_tool_call: ToolCall


class ToolUseEvalResult(TypedDict):
    user_input: str
    expected_tool_call: ToolCall
    actual_tool_call: ToolCall | None
    tool_name_matches: bool
    arguments_match: bool
    passed: bool


ToolCallGenerator = Callable[[str], ToolCall | None]


def load_tool_use_eval_cases(input_file: Path) -> list[ToolUseEvalCase]:
    with open(input_file, encoding="utf-8") as file:
        cases: list[ToolUseEvalCase] = json.load(file)

    return cases


def evaluate_tool_use_case(
    case: ToolUseEvalCase,
    actual_tool_call: ToolCall | None,
) -> ToolUseEvalResult:
    if actual_tool_call is None:
        tool_name_matches = False
        arguments_match = False

    else:
        tool_name_matches = actual_tool_call["tool_name"] == case["expected_tool_call"]["tool_name"]
        arguments_match = actual_tool_call["arguments"] == case["expected_tool_call"]["arguments"]

    passed = tool_name_matches and arguments_match

    return ToolUseEvalResult(
        user_input=case["user_input"],
        expected_tool_call=case["expected_tool_call"],
        actual_tool_call=actual_tool_call,
        tool_name_matches=tool_name_matches,
        arguments_match=arguments_match,
        passed=passed,
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
