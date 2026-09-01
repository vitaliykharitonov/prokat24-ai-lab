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


class ToolSequenceEvalCase(TypedDict):
    user_input: str
    expected_tool_calls: list[ToolCall]


class ToolUseEvalResult(TypedDict):
    user_input: str
    expected_tool_call: ToolCall | None
    actual_tool_call: ToolCall | None
    tool_call_decision_match: bool
    tool_name_match: bool
    arguments_match: bool
    passed: bool


class ToolSequenceEvalResult(TypedDict):
    user_input: str
    expected_tool_calls: list[ToolCall]
    actual_tool_calls: list[ToolCall]
    tool_count_match: bool
    tool_names_match: bool
    arguments_match: bool
    passed: bool


class RepeatedToolUseEvalResult(TypedDict):
    user_input: str
    expected_tool_call: ToolCall | None
    results: list[ToolUseEvalResult]


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


class ToolSequenceEvalSummary(TypedDict):
    total_cases: int
    passed_cases: int
    pass_rate: float
    tool_count_matches: int
    tool_count_match_rate: float
    tool_name_matches: int
    tool_name_match_rate: float
    arguments_matches: int
    arguments_match_rate: float


ToolCallGenerator = Callable[[str], ToolCall | None]

ToolSequenceGenerator = Callable[[str], list[ToolCall]]


def load_tool_use_eval_cases(input_file: Path) -> list[ToolUseEvalCase]:
    with open(input_file, encoding="utf-8") as file:
        cases: list[ToolUseEvalCase] = json.load(file)

    return cases


def load_tool_sequence_eval_cases(input_file: Path) -> list[ToolSequenceEvalCase]:
    with open(input_file, encoding="utf-8") as file:
        cases: list[ToolSequenceEvalCase] = json.load(file)

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
        arguments_match = (
            tool_name_matches and actual_tool_call["arguments"] == expected_tool_call["arguments"]
        )

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


def evaluate_tool_sequence_case(
    case: ToolSequenceEvalCase,
    actual_tool_calls: list[ToolCall],
) -> ToolSequenceEvalResult:
    expected_tool_calls = case["expected_tool_calls"]
    tool_count_match = len(expected_tool_calls) == len(actual_tool_calls)

    tool_names_match = tool_count_match and all(
        expected_tool["tool_name"] == actual_tool["tool_name"]
        for expected_tool, actual_tool in zip(expected_tool_calls, actual_tool_calls, strict=False)
    )

    arguments_match = (
        tool_count_match
        and tool_names_match
        and all(
            expected_tool["arguments"] == actual_tool["arguments"]
            for expected_tool, actual_tool in zip(
                expected_tool_calls, actual_tool_calls, strict=False
            )
        )
    )

    passed = expected_tool_calls == actual_tool_calls

    return ToolSequenceEvalResult(
        user_input=case["user_input"],
        expected_tool_calls=expected_tool_calls,
        actual_tool_calls=actual_tool_calls,
        tool_count_match=tool_count_match,
        tool_names_match=tool_names_match,
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


def run_tool_sequence_evals(
    cases: list[ToolSequenceEvalCase],
    generate_tool_sequence: ToolSequenceGenerator,
) -> list[ToolSequenceEvalResult]:
    results: list[ToolSequenceEvalResult] = []

    for case in cases:
        actual_tool_calls = generate_tool_sequence(case["user_input"])

        results.append(evaluate_tool_sequence_case(case, actual_tool_calls))

    return results


def run_repeated_tool_use_evals(
    cases: list[ToolUseEvalCase],
    generate_tool_call: ToolCallGenerator,
    runs: int,
) -> list[RepeatedToolUseEvalResult]:
    results: list[RepeatedToolUseEvalResult] = []

    for case in cases:
        case_results: list[ToolUseEvalResult] = []

        for _ in range(runs):
            actual_tool_call = generate_tool_call(case["user_input"])
            case_results.append(evaluate_tool_use_case(case, actual_tool_call))

        results.append(
            RepeatedToolUseEvalResult(
                user_input=case["user_input"],
                expected_tool_call=case["expected_tool_call"],
                results=case_results,
            )
        )

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


def summarize_tool_sequence_eval_results(
    results: list[ToolSequenceEvalResult],
) -> ToolSequenceEvalSummary:
    if not results:
        return ToolSequenceEvalSummary(
            total_cases=0,
            passed_cases=0,
            pass_rate=0.0,
            tool_count_matches=0,
            tool_count_match_rate=0.0,
            tool_name_matches=0,
            tool_name_match_rate=0.0,
            arguments_matches=0,
            arguments_match_rate=0.0,
        )

    total_cases = len(results)
    passed_cases = 0
    tool_count_matches = 0
    tool_name_matches = 0
    arguments_matches = 0

    for result in results:
        if result["passed"]:
            passed_cases += 1

        if result["tool_count_match"]:
            tool_count_matches += 1

        if result["tool_names_match"]:
            tool_name_matches += 1

        if result["arguments_match"]:
            arguments_matches += 1

    pass_rate = passed_cases / total_cases
    tool_count_match_rate = tool_count_matches / total_cases
    tool_name_match_rate = tool_name_matches / total_cases
    arguments_match_rate = arguments_matches / total_cases

    return ToolSequenceEvalSummary(
        total_cases=total_cases,
        passed_cases=passed_cases,
        pass_rate=pass_rate,
        tool_count_matches=tool_count_matches,
        tool_count_match_rate=tool_count_match_rate,
        tool_name_matches=tool_name_matches,
        tool_name_match_rate=tool_name_match_rate,
        arguments_matches=arguments_matches,
        arguments_match_rate=arguments_match_rate,
    )
