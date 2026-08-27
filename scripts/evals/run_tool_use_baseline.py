import argparse
from argparse import Namespace

from openai import OpenAI

from prokat24_ai_lab.agents.openai_tool_call import create_openai_tool_call_generator
from prokat24_ai_lab.evals.tool_use import (
    ToolCall,
    ToolCallGenerator,
    load_tool_use_eval_cases,
    run_tool_use_evals,
    summarize_tool_use_eval_results,
)
from prokat24_ai_lab.project_paths import TOOL_USE_EVAL_CASES_FILE


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser(description="Use real or fake tool-call")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Use the real OpenAI tool-call generator",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = load_tool_use_eval_cases(TOOL_USE_EVAL_CASES_FILE)

    def fake_generate_tool_call(_user_input: str) -> ToolCall | None:
        return {
            "tool_name": "get_delivery_info",
            "arguments": {"location": "Київ"},
        }

    generate_tool_call: ToolCallGenerator = fake_generate_tool_call

    if args.live:
        client = OpenAI()
        generate_tool_call = create_openai_tool_call_generator(client)

    results = run_tool_use_evals(cases, generate_tool_call)

    summary = summarize_tool_use_eval_results(results)

    for result in results:
        print(f"User input: {result['user_input']}")
        print(f"Expected tool: {result['expected_tool_call']}")
        print(f"Actual tool: {result['actual_tool_call']}")
        print(f"Tool name matches: {result['tool_name_match']}")
        print(f"Arguments matches: {result['arguments_match']}")
        print(f"Passed: {result['passed']}")
        print()

    print("TOOL USE SUMMARY\n")
    print(f"Cases: {summary['total_cases']}")
    print(f"Passed: {summary['passed_cases']}/{summary['total_cases']}")
    print(f"Pass rate: {summary['pass_rate']:.2%}")
    print(f"Tool name match rate: {summary['tool_name_match_rate']:.2%}")
    print(f"Arguments match rate: {summary['arguments_match_rate']:.2%}")


if __name__ == "__main__":
    main()
