from prokat24_ai_lab.evals.tool_use import (
    ToolCall,
    load_tool_use_eval_cases,
    run_tool_use_evals,
    summarize_tool_use_eval_results,
)
from prokat24_ai_lab.project_paths import TOOL_USE_EVAL_CASES_FILE


def main() -> None:
    cases = load_tool_use_eval_cases(TOOL_USE_EVAL_CASES_FILE)

    def fake_generate_tool_call(_user_input: str) -> ToolCall | None:
        return {
            "tool_name": "get_delivery_info",
            "arguments": {"location": "Київ"},
        }

    results = run_tool_use_evals(cases, fake_generate_tool_call)

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
