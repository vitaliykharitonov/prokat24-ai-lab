from prokat24_ai_lab.evals.tool_use import (
    ToolCall,
    ToolSequenceGenerator,
    load_tool_sequence_eval_cases,
    run_tool_sequence_evals,
    summarize_tool_sequence_eval_results,
)
from prokat24_ai_lab.project_paths import TOOL_SEQUENCE_EVAL_CASES_FILE


def main() -> None:
    cases = load_tool_sequence_eval_cases(TOOL_SEQUENCE_EVAL_CASES_FILE)

    def fake_generate_tool_sequence(_user_input: str) -> list[ToolCall]:
        return [
            ToolCall(
                tool_name="search_products",
                arguments={"query": "2-місний намет"},
            ),
            ToolCall(
                tool_name="get_product_price",
                arguments={
                    "product_url": "https://prokat24.com.ua/ua/turisticheskoe-snaryazhenie/prokat-palatok/prokat-palatki-2-mestnaya/",
                    "days": 3,
                },
            ),
        ]

    generate_tool_sequence: ToolSequenceGenerator = fake_generate_tool_sequence

    results = run_tool_sequence_evals(cases, generate_tool_sequence)

    summary = summarize_tool_sequence_eval_results(results)

    for result in results:
        print(result["user_input"])
        print(result["expected_tool_calls"])
        print(result["actual_tool_calls"])
        print(result["passed"])
        print()

    print("TOOL SEQUENCE SUMMARY")
    print()
    print(f"Cases: {summary['total_cases']}")
    print(f"Passed: {summary['passed_cases']}")
    print(f"Pass rate: {summary['pass_rate']:.2%}")
    print(f"Tool count match rate: {summary['tool_count_match_rate']:.2%}")
    print(f"Tool name match rate: {summary['tool_name_match_rate']:.2%}")
    print(f"Arguments match rate: {summary['arguments_match_rate']:.2%}")


if __name__ == "__main__":
    main()
