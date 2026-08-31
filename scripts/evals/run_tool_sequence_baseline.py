from prokat24_ai_lab.evals.tool_use import (
    ToolCall,
    ToolSequenceGenerator,
    load_tool_sequence_eval_cases,
    run_tool_sequence_evals,
)
from prokat24_ai_lab.project_paths import TOOL_SEQUENCE_EVAL_CASES_FILE


def main() -> None:
    cases = load_tool_sequence_eval_cases(TOOL_SEQUENCE_EVAL_CASES_FILE)

    def fake_generate_tool_sequence(_user_input: str) -> list[ToolCall]:
        return [
            ToolCall(
                tool_name="search_products",
                arguments={"query": "намет"},
            ),
            ToolCall(
                tool_name="get_product_price",
                arguments={"product": "намет", "days": 3},
            ),
        ]

    generate_tool_sequence: ToolSequenceGenerator = fake_generate_tool_sequence

    results = run_tool_sequence_evals(cases, generate_tool_sequence)

    for result in results:
        print(result["user_input"])
        print(result["expected_tool_calls"])
        print(result["actual_tool_calls"])
        print(result["passed"])
        print()


if __name__ == "__main__":
    main()
