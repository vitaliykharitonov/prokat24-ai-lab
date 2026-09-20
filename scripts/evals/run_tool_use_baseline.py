import argparse
from argparse import Namespace

from openai import OpenAI

from prokat24_ai_lab.agents.openai_tool_call import create_openai_tool_call_generator
from prokat24_ai_lab.agents.openai_tool_schema import create_openai_tools
from prokat24_ai_lab.agents.tool_registry import create_tool_registry
from prokat24_ai_lab.agents.types import ToolCall, ToolCallGenerator
from prokat24_ai_lab.evals.tool_use import (
    RepeatedToolUseEvalResult,
    ToolUseEvalResult,
    load_tool_use_eval_cases,
    run_repeated_tool_use_evals,
    summarize_tool_use_eval_results,
)
from prokat24_ai_lab.knowledge.base import load_knowledge_base
from prokat24_ai_lab.products.catalog import load_products
from prokat24_ai_lab.project_paths import (
    KNOWLEDGE_PAGES_FILE,
    PRODUCTS_FILE,
    TOOL_USE_EVAL_CASES_FILE,
)


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser(description="Use real or fake tool-call")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Use the real OpenAI tool-call generator",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of evaluation runs",
    )

    return parser.parse_args()


def format_run_result(result: ToolUseEvalResult) -> str:
    if result["passed"]:
        return "PASS"

    if not result["tool_call_decision_match"]:
        if result["actual_tool_call"] is None:
            return "FAIL: no call"

        return "FAIL: unexpected tool"

    if not result["tool_name_match"]:
        return "FAIL: wrong tool"

    return "FAIL: arguments"


def print_repeated_results_table(
    results: list[RepeatedToolUseEvalResult],
) -> None:
    print("\nTOOL USE EVAL RESULTS")
    print()

    if not results:
        return
    runs = len(results[0]["results"])

    header = ["Case", "Expected tool"]

    for run_number in range(1, runs + 1):
        header.append(f"Run {run_number}")

    header.append("Passed")

    rows: list[list[str]] = []

    for case_number, repeated_result in enumerate(results, start=1):
        row = [str(case_number)]
        expected_tool_call = repeated_result["expected_tool_call"]

        if expected_tool_call is None:
            expected_tool_name = "None"
        else:
            expected_tool_name = expected_tool_call["tool_name"]

        row.append(expected_tool_name)

        for run_result in repeated_result["results"]:
            row.append(format_run_result(run_result))

        passed_runs = sum(run_result["passed"] for run_result in repeated_result["results"])

        row.append(f"{passed_runs}/{runs}")

        rows.append(row)

    column_widths: list[int] = []
    all_rows = [header, *rows]

    for column_index in range(len(header)):
        width = max(len(row[column_index]) for row in all_rows)
        column_widths.append(width)

    formatted_header = [
        f"{value:<{width}}" for value, width in zip(header, column_widths, strict=False)
    ]

    print(" | ".join(formatted_header))

    separator = "-+-".join("-" * width for width in column_widths)
    print(separator)

    for row in rows:
        formatted_cells = [
            f"{value:<{width}}" for value, width in zip(row, column_widths, strict=False)
        ]

        print(" | ".join(formatted_cells))


def main() -> None:
    args = parse_args()

    if args.runs < 1:
        raise ValueError("--runs must be at least 1")

    cases = load_tool_use_eval_cases(TOOL_USE_EVAL_CASES_FILE)

    def fake_generate_tool_call(_user_input: str) -> ToolCall | None:
        return {
            "tool_name": "search_products",
            "arguments": {"query": "намет"},
        }

    generate_tool_call: ToolCallGenerator = fake_generate_tool_call

    if args.live:
        products = load_products(PRODUCTS_FILE)
        pages = load_knowledge_base(KNOWLEDGE_PAGES_FILE)

        registry = create_tool_registry(products, pages)
        openai_tools = create_openai_tools(registry)

        client = OpenAI()
        generate_tool_call = create_openai_tool_call_generator(
            client,
            openai_tools,
        )

    results = run_repeated_tool_use_evals(cases, generate_tool_call, args.runs)

    all_run_results = [
        run_result for repeated_result in results for run_result in repeated_result["results"]
    ]

    summary = summarize_tool_use_eval_results(all_run_results)

    print_repeated_results_table(results)

    for repeated_result in results:
        for run_result in repeated_result["results"]:
            if run_result["tool_name_match"] and not run_result["arguments_match"]:
                print()
                print(f"Expected: {run_result['expected_tool_call']}")
                print(f"Actual:   {run_result['actual_tool_call']}")

    print()
    print("TOOL USE SUMMARY")
    print()

    print(f"Cases: {len(results)}")
    print(f"Runs per case: {args.runs}")
    print(f"Total evaluations: {len(all_run_results)}")

    print(
        f"Passed: {summary['passed_cases']}/{summary['total_cases']} ({summary['pass_rate']:.2%})"
    )
    print(
        f"Tool-call decision: "
        f"{summary['tool_call_decision_matches']}/{summary['total_cases']} "
        f"({summary['tool_call_decision_match_rate']:.2%})"
    )

    print(
        f"Tool name matches: "
        f"{summary['tool_name_matches']}/{summary['tool_call_cases']} "
        f"({summary['tool_name_match_rate']:.2%})"
    )

    print(
        f"Arguments matches: "
        f"{summary['arguments_matches']}/{summary['tool_call_cases']} "
        f"({summary['arguments_match_rate']:.2%})"
    )


if __name__ == "__main__":
    main()
