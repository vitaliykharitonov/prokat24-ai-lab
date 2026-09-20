from openai import OpenAI

from prokat24_ai_lab.agents.agent_loop import run_agent
from prokat24_ai_lab.agents.openai_agent_step import create_openai_agent_step_generator
from prokat24_ai_lab.agents.openai_tool_schema import create_openai_tools
from prokat24_ai_lab.agents.tool_executor import create_tool_executor
from prokat24_ai_lab.agents.tool_registry import create_tool_registry
from prokat24_ai_lab.agents.types import AgentRunResult
from prokat24_ai_lab.evals.agent_evals import (
    evaluate_agent_capability,
    evaluate_agent_grounding,
    load_agent_outcome_eval_cases,
    run_agent_outcome_evals,
    summarize_agent_outcome_eval_results,
)
from prokat24_ai_lab.judges.openai_judge import create_openai_criterion_judge
from prokat24_ai_lab.knowledge.base import load_knowledge_base
from prokat24_ai_lab.products.catalog import load_products
from prokat24_ai_lab.project_paths import (
    AGENT_OUTCOME_EVAL_CASES_FILE,
    KNOWLEDGE_PAGES_FILE,
    PRODUCTS_FILE,
)


def main() -> None:
    cases = load_agent_outcome_eval_cases(AGENT_OUTCOME_EVAL_CASES_FILE)
    products = load_products(PRODUCTS_FILE)
    pages = load_knowledge_base(KNOWLEDGE_PAGES_FILE)

    client = OpenAI()

    criterion_judge = create_openai_criterion_judge(client)

    registry = create_tool_registry(products, pages)
    openai_tools = create_openai_tools(registry)

    generate_agent_step = create_openai_agent_step_generator(
        client,
        openai_tools,
    )
    execute_tool_call = create_tool_executor(registry)

    def agent_runner(user_input: str) -> AgentRunResult:
        return run_agent(
            user_input,
            generate_agent_step,
            execute_tool_call,
        )

    results = run_agent_outcome_evals(cases, agent_runner)
    summary = summarize_agent_outcome_eval_results(results)

    capability_results = []

    for result in results:
        capability_result = evaluate_agent_capability(result, criterion_judge)
        capability_results.append(capability_result)

    capability_passed = sum(result["passed"] for result in capability_results)
    capability_total = len(capability_results)

    capability_pass_rate = capability_passed / capability_total if capability_total else 0.0

    grounding_results = []

    for result in results:
        grounding_result = evaluate_agent_grounding(result, criterion_judge)
        grounding_results.append(grounding_result)

    grounding_passed = sum(result["passed"] for result in grounding_results)
    grounding_total = len(grounding_results)

    grounding_pass_rate = grounding_passed / grounding_total if grounding_total else 0.0

    for result, capability_result, grounding_result in zip(
        results,
        capability_results,
        grounding_results,
        strict=True,
    ):
        print("=" * 80)
        print(f"Question: {result['user_input']}")
        print()

        print("OUTCOME")
        print(f"Passed: {result['passed']}")
        print(f"Missing fragments: {result['missing_fragments']}")
        print(f"Unexpected fragments: {result['unexpected_fragments']}")
        print(f"Missing tool names: {result['missing_tool_names']}")
        print()

        print("CAPABILITY")
        print(f"Passed: {capability_result['passed']}")
        print(f"Failed criteria: {capability_result['failed_criteria']}")

        for criterion in capability_result["criteria"]:
            print(f"- {criterion['name']}: {'PASS' if criterion['passed'] else 'FAIL'}")
            print(f"  Reason: {criterion['reason']}")

        print()

        print("GROUNDING")
        print(f"Passed: {grounding_result['passed']}")
        print(f"Failed criteria: {grounding_result['failed_criteria']}")

        for criterion in grounding_result["criteria"]:
            print(f"- {criterion['name']}: {'PASS' if criterion['passed'] else 'FAIL'}")
            print(f"  Reason: {criterion['reason']}")

        print()

        print("ANSWER")
        print(result["final_answer"])
        print()

        print("TOOL CALLS")
        print(result["tool_calls"])
        print()

        print("TOOL EXECUTIONS")
        print(result["tool_executions"])
        print()

    print("=" * 80)
    print("SUMMARY")
    print()

    print("AGENT OUTCOME")
    print(f"Cases: {summary['total_cases']}")
    print(f"Passed: {summary['passed_cases']}")
    print(f"Pass rate: {summary['pass_rate']:.2%}")
    print()

    print("AGENT CAPABILITY")
    print(f"Cases: {capability_total}")
    print(f"Passed: {capability_passed}")
    print(f"Pass rate: {capability_pass_rate:.2%}")
    print()

    print("AGENT GROUNDING")
    print(f"Cases: {grounding_total}")
    print(f"Passed: {grounding_passed}")
    print(f"Pass rate: {grounding_pass_rate:.2%}")


if __name__ == "__main__":
    main()
