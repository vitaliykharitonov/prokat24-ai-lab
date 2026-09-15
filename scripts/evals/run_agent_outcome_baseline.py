from openai import OpenAI

from prokat24_ai_lab.agents.agent_loop import run_agent
from prokat24_ai_lab.agents.openai_agent_step import create_openai_agent_step_generator
from prokat24_ai_lab.agents.openai_tool_schema import create_openai_tools
from prokat24_ai_lab.agents.tool_executor import create_tool_executor
from prokat24_ai_lab.agents.tool_registry import create_tool_registry
from prokat24_ai_lab.agents.types import AgentRunResult
from prokat24_ai_lab.evals.agent_evals import (
    load_agent_outcome_eval_cases,
    run_agent_outcome_evals,
    summarize_agent_outcome_eval_results,
)
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

    client = OpenAI()

    registry = create_tool_registry(products)
    pages = load_knowledge_base(KNOWLEDGE_PAGES_FILE)

    openai_tools = create_openai_tools(registry, pages)

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

    for result in results:
        print(f"Question: {result['user_input']}")
        print(f"Passed: {result['passed']}")
        print(f"Missing fragments: {result['missing_fragments']}")
        print(f"Unexpected fragments: {result['unexpected_fragments']}")
        print(f"Missing tool names: {result['missing_tool_names']}")
        print(f"Answer:\n{result['final_answer']}\n")
        print(f"Tool calls:\n{result['tool_calls']}\n")

    print("AGENT OUTCOME SUMMARY\n")
    print(f"Cases: {summary['total_cases']}")
    print(f"Passed: {summary['passed_cases']}")
    print(f"Pass rate: {summary['pass_rate']:.2%}")


if __name__ == "__main__":
    main()
