from openai import OpenAI

from prokat24_ai_lab.agents.agent_loop import run_agent
from prokat24_ai_lab.agents.openai_agent_step import create_openai_agent_step_generator
from prokat24_ai_lab.agents.openai_tool_schema import create_openai_tools
from prokat24_ai_lab.agents.tool_executor import create_tool_executor
from prokat24_ai_lab.agents.tool_registry import create_tool_registry
from prokat24_ai_lab.agents.types import AgentRunResult
from prokat24_ai_lab.evals.safety.agent import (
    AgentSafetyEvalCase,
    load_agent_safety_eval_cases,
    run_agent_safety_evals,
)
from prokat24_ai_lab.judges.openai_judge import create_openai_criterion_judge
from prokat24_ai_lab.knowledge.base import load_knowledge_base
from prokat24_ai_lab.products.catalog import load_products
from prokat24_ai_lab.project_paths import (
    AGENT_SAFETY_EVAL_CASES_FILE,
    AGENT_SAFETY_MALICIOUS_BOOKING_KNOWLEDGE_PAGES_FILE,
    AGENT_SAFETY_MALICIOUS_CONTACTS_KNOWLEDGE_PAGES_FILE,
    AGENT_SAFETY_MALICIOUS_KNOWLEDGE_PAGES_FILE,
    KNOWLEDGE_PAGES_FILE,
    PRODUCTS_FILE,
)

MAX_AGENT_STEPS = 5


def main() -> None:
    cases = load_agent_safety_eval_cases(AGENT_SAFETY_EVAL_CASES_FILE)
    products = load_products(PRODUCTS_FILE)
    pages = load_knowledge_base(KNOWLEDGE_PAGES_FILE)
    malicious_pages = load_knowledge_base(AGENT_SAFETY_MALICIOUS_KNOWLEDGE_PAGES_FILE)
    malicious_contacts_pages = load_knowledge_base(
        AGENT_SAFETY_MALICIOUS_CONTACTS_KNOWLEDGE_PAGES_FILE
    )
    malicious_booking_pages = load_knowledge_base(
        AGENT_SAFETY_MALICIOUS_BOOKING_KNOWLEDGE_PAGES_FILE
    )

    client = OpenAI()
    criterion_judge = create_openai_criterion_judge(client)

    registry = create_tool_registry(products, pages)
    malicious_registry = create_tool_registry(products, malicious_pages)
    malicious_contacts_registry = create_tool_registry(
        products,
        malicious_contacts_pages,
    )
    malicious_booking_registry = create_tool_registry(
        products,
        malicious_booking_pages,
    )

    openai_tools = create_openai_tools(registry)

    base_agent_step_generator = create_openai_agent_step_generator(client, openai_tools)
    base_tool_executor = create_tool_executor(registry)
    malicious_tool_executor = create_tool_executor(malicious_registry)
    malicious_contacts_tool_executor = create_tool_executor(malicious_contacts_registry)
    malicious_booking_tool_executor = create_tool_executor(malicious_booking_registry)

    def agent_runner(case: AgentSafetyEvalCase) -> AgentRunResult:
        user_input = case["user_input"]
        fixture = case.get("knowledge_fixture")

        if fixture == "malicious_delivery":
            tool_executor = malicious_tool_executor

        elif fixture == "malicious_contacts":
            tool_executor = malicious_contacts_tool_executor

        elif fixture == "malicious_booking":
            tool_executor = malicious_booking_tool_executor

        else:
            tool_executor = base_tool_executor

        agent_result = run_agent(
            user_input,
            base_agent_step_generator,
            tool_executor,
            max_steps=MAX_AGENT_STEPS,
        )

        return agent_result

    results = run_agent_safety_evals(
        cases=cases,
        agent_runner=agent_runner,
        judge=criterion_judge,
    )

    for result in results:
        print("=" * 80)
        print(f"ID: {result['id']}")
        print(f"Category: {result['category']}")
        print()

        deterministic = result["deterministic"]
        behavior = result["behavior"]

        print("DETERMINISTIC")
        print(f"Passed: {deterministic['passed']}")
        print(f"Missing fragments: {deterministic['missing_fragments']}")
        print(f"Unexpected fragments: {deterministic['unexpected_fragments']}")
        print(f"Missing tool names: {deterministic['missing_tool_names']}")
        print(f"Forbidden tool names used: {deterministic['forbidden_tool_names_used']}")
        print()

        print("BEHAVIOR")
        print(f"Passed: {behavior['passed']}")
        print(f"Failed criteria: {behavior['failed_criteria']}")

        for criterion in behavior["criteria"]:
            print(f"- {criterion['name']}: {'PASS' if criterion['passed'] else 'FAIL'}")
            print(f"  Reason: {criterion['reason']}")

        print()

        print("FINAL")
        print(f"Passed: {result['passed']}")
        print()

        print("ANSWER")
        print(deterministic["final_answer"])
        print()

        print("TOOL CALLS")
        print(deterministic["tool_calls"])
        print()

        print("TOOL EXECUTIONS")
        print(deterministic["tool_executions"])

    passed_cases = sum(1 for result in results if result["passed"])

    print("=" * 80)
    print("SUMMARY")
    print(f"Cases: {len(results)}")
    print(f"Passed: {passed_cases}")
    print(f"Pass rate: {passed_cases / len(results):.2%}")


if __name__ == "__main__":
    main()
