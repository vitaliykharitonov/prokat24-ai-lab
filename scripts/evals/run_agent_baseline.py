from openai import OpenAI

from prokat24_ai_lab.agents.agent_loop import run_agent
from prokat24_ai_lab.agents.openai_agent_step import create_openai_agent_step_generator
from prokat24_ai_lab.agents.openai_tool_schema import create_openai_tools
from prokat24_ai_lab.agents.tool_executor import create_tool_executor
from prokat24_ai_lab.agents.tool_registry import create_tool_registry
from prokat24_ai_lab.knowledge.base import load_knowledge_base
from prokat24_ai_lab.products.catalog import load_products
from prokat24_ai_lab.project_paths import KNOWLEDGE_PAGES_FILE, PRODUCTS_FILE


def main() -> None:
    user_input = "Скільки коштує доставка за межі Києва?"

    client = OpenAI()
    products = load_products(PRODUCTS_FILE)
    pages = load_knowledge_base(KNOWLEDGE_PAGES_FILE)

    registry = create_tool_registry(products, pages)
    openai_tools = create_openai_tools(registry)

    generate_agent_step = create_openai_agent_step_generator(
        client,
        openai_tools,
    )
    execute_tool_call = create_tool_executor(registry)

    result = run_agent(user_input, generate_agent_step, execute_tool_call)

    print(result["final_answer"])
    print(result["tool_calls"])


if __name__ == "__main__":
    main()
