from openai import OpenAI

from prokat24_ai_lab.agents.agent_loop import run_agent
from prokat24_ai_lab.agents.openai_agent_step import create_openai_agent_step_generator
from prokat24_ai_lab.agents.tool_executor import create_tool_executor
from prokat24_ai_lab.products.catalog import load_products
from prokat24_ai_lab.project_paths import PRODUCTS_FILE


def main() -> None:
    user_input = "Скільки буде коштувати 2-місний намет на 3 дні?"

    client = OpenAI()
    products = load_products(PRODUCTS_FILE)
    generate_agent_step = create_openai_agent_step_generator(client)
    execute_tool_call = create_tool_executor(products)

    result = run_agent(user_input, generate_agent_step, execute_tool_call)

    print(result["final_answer"])
    print(result["tool_calls"])


if __name__ == "__main__":
    main()
