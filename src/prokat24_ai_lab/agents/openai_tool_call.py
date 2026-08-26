import json

from openai import OpenAI

from prokat24_ai_lab.agents.tools import TOOLS
from prokat24_ai_lab.evals.tool_use import ToolCall, ToolCallGenerator

OPENAI_TOOL_CALL_MODEL = "gpt-5-mini"


def create_openai_tool_call_generator(
    client: OpenAI,
    model: str = OPENAI_TOOL_CALL_MODEL,
) -> ToolCallGenerator:
    def generate_tool_call(user_input: str) -> ToolCall | None:
        response = client.responses.create(model=model, input=user_input, tools=TOOLS)

        for output_item in response.output:
            if output_item.type == "function_call":
                tool_name = output_item.name
                arguments = json.loads(output_item.arguments)

                return ToolCall(
                    tool_name=tool_name,
                    arguments=arguments,
                )

        return None

    return generate_tool_call
