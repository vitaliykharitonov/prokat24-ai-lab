import json
from typing import cast

from prokat24_ai_lab.agents.openai_tool_call import (
    OPENAI_TOOL_CALL_MODEL,
    FunctionCallProtocol,
    ToolCallClientProtocol,
)
from prokat24_ai_lab.agents.tool_schemas import TOOLS
from prokat24_ai_lab.agents.types import AgentStep, AgentStepGenerator, ToolCall, ToolExecution

AGENT_STEP_INSTRUCTIONS = (
    "Use the available tools for delivery information, product search, "
    "and rental price questions.\n"
    "Use previous tool outputs when deciding the next step.\n"
    "If more information is needed, call the appropriate tool.\n"
    "If enough information is available, answer the user.\n"
    "Do not invent product information, prices, URLs, or delivery details.\n"
)


def build_agent_step_input(
    user_input: str,
    tool_executions: list[ToolExecution],
) -> str:
    parts = [f"User request:\n{user_input}"]

    if not tool_executions:
        return "\n\n".join(parts)

    parts.append("Executed tools:")

    for index, tool_execution in enumerate(tool_executions, start=1):
        parts.append(f"{index}. {tool_execution['tool_call']['tool_name']}")
        parts.append(
            f"Arguments: {json.dumps(tool_execution['tool_call']['arguments'], ensure_ascii=False)}"
        )
        parts.append(f"Output:\n{tool_execution['output']}")

    return "\n".join(parts)


def create_openai_agent_step_generator(
    client: ToolCallClientProtocol,
    model: str = OPENAI_TOOL_CALL_MODEL,
) -> AgentStepGenerator:
    def generate_agent_step(user_input: str, tool_executions: list[ToolExecution]) -> AgentStep:
        input_text = build_agent_step_input(user_input, tool_executions)
        response = client.responses.create(
            model=model,
            input=input_text,
            instructions=AGENT_STEP_INSTRUCTIONS,
            tools=TOOLS,
            tool_choice="auto",
        )

        for output_item in response.output:
            if output_item.type == "function_call":
                function_call = cast(FunctionCallProtocol, output_item)
                tool_name = function_call.name
                arguments = json.loads(function_call.arguments)

                return AgentStep(
                    final_answer=None,
                    tool_call=ToolCall(
                        tool_name=tool_name,
                        arguments=arguments,
                    ),
                )

        return AgentStep(
            final_answer=response.output_text,
            tool_call=None,
        )

    return generate_agent_step
