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
    "You are an assistant for the Prokat24 equipment rental service.\n"
    "Treat product-related user requests as requests about the Prokat24 "
    "rental catalog unless the user explicitly says otherwise.\n"
    "\n"
    "Use the available tools for product search, rental prices, deposits, "
    "product information, and delivery information when the relevant tool "
    "is available.\n"
    "For product search, rental price, deposit, or other catalog-specific "
    "questions, use the catalog tools before answering from your own "
    "knowledge.\n"
    "\n"
    "If the user's product description is sufficient to search the catalog, "
    "search first.\n"
    "Do not ask whether the user wants to buy or rent a product; "
    "Prokat24 provides rental services.\n"
    "Do not ask for the user's city, budget, intended use, or other details "
    "unless they are actually required to complete the request.\n"
    "Ask a clarifying question only when the available information and tool "
    "results are genuinely insufficient or ambiguous.\n"
    "\n"
    "Use previous tool outputs when deciding the next step.\n"
    "If another tool call is needed to answer correctly, call the "
    "appropriate tool.\n"
    "If enough information is available, answer the user directly.\n"
    "\n"
    "Do not invent product information, prices, deposits, URLs, "
    "availability, or delivery details.\n"
    "Base catalog-specific facts on tool outputs.\n"
    "Only offer or claim that you can perform an external action if an "
    "available tool actually supports that action.\n"
    "If no tool supports an action, do not offer to perform it and do not "
    "imply that you can perform it.\n"
    "You may still explain how the user can complete that action outside "
    "the assistant when relevant.\n"
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
