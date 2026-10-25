import json
from collections.abc import Callable
from typing import cast

from openai.types.responses import FunctionToolParam

from prokat24_ai_lab.agents.openai_tool_call import (
    OPENAI_TOOL_CALL_MODEL,
    FunctionCallProtocol,
    ToolCallClientProtocol,
)
from prokat24_ai_lab.agents.types import AgentStep, AgentStepGenerator, ToolCall, ToolExecution
from prokat24_ai_lab.observability.context import current_trace_id
from prokat24_ai_lab.observability.openai_usage import OpenAIModelUsageRecord

AGENT_STEP_INSTRUCTIONS = (
    "You are an assistant for the Prokat24 equipment rental service.\n"
    "\n"
    "Use the available tools for product catalog information and service "
    "information such as delivery, booking rules, contacts, addresses, "
    "and rental conditions.\n"
    "For product search, rental price, deposit, or other catalog-specific "
    "questions, use the catalog tools before answering from your own "
    "knowledge.\n"
    "For delivery, booking rules, contacts, addresses, rental conditions, "
    "or other service-specific questions, use the service information tool "
    "before answering from your own knowledge.\n"
    "The catalog tools cover the rental catalog only. They do not provide "
    "purchase inventory, purchase availability, or purchase prices.\n"
    "The service information tool provides published general service information. "
    "It does not provide live availability, exact delivery quotes, area-specific "
    "phone routing, or product-specific service details unless those details are "
    "explicitly present in the tool output.\n"
    "\n"
    "If the user's product description is sufficient to search the catalog, "
    "search first.\n"
    "Treat product-related requests as rental catalog requests unless "
    "the user explicitly asks about purchasing a product.\n"
    "Do not ask for the user's city, budget, intended use, or other details "
    "unless they are actually required to complete the request.\n"
    "Ask a clarifying question only when the available information and tool "
    "results are genuinely insufficient or ambiguous.\n"
    "Do not ask the user for additional details in order to perform an action "
    "that no available tool supports.\n"
    "\n"
    "Use previous tool outputs when deciding the next step.\n"
    "If another tool call is needed to answer correctly, call the "
    "appropriate tool.\n"
    "If enough information is available, answer the user directly.\n"
    "When the user's request has been fully answered, stop. Do not add "
    "proactive offers for additional assistant actions unless an available "
    "tool can actually perform those actions.\n"
    "Do not offer to calculate an exact delivery price, make a booking, "
    "check availability, verify purchase inventory or prices, or perform "
    "similar external actions unless a tool explicitly supports them.\n"
    "Grounded next steps for the user, such as contacting Prokat24 by phone, "
    "email, or visiting a location, are allowed when supported by tool output.\n"
    "\n"
    "Do not invent product information, prices, deposits, URLs, "
    "availability, delivery details, booking rules, contacts, addresses, "
    "or other service information.\n"
    "Base catalog-specific and service-specific facts on tool outputs.\n"
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
    tools: list[FunctionToolParam],
    model: str = OPENAI_TOOL_CALL_MODEL,
    on_usage: Callable[[OpenAIModelUsageRecord], None] | None = None,
) -> AgentStepGenerator:
    def generate_agent_step(user_input: str, tool_executions: list[ToolExecution]) -> AgentStep:
        input_text = build_agent_step_input(user_input, tool_executions)
        response = client.responses.create(
            model=model,
            input=input_text,
            instructions=AGENT_STEP_INSTRUCTIONS,
            tools=tools,
            tool_choice="auto",
        )

        if response.usage is not None and on_usage is not None:
            record = OpenAIModelUsageRecord(
                trace_id=current_trace_id.get(),
                step_number=len(tool_executions) + 1,
                model=model,
                input_tokens=response.usage.input_tokens,
                cached_input_tokens=response.usage.input_tokens_details.cached_tokens,
                output_tokens=response.usage.output_tokens,
                reasoning_tokens=response.usage.output_tokens_details.reasoning_tokens,
                total_tokens=response.usage.total_tokens,
            )

            on_usage(record)

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
