from openai.types.responses import FunctionToolParam

from prokat24_ai_lab.agents.types import ToolDefinition, ToolRegistry


def create_openai_tool_schema(definition: ToolDefinition) -> FunctionToolParam:
    parameters: dict[str, object] = dict(definition["parameters"])

    return {
        "type": "function",
        "name": definition["name"],
        "description": definition["description"],
        "parameters": parameters,
        "strict": False,
    }


def create_openai_tools(registry: ToolRegistry) -> list[FunctionToolParam]:
    tools: list[FunctionToolParam] = []

    for registered_tool in registry.values():
        definition = registered_tool["definition"]
        tools.append(create_openai_tool_schema(definition))

    return tools
