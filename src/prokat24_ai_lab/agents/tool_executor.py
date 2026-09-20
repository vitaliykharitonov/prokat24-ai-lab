from prokat24_ai_lab.agents.types import ToolCall, ToolExecutor, ToolRegistry


def create_tool_executor(registry: ToolRegistry) -> ToolExecutor:
    def execute_tool_call(tool_call: ToolCall) -> str:
        tool_name = tool_call["tool_name"]
        arguments = tool_call["arguments"]

        registered_tool = registry.get(tool_name)

        if registered_tool is None:
            raise ValueError(f"Unsupported tool: {tool_name}")

        handler = registered_tool["handler"]

        return handler(arguments)

    return execute_tool_call
