from prokat24_ai_lab.agents.types import (
    AgentRunResult,
    AgentStepGenerator,
    ToolCall,
    ToolExecution,
    ToolExecutor,
)


def run_agent(
    user_input: str,
    generate_agent_step: AgentStepGenerator,
    execute_tool_call: ToolExecutor,
) -> AgentRunResult:
    tool_executions: list[ToolExecution] = []
    tool_calls: list[ToolCall] = []

    agent_step = generate_agent_step(user_input, tool_executions)

    if agent_step["final_answer"] is not None:
        return AgentRunResult(
            final_answer=agent_step["final_answer"],
            tool_calls=tool_calls,
        )

    while agent_step["final_answer"] is None:
        tool_call = agent_step["tool_call"]

        if tool_call is None:
            raise ValueError("Agent step must contain a final answer or a tool call")

        output = execute_tool_call(tool_call)
        tool_execution = ToolExecution(
            tool_call=tool_call,
            output=output,
        )
        tool_executions.append(tool_execution)
        tool_calls.append(tool_call)

        agent_step = generate_agent_step(user_input, tool_executions)

        if agent_step["final_answer"] is not None:
            return AgentRunResult(
                final_answer=agent_step["final_answer"],
                tool_calls=tool_calls,
            )
