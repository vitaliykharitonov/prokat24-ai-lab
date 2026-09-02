from collections.abc import Callable
from typing import TypedDict


class ToolCall(TypedDict):
    tool_name: str
    arguments: dict[str, str | int | bool]


ToolExecutor = Callable[[ToolCall], str]
