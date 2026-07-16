from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


try:
    from langchain_core.tools import tool as lc_tool
except ImportError:
    lc_tool = None


@dataclass(frozen=True)
class LocalTool:
    name: str
    description: str
    func: Callable[..., Any]

    def invoke(self, input: dict[str, Any] | None = None, **kwargs: Any) -> Any:
        payload = input or kwargs
        return self.func(**payload)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.func(*args, **kwargs)


def tool(name: str | None = None, description: str | None = None):
    """Use LangChain @tool when installed, otherwise expose a compatible local tool."""
    if lc_tool is not None:
        return lc_tool(name, description=description) if name else lc_tool

    def decorator(func: Callable[..., Any]) -> LocalTool:
        tool_name = name or func.__name__
        tool_description = description or (func.__doc__ or "").strip()
        return LocalTool(name=tool_name, description=tool_description, func=func)

    return decorator


def invoke_tool(tool_obj: Any, payload: dict[str, Any]) -> Any:
    if hasattr(tool_obj, "invoke"):
        return tool_obj.invoke(payload)
    return tool_obj(**payload)
