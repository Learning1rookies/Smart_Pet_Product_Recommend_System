from __future__ import annotations

import time
from collections.abc import Callable
from functools import wraps
from typing import Any

from langgraph.errors import GraphInterrupt

from pet_agent.agent.state import AgentState
from pet_agent.utils.logger_handler import logger

# 用于dbug结果调试的
def log_node(node_name: str):
    def decorator(func: Callable[[AgentState], AgentState]):
        @wraps(func)
        def wrapper(*args, **kwargs) -> AgentState:
            state = args[-1] if args else kwargs.get("state", {})
            started_at = time.perf_counter()
            logger.info(
                "[agent node] start=%s user_id=%s session_id=%s",
                node_name,
                state.get("user_id") if isinstance(state, dict) else None,
                state.get("session_id") if isinstance(state, dict) else None,
            )
            try:
                result = func(*args, **kwargs)
            except GraphInterrupt:
                elapsed_ms = (time.perf_counter() - started_at) * 1000
                logger.info("[agent node] interrupt=%s elapsed_ms=%.2f", node_name, elapsed_ms)
                raise
            except Exception:
                logger.exception("[agent node] failed=%s", node_name)
                raise
            elapsed_ms = (time.perf_counter() - started_at) * 1000
            logger.info("[agent node] done=%s elapsed_ms=%.2f", node_name, elapsed_ms)
            return result

        return wrapper

    return decorator


def log_tool_call(tool_name: str, payload: dict[str, Any] | None = None) -> None:
    logger.info("[tool monitor] tool=%s payload=%s", tool_name, payload or {})
