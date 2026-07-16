from __future__ import annotations

from pet_agent.utils.config_loader import load_prompts_config
from pet_agent.utils.path_tool import get_abs_path


def _load_prompt(config_key: str) -> str:
    prompts_conf = load_prompts_config()
    prompt_path = prompts_conf[config_key]
    return get_abs_path(prompt_path).read_text(encoding="utf-8")

# 系统提示词加载
def load_system_prompt() -> str:
    return _load_prompt("main_prompt_path")

# 推荐提示词加载
def load_recommendation_prompt() -> str:
    return _load_prompt("recommendation_prompt_path")

# 边界提示词加载
def load_guardrail_prompt() -> str:
    return _load_prompt("guardrail_prompt_path")

