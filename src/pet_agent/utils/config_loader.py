"""
yml文件路径加载
"""


from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from pet_agent.utils.path_tool import get_abs_path


def _simple_yaml_value(value: str) -> Any:
    value = value.strip()
    if not value:
        return ""
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value.startswith("[") and value.endswith("]"):
        return [item.strip().strip("'\"") for item in value[1:-1].split(",") if item.strip()]
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value.strip("'\"")


def _fallback_yaml_load(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    current_key: str | None = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("- ") and current_key:
            result.setdefault(current_key, []).append(_simple_yaml_value(stripped[2:]))
            continue
        if ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value:
            result[key] = _simple_yaml_value(value)
            current_key = None
        else:
            result[key] = []
            current_key = key
    return result


@lru_cache(maxsize=16)
def load_yaml_config(path: str) -> dict[str, Any]:
    config_path = get_abs_path(path)
    try:
        import yaml
    except ImportError:
        return _fallback_yaml_load(config_path)
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_app_config() -> dict[str, Any]:
    return load_yaml_config("config/app.yml")


def load_model_config() -> dict[str, Any]:
    return load_yaml_config("config/model.yml")


def load_agent_config() -> dict[str, Any]:
    return load_yaml_config("config/agent.yml")


def load_chroma_config() -> dict[str, Any]:
    return load_yaml_config("config/chroma.yml")


def load_rerank_config() -> dict[str, Any]:
    return load_yaml_config("config/rerank.yml")


def load_prompts_config() -> dict[str, Any]:
    return load_yaml_config("config/prompts.yml")

