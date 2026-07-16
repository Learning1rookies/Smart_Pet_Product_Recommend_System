from __future__ import annotations

import os
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Any

from pet_agent.config import Settings
from pet_agent.utils.config_loader import load_model_config


# ----------------------抽象接口： 继承提取方法-------------------
class BaseModelFactory(ABC):
    @abstractmethod
    def generate(self) -> Any:
        raise NotImplementedError

# --------OpenAI聊天模型改配置--------
class OpenAIChatModelFactory(BaseModelFactory):
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings.from_env()
        self.model_conf = load_model_config()

    def generate(self) -> Any:
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            return None
        if not self.settings.openai_api_key:
            return None
        return ChatOpenAI(
            model=self.settings.openai_model or self.model_conf.get("chat_model", "gpt-4o-mini"),
            temperature=float(self.model_conf.get("temperature", 0.2)),
            api_key=self.settings.openai_api_key,
            base_url=self.settings.openai_base_url,
        )

# ------OpenAI嵌入模型配置-------
class OpenAIEmbeddingFactory(BaseModelFactory):
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings.from_env()
        self.model_conf = load_model_config()

    def generate(self) -> Any:
        try:
            from langchain_openai import OpenAIEmbeddings
        except ImportError:
            return None
        if not self.settings.openai_api_key:
            return None
        return OpenAIEmbeddings(
            model=self.settings.openai_embedding_model or self.model_conf.get("embedding_model", "text-embedding-3-small"),
            api_key=self.settings.openai_api_key,
            base_url=self.settings.openai_base_url,
        )

# ---------通义聊天模型改配置------------
class TongyiChatModelFactory(BaseModelFactory):
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings.from_env()
        self.model_conf = load_model_config()

    def generate(self, temperature: float | None = None, response_format: dict[str, Any] | None = None) -> Any:
        if not self.settings.dashscope_api_key:
            raise RuntimeError("DASHSCOPE_API_KEY is not configured.")
        os.environ.setdefault("DASHSCOPE_API_KEY", self.settings.dashscope_api_key)
        try:
            from langchain_community.chat_models.tongyi import ChatTongyi
        except ImportError as exc:
            raise RuntimeError("langchain-community and dashscope are required for Tongyi provider.") from exc
        model = ChatTongyi(
            model=self.settings.openai_model or self.model_conf.get("chat_model", "qwen-plus"),
            temperature=float(temperature if temperature is not None else self.model_conf.get("temperature", 0.2)),
        )
        if response_format:
            model.model_kwargs = {**getattr(model, "model_kwargs", {}), "response_format": response_format}
        return model

# -----DashScope嵌入模型配置------
class DashScopeEmbeddingFactory(BaseModelFactory):
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings.from_env()
        self.model_conf = load_model_config()

    def generate(self) -> Any:
        if not self.settings.dashscope_api_key:
            raise RuntimeError("DASHSCOPE_API_KEY is not configured.")
        os.environ.setdefault("DASHSCOPE_API_KEY", self.settings.dashscope_api_key)
        try:
            from langchain_community.embeddings import DashScopeEmbeddings
        except ImportError as exc:
            raise RuntimeError("langchain-community and dashscope are required for DashScope embeddings.") from exc
        return DashScopeEmbeddings(
            model=self.settings.openai_embedding_model or self.model_conf.get("embedding_model", "text-embedding-v4"),
        )

# -----集成聊天模型--------
@lru_cache(maxsize=8)
def _cached_tongyi_chat_model(
    api_key: str,
    model_name: str,
    temperature: float,
    response_format_key: str,
) -> Any:
    """Reuse a ChatTongyi client for identical runtime settings."""
    settings = Settings.from_env()
    settings = type(settings)(**{**settings.__dict__, "dashscope_api_key": api_key, "openai_model": model_name})
    response_format = {"type": "json_object"} if response_format_key == "json_object" else None
    return TongyiChatModelFactory(settings).generate(temperature=temperature, response_format=response_format)


@lru_cache(maxsize=4)
def _cached_openai_client(api_key: str, base_url: str | None) -> Any:
    from openai import OpenAI

    return OpenAI(api_key=api_key, base_url=base_url)


@lru_cache(maxsize=1)
def get_chat_model() -> Any:
    settings = Settings.from_env()
    provider = (settings.model_provider or "openai").lower()
    if provider in {"tongyi", "dashscope", "qwen"}:
        return _cached_tongyi_chat_model(
            str(settings.dashscope_api_key or ""),
            settings.openai_model,
            float(load_model_config().get("temperature", 0.2)),
            "",
        )
    return OpenAIChatModelFactory(settings).generate()

# ------集成嵌入模型-----
@lru_cache(maxsize=1)
def get_embedding_model() -> Any:
    settings = Settings.from_env()
    provider = (settings.model_provider or "openai").lower()
    if provider in {"tongyi", "dashscope", "qwen"}:
        return DashScopeEmbeddingFactory(settings).generate()
    return OpenAIEmbeddingFactory(settings).generate()


def preload_chat_model() -> None:
    """Construct the configured chat client before the first user request."""
    get_chat_model()

# ----集成聊天文本-----
def generate_chat_text(
    messages: list[dict[str, str]],
    temperature: float | None = None,
    response_format: dict[str, Any] | None = None,
) -> str | None:
    settings = Settings.from_env()   # 虚拟环境
    model_conf = load_model_config()  # 模型配置
    provider = (settings.model_provider or model_conf.get("provider", "openai")).lower()
    if provider in {"tongyi", "dashscope", "qwen"}:
        return _generate_with_tongyi(settings, messages, temperature, response_format=response_format)

    # 判断是否设置openai密钥
    if not settings.openai_api_key:
        return None

    # --------尝试openai连接----------
    try:
        from openai import OpenAI
    except ImportError:
        requests_result = _generate_with_requests(
            settings,
            model_conf,
            messages,
            temperature,
            response_format=response_format,
        )
        if requests_result:
            return requests_result
        return _generate_with_langchain(messages, temperature, response_format=response_format)
    # --------使用代理连接方式------
    client = _cached_openai_client(settings.openai_api_key, settings.openai_base_url)
    payload: dict[str, Any] = {
        "model": settings.openai_model or model_conf.get("chat_model", "gpt-4o-mini"),
        "messages": messages,
        "temperature": float(temperature if temperature is not None else model_conf.get("temperature", 0.2)),
    }
    if response_format:
        payload["response_format"] = response_format
    response = client.chat.completions.create(**payload)
    content = response.choices[0].message.content
    return content.strip() if content else None


# ===========集成使用同一模型============
def _generate_with_tongyi(
    settings: Settings,
    messages: list[dict[str, str]],
    temperature: float | None = None,
    response_format: dict[str, Any] | None = None,
) -> str | None:
    model_conf = load_model_config()
    resolved_temperature = float(temperature if temperature is not None else model_conf.get("temperature", 0.2))
    response_format_key = "json_object" if response_format == {"type": "json_object"} else ""
    model = _cached_tongyi_chat_model(
        str(settings.dashscope_api_key or ""),
        settings.openai_model or model_conf.get("chat_model", "qwen-plus"),
        resolved_temperature,
        response_format_key,
    )
    response = model.invoke([(item["role"], item["content"]) for item in messages])
    content = getattr(response, "content", None)
    return content.strip() if isinstance(content, str) and content.strip() else None


def _generate_with_requests(
    settings: Settings,
    model_conf: dict[str, Any],
    messages: list[dict[str, str]],
    temperature: float | None = None,
    response_format: dict[str, Any] | None = None,
) -> str | None:
    try:
        import requests
    except ImportError:
        return None

    base_url = (settings.openai_base_url or "https://api.openai.com/v1").rstrip("/")
    payload: dict[str, Any] = {
        "model": settings.openai_model or model_conf.get("chat_model", "gpt-4o-mini"),
        "messages": messages,
        "temperature": float(temperature if temperature is not None else model_conf.get("temperature", 0.2)),
    }
    if response_format:
        payload["response_format"] = response_format
    response = requests.post(
        f"{base_url}/chat/completions",
        headers={
            "Authorization": f"Bearer {settings.openai_api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()
    content = data["choices"][0]["message"]["content"]
    return content.strip() if content else None


def _generate_with_langchain(
    messages: list[dict[str, str]],
    temperature: float | None = None,
    response_format: dict[str, Any] | None = None,
) -> str | None:
    model = OpenAIChatModelFactory().generate()
    if model is None:
        return None
    if temperature is not None and hasattr(model, "temperature"):
        model.temperature = temperature
    if response_format and hasattr(model, "model_kwargs"):
        model.model_kwargs = {**getattr(model, "model_kwargs", {}), "response_format": response_format}
    response = model.invoke([(item["role"], item["content"]) for item in messages])
    content = getattr(response, "content", None)
    return content.strip() if isinstance(content, str) and content.strip() else None
