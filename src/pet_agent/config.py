from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()   # 加载环境


def _split_env_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


@dataclass(frozen=True)
class Settings:
    app_data_dir: Path
    sqlite_path: Path
    chroma_path: Path
    openai_api_key: str | None
    openai_base_url: str | None
    openai_model: str
    openai_embedding_model: str
    model_provider: str
    dashscope_api_key: str | None
    api_cors_origins: list[str]

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv_if_available()
        try:
            from pet_agent.utils.config_loader import load_app_config, load_model_config
        except ImportError:
            app_conf = {}
            model_conf = {}
        else:
            app_conf = load_app_config()
            model_conf = load_model_config()
        app_data_dir = Path(os.getenv("APP_DATA_DIR", app_conf.get("data_dir", "data/runtime")))
        return cls(
            app_data_dir=app_data_dir,
            sqlite_path=Path(os.getenv("SQLITE_PATH", app_conf.get("sqlite_path", app_data_dir / "pet_products.sqlite3"))),
            chroma_path=Path(os.getenv("CHROMA_PATH", app_conf.get("chroma_path", app_data_dir / "chroma"))),
            openai_api_key=os.getenv("OPENAI_API_KEY") or None,
            openai_base_url=os.getenv("OPENAI_BASE_URL") or None,
            openai_model=os.getenv("OPENAI_MODEL", model_conf.get("chat_model", "gpt-4o-mini")),
            openai_embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", model_conf.get("embedding_model", "text-embedding-3-small")),
            model_provider=os.getenv("MODEL_PROVIDER", model_conf.get("provider", "openai")),
            dashscope_api_key=os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY") or None,
            api_cors_origins=_split_env_list(
                os.getenv(
                    "API_CORS_ORIGINS",
                    ",".join(app_conf.get("api_cors_origins", [])),
                )
            ),
        )
