from __future__ import annotations

from functools import lru_cache

from pet_agent.agent.graph import PetRecommendationAgent
from pet_agent.config import Settings
from pet_agent.storage.sqlite_store import SQLiteStore
from pet_agent.storage.vector_store import VectorStore


@lru_cache(maxsize=1)
def get_agent() -> PetRecommendationAgent:
    """Create and reuse the runtime Agent for all API requests."""
    settings = Settings.from_env()

    sqlite_store = SQLiteStore(settings.sqlite_path)
    sqlite_store.init_schema()

    vector_store = VectorStore(settings.chroma_path)

    return PetRecommendationAgent(sqlite_store=sqlite_store, vector_store=vector_store)
