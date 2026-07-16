from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pet_agent.agent.graph import PetRecommendationAgent
from pet_agent.config import Settings
from pet_agent.storage.sqlite_store import SQLiteStore
from pet_agent.storage.vector_store import VectorStore


def build_studio_graph():
    settings = Settings.from_env()
    sqlite_store = SQLiteStore(settings.sqlite_path)
    sqlite_store.init_schema()
    vector_store = VectorStore(settings.chroma_path)
    agent = PetRecommendationAgent(sqlite_store, vector_store)
    # Studio/Agent Server owns checkpoint persistence and rejects graphs that
    # bundle a local checkpointer such as InMemorySaver.
    return agent.build_studio_graph()


graph = build_studio_graph()
