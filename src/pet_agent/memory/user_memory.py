from __future__ import annotations

from pet_agent.storage.sqlite_store import SQLiteStore
from pet_agent.storage.vector_store import VectorStore
from pet_agent.utils.logger_handler import logger


class UserMemory:
    """Long-term user profile storage.

    Only confirmed memory_event data should enter this class. Conversation
    history stays in AgentState.history and is not persisted here.
    """

    def __init__(self, sqlite_store: SQLiteStore, vector_store: VectorStore):
        self.sqlite_store = sqlite_store
        self.vector_store = vector_store

    def load(self, user_id: str) -> dict[str, object]:
        memory = self.sqlite_store.load_memory(user_id)
        memory.pop("active_purchase_context", None)
        history = memory.get("confirmed_purchase_history")
        legacy_purchase = memory.pop("last_confirmed_purchase", None)
        if not isinstance(history, list):
            history = []
        if not history and isinstance(legacy_purchase, dict):
            history = [legacy_purchase]
        if history:
            memory["confirmed_purchase_history"] = history[-10:]
        return memory

    def save_preference(self, user_id: str, key: str, value: object) -> bool:
        self.sqlite_store.save_memory(user_id, key, value)
        return self._upsert_vector_index(user_id, key, value)

    def confirm_purchase(self, user_id: str, purchase: dict[str, object], max_items: int = 10) -> dict:
        raw_memory = self.sqlite_store.load_memory(user_id)
        initial_items: list[object] = []
        legacy_purchase = raw_memory.get("last_confirmed_purchase")
        if "confirmed_purchase_history" not in raw_memory and isinstance(legacy_purchase, dict):
            initial_items.append(legacy_purchase)
        history, removed = self.sqlite_store.append_bounded_memory_list(
            user_id,
            "confirmed_purchase_history",
            purchase,
            max_items=max_items,
            initial_items=initial_items,
        )
        vector_indexed = self._upsert_vector_index(user_id, "confirmed_purchase_history", history)
        return {
            "saved": True,
            "memory_key": "confirmed_purchase_history",
            "purchase_count": len(history),
            "max_items": max_items,
            "removed_oldest": removed,
            "vector_indexed": vector_indexed,
        }

    def _upsert_vector_index(self, user_id: str, key: str, value: object) -> bool:
        text = f"用户偏好 {key}: {value}"
        try:
            self.vector_store.upsert_texts(
                ids=[f"user:{user_id}:{key}"],
                texts=[text],
                metadatas=[{"doc_type": "user_memory", "user_id": user_id, "memory_key": key}],
            )
        except Exception:
            logger.exception("[user memory] vector index update failed user_id=%s memory_key=%s", user_id, key)
            return False
        return True

    def apply_memory_event(self, user_id: str, memory_event: dict | None) -> dict:
        if not memory_event:
            return {"saved": False, "reason": "no_memory_event"}
        key = str(memory_event.get("memory_key") or memory_event.get("event_type") or "").strip()
        if not key:
            return {"saved": False, "reason": "missing_memory_key"}
        value = memory_event.get("memory_value")
        if value in (None, "", [], {}):
            return {"saved": False, "reason": "empty_memory_value", "memory_key": key}
        vector_indexed = self.save_preference(user_id, key, value)
        return {"saved": True, "memory_key": key, "vector_indexed": vector_indexed}

    def search_user_memory(self, user_id: str, query: str, limit: int = 3) -> list[dict]:
        return self.vector_store.query(
            query,
            where={"doc_type": "user_memory", "user_id": user_id},
            limit=limit,
        )
