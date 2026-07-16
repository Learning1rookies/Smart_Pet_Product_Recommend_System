from pet_agent.memory.user_memory import UserMemory
from pet_agent.storage.sqlite_store import SQLiteStore
from pet_agent.storage.vector_store import VectorStore
from tests.fakes import FakeEmbeddingModel


def test_user_memory_isolated(tmp_path):
    sqlite_store = SQLiteStore(tmp_path / "test.sqlite3")
    sqlite_store.init_schema()
    vector_store = VectorStore(tmp_path / "vectors", embedding_model=FakeEmbeddingModel())
    memory = UserMemory(sqlite_store, vector_store)

    memory.save_preference("user_a", "budget", {"max": 200})
    memory.save_preference("user_b", "budget", {"max": 800})

    assert memory.load("user_a")["budget"]["max"] == 200
    assert memory.load("user_b")["budget"]["max"] == 800

    results = memory.search_user_memory("user_a", "budget", limit=5)
    assert results
    assert all(item["metadata"]["user_id"] == "user_a" for item in results)


def test_unconfirmed_legacy_purchase_context_is_not_loaded(tmp_path):
    sqlite_store = SQLiteStore(tmp_path / "test.sqlite3")
    sqlite_store.init_schema()
    vector_store = VectorStore(tmp_path / "vectors", embedding_model=FakeEmbeddingModel())
    memory = UserMemory(sqlite_store, vector_store)

    sqlite_store.save_memory("user_a", "active_purchase_context", {"product_type": "智能宠物摄像头"})

    assert "active_purchase_context" not in memory.load("user_a")


def test_purchase_history_keeps_latest_ten_records(tmp_path):
    sqlite_store = SQLiteStore(tmp_path / "test.sqlite3")
    sqlite_store.init_schema()

    removed = None
    for index in range(12):
        history, removed = sqlite_store.append_bounded_memory_list(
            "user_a",
            "confirmed_purchase_history",
            {"product_id": f"p{index}"},
            max_items=10,
        )

    assert len(history) == 10
    assert history[0]["product_id"] == "p2"
    assert history[-1]["product_id"] == "p11"
    assert removed == {"product_id": "p1"}


def test_lightweight_app_users_can_switch_without_mixing_memory(tmp_path):
    sqlite_store = SQLiteStore(tmp_path / "test.sqlite3")
    sqlite_store.init_schema()
    vector_store = VectorStore(tmp_path / "vectors", embedding_model=FakeEmbeddingModel())
    memory = UserMemory(sqlite_store, vector_store)

    user_a = sqlite_store.create_app_user("用户A", user_id="user_a")
    user_b = sqlite_store.create_app_user("用户B", user_id="user_b")
    sqlite_store.touch_app_user(user_b["user_id"])

    users = sqlite_store.list_app_users()
    assert {item["user_id"] for item in users} == {"user_a", "user_b"}

    memory.save_preference(user_a["user_id"], "priority", {"tags": ["噪音/静音"]})
    memory.save_preference(user_b["user_id"], "priority", {"tags": ["画质/夜视"]})

    assert memory.load("user_a")["priority"]["tags"] == ["噪音/静音"]
    assert memory.load("user_b")["priority"]["tags"] == ["画质/夜视"]
