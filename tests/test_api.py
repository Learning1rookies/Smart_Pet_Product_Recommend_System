from fastapi.testclient import TestClient

from pet_agent.api import main as api_main


class FakeAgent:
    def __init__(self):
        self.last_state = None

    def invoke(self, state):
        self.last_state = state
        return {
            "intent": "recommendation",
            "product_type": "智能宠物饮水机",
            "budget_min": None,
            "budget_max": 200,
            "priority_tags": ["噪音/静音"],
            "avoid_tags": ["无特别避免"],
            "budget_confirmed": True,
            "priority_confirmed": True,
            "avoid_confirmed": True,
            "missing_fields": [],
            "requirement_status": "ready",
            "response": {
                "recommendation_reason": "这是 API 测试回答。",
                "recommended_products": [{"product_id": "p1"}],
                "comparison_table": [],
                "review_evidence": [],
                "risk_notes": ["价格和销量不是实时数据。"],
            },
        }


def test_health_check():
    client = TestClient(api_main.app)

    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_custom_api_reference_page():
    client = TestClient(api_main.app)

    response = client.get("/api-docs")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Smart Pet Agent API" in response.text
    assert "/api/chat" in response.text


def test_chat_accepts_frontend_context_and_returns_runtime_context(monkeypatch):
    fake_agent = FakeAgent()
    monkeypatch.setattr(api_main, "get_agent", lambda: fake_agent)
    client = TestClient(api_main.app)

    response = client.post(
        "/api/chat",
        json={
            "user_id": "user_a",
            "session_id": "session_a",
            "message": "我选择预算 200 元以内",
            "frontend_selection": {
                "product_type": "智能宠物饮水机",
                "budget_max": 200,
                "priority_tags": ["噪音/静音"],
                "avoid_tags": ["无特别避免"],
                "budget_confirmed": True,
                "priority_confirmed": True,
                "avoid_confirmed": True,
            },
            "history": [{"role": "user", "content": "我要买饮水机"}],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "这是 API 测试回答。"
    assert body["product_type"] == "智能宠物饮水机"
    assert body["requirement_status"] == "ready"
    assert body["requirement_context"]["budget_max"] == 200
    assert body["requirement_context"]["budget_confirmed"] is True
    assert body["recommended_products"] == [{"product_id": "p1"}]
    assert fake_agent.last_state["budget_confirmed"] is True
    assert fake_agent.last_state["priority_confirmed"] is True
    assert fake_agent.last_state["avoid_confirmed"] is True


def test_chat_returns_required_action_for_frontend_popup(monkeypatch):
    class NeedBudgetAgent:
        def invoke(self, state):
            return {
                "intent": "recommendation",
                "product_type": "智能宠物饮水机",
                "budget_confirmed": False,
                "priority_confirmed": False,
                "avoid_confirmed": False,
                "missing_fields": ["预算"],
                "requirement_status": "needs_clarification",
                "response": {
                    "recommendation_reason": "请选择预算。",
                    "required_action": "ask_budget",
                    "action_options": [{"label": "0-200元", "budget_min": 0, "budget_max": 200}],
                    "recommended_products": [],
                    "comparison_table": [],
                    "review_evidence": [],
                    "risk_notes": [],
                },
            }

    monkeypatch.setattr(api_main, "get_agent", lambda: NeedBudgetAgent())
    client = TestClient(api_main.app)

    response = client.post(
        "/api/chat",
        json={
            "user_id": "user_a",
            "session_id": "session_a",
            "message": "我要买饮水机",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["required_action"] == "ask_budget"
    assert body["action_options"][0]["budget_max"] == 200
    assert body["missing_fields"] == ["预算"]
