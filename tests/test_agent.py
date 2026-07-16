import json

from pet_agent.agent.graph import PetRecommendationAgent
from pet_agent.data.cleaning import clean_products, clean_reviews
from pet_agent.data.schemas import ProductTagStats
from pet_agent.retrieval.reranker import EvidenceRerankConfig, EvidenceRerankPipeline
from pet_agent.storage.sqlite_store import SQLiteStore
from pet_agent.storage.vector_store import VectorStore, index_knowledge
from tests.fakes import FakeEmbeddingModel, FakeReranker


def build_agent(tmp_path):
    product_rows = [
        {
            "product_id": "p1",
            "title": "智能宠物饮水机静音循环活水",
            "price": "199",
            "shop_name": "A",
            "sales": "2300",
            "source": "淘宝",
        },
        {
            "product_id": "p2",
            "title": "智能宠物饮水机大容量无线款",
            "price": "399",
            "shop_name": "B",
            "sales": "1200",
            "source": "淘宝",
        },
    ]
    review_rows = [
        {
            "product_id": "p1",
            "purchase_date": "2026-06-01",
            "sku_type": "标准",
            "review_content": "声音小，清洗方便。",
        }
    ]
    products = clean_products(product_rows)
    reviews = clean_reviews(review_rows, products)
    sqlite_store = SQLiteStore(tmp_path / "test.sqlite3")
    sqlite_store.init_schema()
    sqlite_store.replace_products(products)
    sqlite_store.replace_reviews(reviews)
    sqlite_store.replace_product_tag_stats(
        [
            ProductTagStats(
                product_id="p1",
                product_type="智能宠物饮水机",
                tag_name="噪音/静音",
                product_review_count=10,
                mention_count=6,
                advantage_count=5,
                problem_count=0,
                mixed_count=1,
                neutral_count=0,
                mention_rate=0.6,
                smoothed_advantage_rate=0.7,
                smoothed_problem_rate=0.1,
                source_method="test",
                confidence=0.7,
                advantage_support=0.49,
                problem_pressure=0.07,
            )
        ]
    )
    vector_store = VectorStore(tmp_path / "vectors", embedding_model=FakeEmbeddingModel())
    index_knowledge(products, reviews, vector_store)
    evidence_reranker = EvidenceRerankPipeline(
        reranker=FakeReranker(),
        config=EvidenceRerankConfig(per_product_recall_limit=20, rule_keep_limit=20, final_limit=5),
    )
    return PetRecommendationAgent(sqlite_store, vector_store, evidence_reranker=evidence_reranker)


def base_state(message: str, **overrides):
    state = {
        "user_id": "u1",
        "session_id": "s1",
        "thread_id": "s1",
        "message": message,
        "priority_tags": [],
        "avoid_tags": [],
        "history": [],
    }
    state.update(overrides)
    return state


def interrupt_payload(result):
    interrupts = result.get("__interrupt__")
    assert interrupts
    return getattr(interrupts[0], "value")


def fake_model_response(messages, temperature=None, response_format=None):
    system_prompt = messages[0]["content"]
    payload = json.loads(messages[1]["content"])
    message = payload.get("message", "")

    if "问题类型判断节点" in system_prompt:
        if "上一次" in message or "上次购买" in message:
            return json.dumps(
                {
                    "query_type": "direct_answer",
                    "intent": "memory_query",
                    "reason": "confirmed_purchase_history",
                    "direct_answer": "从已确认购买记忆中回答。",
                    "memory_query_scope": "latest",
                },
                ensure_ascii=False,
            )
        if "购买记录" in message or "买过哪些" in message:
            return json.dumps(
                {
                    "query_type": "direct_answer",
                    "intent": "memory_query",
                    "reason": "confirmed_purchase_history",
                    "direct_answer": "从已确认购买记忆中回答。",
                    "memory_query_scope": "history",
                },
                ensure_ascii=False,
            )
        if "1+1" in message:
            return json.dumps(
                {
                    "query_type": "direct_answer",
                    "intent": "general_chat",
                    "reason": "simple_math",
                    "direct_answer": "1+1=2。",
                },
                ensure_ascii=False,
            )
        if "吹风箱" in message:
            return json.dumps(
                {
                    "query_type": "unsupported_product_direct_answer",
                    "intent": "unsupported_product",
                    "reason": "unsupported_product_type",
                    "mentioned_product": "智能宠物吹风箱",
                    "direct_answer": None,
                },
                ensure_ascii=False,
            )
        return json.dumps(
            {
                "query_type": "product_recommend",
                "intent": "recommendation",
                "reason": "purchase_related",
                "direct_answer": None,
            },
            ensure_ascii=False,
        )

    if "结构化需求抽取节点" in system_prompt:
        if "吹风箱" in message:
            data = {
                "product_type": None,
                "mentioned_product": "智能宠物吹风箱",
                "budget_min": None,
                "budget_max": None,
                "priority_tags": [],
                "avoid_tags": [],
                "budget_confirmed": False,
                "priority_confirmed": False,
                "avoid_confirmed": False,
            }
        elif "1000-1200" in message:
            data = {
                "product_type": "智能宠物饮水机",
                "mentioned_product": "智能宠物饮水机",
                "budget_min": 1000,
                "budget_max": 1200,
                "priority_tags": ["噪音/静音"],
                "avoid_tags": ["无特别避免"],
                "budget_confirmed": True,
                "priority_confirmed": True,
                "avoid_confirmed": True,
            }
        elif "推荐一个产品" in message:
            data = {
                "product_type": None,
                "mentioned_product": None,
                "budget_min": None,
                "budget_max": None,
                "priority_tags": [],
                "avoid_tags": [],
                "budget_confirmed": False,
                "priority_confirmed": False,
                "avoid_confirmed": False,
            }
        elif "200" in message:
            data = {
                "product_type": "智能宠物饮水机",
                "mentioned_product": "智能宠物饮水机",
                "budget_min": None,
                "budget_max": 200,
                "priority_tags": ["噪音/静音"],
                "avoid_tags": ["无特别避免"],
                "budget_confirmed": True,
                "priority_confirmed": True,
                "avoid_confirmed": True,
            }
        else:
            data = {
                "product_type": "智能宠物饮水机",
                "mentioned_product": "智能宠物饮水机",
                "budget_min": None,
                "budget_max": None,
                "priority_tags": [],
                "avoid_tags": [],
                "budget_confirmed": False,
                "priority_confirmed": False,
                "avoid_confirmed": False,
            }
        return json.dumps(data, ensure_ascii=False)

    if "推荐生成节点" in system_prompt:
        return "优先推荐测试商品。价格和销量不是实时数据，下单前请以商品页为准。"

    raise AssertionError(f"Unexpected prompt: {system_prompt}")


def patch_model(monkeypatch):
    monkeypatch.setattr("pet_agent.agent.nodes.classify.generate_chat_text", fake_model_response)
    monkeypatch.setattr("pet_agent.agent.nodes.requirements.generate_chat_text", fake_model_response)
    monkeypatch.setattr("pet_agent.agent.nodes.recommendation.generate_chat_text", fake_model_response)


def test_graph_recommends_when_requirement_complete(tmp_path, monkeypatch):
    patch_model(monkeypatch)
    agent = build_agent(tmp_path)

    result = agent.invoke(base_state("我想买 200 元以内的宠物饮水机，最好静音，没有特别避免"))

    assert result["query_type"] == "product_recommend"
    assert result["product_type"] == "智能宠物饮水机"
    assert result["missing_fields"] == []
    assert result["response"]["recommended_products"][0]["product_id"] == "p1"
    assert result["tool_plan"][0]["tool_name"] == "load_product_config"
    assert result["tool_plan"][1]["tool_name"] == "retrieve_evidence"
    assert result["tool_results"]["retrieve_evidence"]["rerank_flow"] == [
        "sqlite_candidate_products",
        "chroma_per_product_top20",
        "rule_rerank_dedupe",
        "bge_rerank_top5",
    ]
    assert "价格和销量不是实时数据" in result["response"]["recommendation_reason"]
    assert "confirmed_purchase_history" not in agent.user_memory.load("u1")


def test_graph_asks_budget_before_recommendation(tmp_path, monkeypatch):
    patch_model(monkeypatch)
    agent = build_agent(tmp_path)

    result = agent.invoke(base_state("我要购买宠物饮水机"))
    payload = interrupt_payload(result)
    state = agent.get_thread_values(user_id="u1", session_id="s1")

    assert state["requirement_status"] == "needs_clarification"
    assert payload["required_action"] == "ask_budget"
    assert "价格参考" in payload["response"]["recommendation_reason"]
    assert payload["response"]["recommended_products"] == []

    first_budget = payload["action_options"][0]
    resumed = agent.resume(
        user_id="u1",
        session_id="s1",
        resume_value={
            "budget_min": first_budget["budget_min"],
            "budget_max": first_budget["budget_max"],
            "budget_confirmed": True,
            "display": first_budget["label"],
        },
    )
    next_payload = interrupt_payload(resumed)
    assert next_payload["required_action"] == "ask_priority_tags"


def test_graph_guides_product_type_when_missing(tmp_path, monkeypatch):
    patch_model(monkeypatch)
    agent = build_agent(tmp_path)

    result = agent.invoke(base_state("你给我推荐一个产品吧"))
    payload = interrupt_payload(result)

    assert payload["missing_fields"] == ["产品类型"]
    assert payload["required_action"] == "select_product_type"
    assert "当前知识库" in payload["response"]["recommendation_reason"]


def test_graph_rejects_unknown_product_without_old_context(tmp_path, monkeypatch):
    patch_model(monkeypatch)
    agent = build_agent(tmp_path)

    result = agent.invoke(base_state("我想买智能宠物吹风箱", product_type="智能宠物饮水机"))

    assert result["query_type"] == "unsupported_product_direct_answer"
    assert result["response"]["recommended_products"] == []
    assert "智能宠物吹风箱" in result["response"]["recommendation_reason"]
    assert "智能宠物饮水机" in result["response"]["recommendation_reason"]


def test_graph_direct_answer_for_non_product_question(tmp_path, monkeypatch):
    patch_model(monkeypatch)
    agent = build_agent(tmp_path)

    result = agent.invoke(base_state("1+1等于多少"))

    assert result["query_type"] == "direct_answer"
    assert result["response"]["recommended_products"] == []
    assert "2" in result["response"]["recommendation_reason"]


def test_graph_reasks_budget_when_user_range_has_no_catalog_overlap(tmp_path, monkeypatch):
    patch_model(monkeypatch)
    agent = build_agent(tmp_path)

    result = agent.invoke(base_state("我想买1000-1200元的宠物饮水机"))
    payload = interrupt_payload(result)
    state = agent.get_thread_values(user_id="u1", session_id="s1")

    assert payload["required_action"] == "ask_budget"
    assert state["budget_validation"] == "outside_catalog"
    assert len(payload["action_options"]) == 5
    assert "无法检索到商品" in payload["response"]["recommendation_reason"]


def test_confirmed_purchase_is_available_in_new_session_and_isolated_by_user(tmp_path, monkeypatch):
    patch_model(monkeypatch)
    agent = build_agent(tmp_path)

    saved = agent.confirm_purchase(
        user_id="u1",
        session_id="purchase-session",
        product={
            "product_id": "p1",
            "title": "智能宠物饮水机静音循环活水",
            "product_type": "智能宠物饮水机",
            "price": 199.0,
            "shop_name": "A",
            "sales": 2300,
            "evidence_status": "sufficient",
        },
        requirement_context={
            "budget_max": 200,
            "priority_tags": ["噪音/静音"],
            "avoid_tags": ["漏水/密封"],
        },
    )

    assert saved["saved"] is True
    assert saved["memory_key"] == "confirmed_purchase_history"
    assert saved["purchase_count"] == 1
    memory = agent.user_memory.load("u1")["confirmed_purchase_history"][-1]
    assert memory["product_id"] == "p1"
    assert memory["source_session_id"] == "purchase-session"

    result = agent.invoke(base_state("我上一次购买的产品是什么？", session_id="new-session"))
    answer = result["response"]["recommendation_reason"]
    assert "智能宠物饮水机静音循环活水" in answer
    assert "199" in answer
    assert "噪音/静音" in answer
    assert result["response"]["model_used"] is False

    agent.confirm_purchase(
        user_id="u1",
        session_id="second-purchase-session",
        product={
            "product_id": "p2",
            "title": "智能宠物摄像头高清夜视款",
            "product_type": "智能宠物摄像头",
            "price": 299.0,
            "shop_name": "B",
            "sales": 1200,
            "evidence_status": "sufficient",
        },
    )
    history_result = agent.invoke(base_state("我的购买记录有哪些？", session_id="history-session"))
    history_answer = history_result["response"]["recommendation_reason"]
    assert "智能宠物摄像头高清夜视款" in history_answer
    assert "智能宠物饮水机静音循环活水" in history_answer

    other_user = agent.invoke(
        base_state("我上一次购买的产品是什么？", user_id="u2", session_id="new-session")
    )
    assert "还没有确认保存" in other_user["response"]["recommendation_reason"]
