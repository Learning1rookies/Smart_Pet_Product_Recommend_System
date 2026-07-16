from pet_agent.agent.nodes.recommendation import generate_recommendation
from pet_agent.agent.tools.langchain_tools import apply_product_evidence_policy
from pet_agent.retrieval.reranker import EvidenceRerankConfig, EvidenceRerankPipeline
from tests.fakes import FakeReranker


def _evidence(evidence_id: str, product_id: str, text: str, distance: float) -> dict:
    return {
        "id": evidence_id,
        "document": text,
        "metadata": {"product_id": product_id},
        "score": distance,
    }


def test_evidence_policy_moves_unsupported_product_after_supported_product():
    products = [
        {"product_id": "p1", "title": "高分但无证据", "recommendation_score": 0.9},
        {"product_id": "p2", "title": "有证据商品", "recommendation_score": 0.4},
    ]
    evidence = [_evidence("e2", "p2", "运行稳定，画面清晰。", 0.2)]

    ranked_products, ranked_evidence, summary = apply_product_evidence_policy(
        products,
        recalled_evidence=evidence,
        final_evidence=evidence,
        insufficient_penalty=0.2,
    )

    assert ranked_products[0]["product_id"] == "p2"
    assert ranked_products[0]["evidence_status"] == "sufficient"
    assert ranked_products[1]["evidence_status"] == "insufficient"
    assert ranked_products[1]["recommendation_score"] == 0.7
    assert ranked_products[1]["evidence_penalty"] == 0.2
    assert ranked_evidence[0]["candidate_product_rank"] == 1
    assert summary["insufficient_product_ids"] == ["p1"]


def test_reranker_reserves_final_evidence_for_each_represented_product():
    pipeline = EvidenceRerankPipeline(
        reranker=FakeReranker(),
        config=EvidenceRerankConfig(rule_keep_limit=3, final_limit=2, duplicate_threshold=0.9),
    )
    products = [
        {"product_id": "p1", "recommendation_score": 0.9},
        {"product_id": "p2", "recommendation_score": 0.2},
    ]
    evidence = [
        _evidence("p1-a", "p1", "声音小，运行稳定。", 0.01),
        _evidence("p1-b", "p1", "清洗方便，容量充足。", 0.02),
        _evidence("p2-a", "p2", "联网正常，画面清晰。", 0.8),
    ]

    result = pipeline.rerank(
        query="稳定清晰",
        evidence=evidence,
        products=products,
        final_limit=2,
    )

    assert {_item["metadata"]["product_id"] for _item in result} == {"p1", "p2"}


def test_all_candidates_without_evidence_skip_model_and_return_reference(monkeypatch):
    products = [
        {
            "product_id": "p1",
            "title": "候选商品",
            "price": 199.0,
            "sales": 1000,
            "recommendation_score": 0.6,
            "evidence_status": "insufficient",
            "evidence_count": 0,
        }
    ]
    state = {
        "evidence_bundle": {
            "candidate_products": products,
            "review_evidence": [],
        }
    }

    def fail_if_called(_state):
        raise AssertionError("The generation model must not run without evidence.")

    monkeypatch.setattr("pet_agent.agent.nodes.recommendation._llm_recommendation", fail_if_called)

    result = generate_recommendation(state, {"recommendation_limit": 3})
    response = result["response"]

    assert response["recommended_products"] == []
    assert response["comparison_table"][0]["product_id"] == "p1"
    assert response["review_evidence"] == []
    assert response["evidence_status"] == "insufficient"
    assert response["model_used"] is False
    assert "不生成带评论结论" in response["recommendation_reason"]
