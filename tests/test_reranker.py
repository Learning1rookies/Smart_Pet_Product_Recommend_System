from pet_agent.retrieval.reranker import EvidenceRerankConfig, EvidenceRerankPipeline
from tests.fakes import FakeReranker


def test_evidence_rerank_pipeline_dedupes_and_scores():
    pipeline = EvidenceRerankPipeline(
        reranker=FakeReranker(),
        config=EvidenceRerankConfig(rule_keep_limit=10, final_limit=2, duplicate_threshold=0.8),
    )
    products = [
        {"product_id": "p1", "recommendation_score": 0.9},
        {"product_id": "p2", "recommendation_score": 0.3},
    ]
    evidence = [
        {
            "id": "e1",
            "document": "声音小，很安静，清洗方便。",
            "metadata": {"product_id": "p1"},
            "score": 0.1,
        },
        {
            "id": "e1-dup",
            "document": "声音小，很安静，清洗方便。",
            "metadata": {"product_id": "p1"},
            "score": 0.1,
        },
        {
            "id": "e2",
            "document": "有点漏水，声音也比较明显。",
            "metadata": {"product_id": "p2"},
            "score": 0.2,
        },
    ]

    result = pipeline.rerank(
        query="饮水机 静音 避免漏水",
        evidence=evidence,
        products=products,
        priority_tags=["噪音/静音"],
        avoid_tags=["漏水/密封"],
        final_limit=2,
    )

    assert len(result) == 2
    assert {item["id"] for item in result} == {"e1", "e2"}
    assert all("rule_rerank_score" in item for item in result)
    assert all("bge_rerank_score" in item for item in result)
    assert all(item["recall_stage"] == "bge_rerank" for item in result)

