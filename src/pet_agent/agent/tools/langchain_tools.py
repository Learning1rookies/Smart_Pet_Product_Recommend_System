from __future__ import annotations

from collections import Counter
from typing import Any

from pet_agent.agent.tools.product_tools import (
    get_common_product_tags,
    get_product_price_range,
    retrieve_review_evidence,
    search_candidate_products,
)
from pet_agent.agent.tools.tool_compat import invoke_tool, tool
from pet_agent.retrieval.reranker import EvidenceRerankPipeline
from pet_agent.storage.sqlite_store import SQLiteStore
from pet_agent.storage.vector_store import VectorStore


def _evidence_product_id(item: dict[str, Any]) -> str:
    return str((item.get("metadata") or {}).get("product_id") or "")


def apply_product_evidence_policy(
    products: list[dict[str, Any]],
    recalled_evidence: list[dict[str, Any]],
    final_evidence: list[dict[str, Any]],
    *,
    insufficient_penalty: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    recall_counts = Counter(_evidence_product_id(item) for item in recalled_evidence)
    final_counts = Counter(_evidence_product_id(item) for item in final_evidence)
    annotated_products: list[dict[str, Any]] = []

    for product in products:
        product_id = str(product.get("product_id") or "")
        base_score = float(product.get("recommendation_score") or 0.0)
        evidence_count = final_counts[product_id]
        evidence_status = "sufficient" if evidence_count > 0 else "insufficient"
        penalty = 0.0 if evidence_count > 0 else max(0.0, insufficient_penalty)
        annotated_products.append(
            {
                **product,
                "base_recommendation_score": round(base_score, 4),
                "recommendation_score": round(base_score - penalty, 4),
                "evidence_recall_count": recall_counts[product_id],
                "evidence_count": evidence_count,
                "evidence_status": evidence_status,
                "evidence_penalty": round(penalty, 4),
            }
        )

    annotated_products.sort(
        key=lambda item: (
            item.get("evidence_status") == "sufficient",
            float(item.get("recommendation_score") or 0.0),
        ),
        reverse=True,
    )
    product_details = {str(item.get("product_id") or ""): item for item in annotated_products}
    product_ranks = {
        str(item.get("product_id") or ""): rank
        for rank, item in enumerate(annotated_products, start=1)
    }
    annotated_evidence: list[dict[str, Any]] = []
    for item in final_evidence:
        product_id = _evidence_product_id(item)
        product = product_details.get(product_id, {})
        annotated_evidence.append(
            {
                **item,
                "candidate_product_rank": product_ranks.get(product_id),
                "candidate_product_score": product.get("recommendation_score"),
                "candidate_product_evidence_status": product.get("evidence_status"),
            }
        )

    insufficient_product_ids = [
        str(item.get("product_id") or "")
        for item in annotated_products
        if item.get("evidence_status") == "insufficient"
    ]
    summary = {
        "all_candidates_insufficient": bool(annotated_products) and not annotated_evidence,
        "insufficient_product_ids": insufficient_product_ids,
        "supported_product_count": len(annotated_products) - len(insufficient_product_ids),
    }
    return annotated_products, annotated_evidence, summary


def build_product_tools(
    sqlite_store: SQLiteStore,
    vector_store: VectorStore,
    evidence_reranker: EvidenceRerankPipeline,
    global_user_id: str = "global",
) -> list[Any]:
    @tool(
        "load_product_config",
        description="按产品类型读取真实价格区间、常见关注点和常见避免项配置。",
    )
    def load_product_config_tool(product_type: str) -> dict[str, Any]:
        tags = get_common_product_tags(sqlite_store, product_type, limit=10)
        risk_tags = sorted(
            tags,
            key=lambda item: float(item.get("avg_problem_pressure") or 0.0),
            reverse=True,
        )
        return {
            "product_type": product_type,
            "price_range": get_product_price_range(sqlite_store, product_type),
            "priority_options": tags,
            "avoid_options": risk_tags,
        }

    @tool(
        "retrieve_evidence",
        description="根据完整推荐请求检索候选商品、标签评分和评论证据。",
    )
    def retrieve_evidence_tool(
        product_type: str,
        budget_min: float | None = None,
        budget_max: float | None = None,
        priority_tags: list[str] | None = None,
        avoid_tags: list[str] | None = None,
        query: str = "",
        product_limit: int = 3,
        evidence_limit: int = 8,
    ) -> dict[str, Any]:
        priority_tags = priority_tags or []
        avoid_tags = avoid_tags or []
        products = search_candidate_products(
            sqlite_store=sqlite_store,
            product_type=product_type,
            budget_min=budget_min,
            budget_max=budget_max,
            limit=product_limit,
            priority_tags=priority_tags,
            avoid_tags=avoid_tags,
        )
        product_ids = [str(item["product_id"]) for item in products]
        tag_names = sorted(set(priority_tags + avoid_tags))
        semantic_query = f"{query} {product_type} {' '.join(tag_names)}".strip()
        recalled_evidence: list[dict[str, Any]] = []
        products_by_id = {str(item["product_id"]): item for item in products}
        for rank, product_id in enumerate(product_ids, start=1):
            product = products_by_id[product_id]
            product_query = f"{semantic_query} {product.get('title') or ''}".strip()
            product_evidence = retrieve_review_evidence(
                vector_store=vector_store,
                sqlite_store=sqlite_store,
                query=product_query,
                product_type=product_type,
                product_ids=[product_id],
                global_user_id=global_user_id,
                limit=evidence_reranker.config.per_product_recall_limit,
                tag_names=tag_names,
            )
            for item in product_evidence:
                recalled_evidence.append(
                    {
                        **item,
                        "candidate_product_rank": rank,
                        "candidate_product_title": product.get("title"),
                        "candidate_product_score": product.get("recommendation_score"),
                        "recall_query": product_query,
                        "recall_stage": "chroma_per_product",
                    }
                )
        evidence = evidence_reranker.rerank(
            query=semantic_query,
            evidence=recalled_evidence,
            products=products,
            priority_tags=priority_tags,
            avoid_tags=avoid_tags,
            final_limit=evidence_limit,
        )
        products, evidence, evidence_summary = apply_product_evidence_policy(
            products,
            recalled_evidence,
            evidence,
            insufficient_penalty=evidence_reranker.config.insufficient_evidence_penalty,
        )
        product_ids = [str(item["product_id"]) for item in products]
        return {
            "candidate_products": products,
            "review_evidence": evidence,
            "product_ids": product_ids,
            "tag_names": tag_names,
            "recall_count": len(recalled_evidence),
            "evidence_summary": evidence_summary,
            "rerank_flow": [
                "sqlite_candidate_products",
                "chroma_per_product_top20",
                "rule_rerank_dedupe",
                "bge_rerank_top5",
            ],
        }

    @tool(
        "get_product_price_range",
        description="按产品类型查询当前知识库中的最低价、最高价、均价和商品数量。",
    )
    def get_product_price_range_tool(product_type: str | None = None) -> dict[str, Any]:
        return get_product_price_range(sqlite_store, product_type)

    @tool(
        "get_common_product_tags",
        description="按产品类型查询评论统计中最常见的关注标签。",
    )
    def get_common_product_tags_tool(product_type: str, limit: int = 8) -> list[dict[str, Any]]:
        return get_common_product_tags(sqlite_store, product_type, limit=limit)

    @tool(
        "search_candidate_products",
        description="按产品类型、预算、关注标签和避免标签筛选并排序候选商品。",
    )
    def search_candidate_products_tool(
        product_type: str | None = None,
        budget_min: float | None = None,
        budget_max: float | None = None,
        priority_tags: list[str] | None = None,
        avoid_tags: list[str] | None = None,
        query_mode: str | None = None,
        limit: int = 8,
    ) -> list[dict[str, Any]]:
        return search_candidate_products(
            sqlite_store=sqlite_store,
            product_type=product_type,
            budget_min=budget_min,
            budget_max=budget_max,
            limit=limit,
            priority_tags=priority_tags or [],
            avoid_tags=avoid_tags or [],
            query_mode=query_mode,
        )

    @tool(
        "retrieve_review_evidence",
        description="围绕候选商品使用 ChromaDB 语义检索评论证据。",
    )
    def retrieve_review_evidence_tool(
        query: str,
        product_type: str | None = None,
        product_ids: list[str] | None = None,
        tag_names: list[str] | None = None,
        limit: int = 8,
    ) -> list[dict[str, Any]]:
        return retrieve_review_evidence(
            vector_store=vector_store,
            sqlite_store=sqlite_store,
            query=query,
            product_type=product_type,
            product_ids=product_ids or [],
            global_user_id=global_user_id,
            limit=limit,
            tag_names=tag_names or [],
        )

    return [
        load_product_config_tool,
        retrieve_evidence_tool,
        get_product_price_range_tool,
        get_common_product_tags_tool,
        search_candidate_products_tool,
        retrieve_review_evidence_tool,
    ]


def tool_by_name(tools: list[Any]) -> dict[str, Any]:
    return {item.name: item for item in tools}


# 链接已知的工具检查
def invoke_named_tool(tools: list[Any], tool_name: str, payload: dict[str, Any]) -> Any:
    mapping = tool_by_name(tools)
    if tool_name not in mapping:
        raise ValueError(f"Unknown tool: {tool_name}")
    return invoke_tool(mapping[tool_name], payload)
