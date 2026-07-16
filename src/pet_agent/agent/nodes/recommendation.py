from __future__ import annotations

import json
from typing import Any

from pet_agent.agent.nodes.utils import make_response
from pet_agent.agent.prompts import GENERATE_RECOMMENDATION_PROMPT
from pet_agent.agent.state import AgentState
from pet_agent.agent.tools.langchain_tools import invoke_named_tool
from pet_agent.model.factory import generate_chat_text


def retrieve_evidence(state: AgentState, tools: list[Any], agent_conf: dict[str, Any]) -> AgentState:
    request = state.get("recommendation_request") or {}
    payload = {
        "product_type": request.get("product_type"),
        "budget_min": request.get("budget_min"),
        "budget_max": request.get("budget_max"),
        "priority_tags": request.get("priority_tags") or [],
        "avoid_tags": request.get("avoid_tags") or [],
        "query": state.get("message", ""),
        "product_limit": int(agent_conf.get("recommendation_limit", 3)),
        "evidence_limit": int(agent_conf.get("retrieval_limit", 8)),
    }
    result = invoke_named_tool(tools, "retrieve_evidence", payload)
    state["evidence_bundle"] = result
    state["candidate_products"] = result.get("candidate_products") or []
    state["review_evidence"] = result.get("review_evidence") or []
    state.setdefault("tool_plan", []).append({"tool_name": "retrieve_evidence", "args": payload})
    state.setdefault("tool_results", {})["retrieve_evidence"] = result
    return state


def generate_recommendation(state: AgentState, agent_conf: dict[str, Any]) -> AgentState:
    bundle = state.get("evidence_bundle") or {}
    products = bundle.get("candidate_products") or []
    evidence = bundle.get("review_evidence") or []
    if not products:
        state["response"] = make_response(
            "当前知识库没有找到满足条件的商品，可以放宽预算、减少限制或换一个产品类型。",
            risk_notes=["推荐仅基于已同步到本地知识库的数据，不能代表淘宝实时库存和实时价格。"],
        )
        return state

    if not evidence:
        state["response"] = make_response(
            "当前找到了符合结构化条件的候选商品，但没有检索到可用于支持推荐的评论证据。"
            "下面仅展示商品信息参考，本次不生成带评论结论的首推建议。",
            recommended_products=[],
            comparison_table=_comparison_table(products),
            review_evidence=[],
            risk_notes=[
                "候选商品缺少可用评论证据，不能据此判断实际使用优势或问题。",
                "价格和销量来自本地历史快照，不是实时数据。",
            ],
            evidence_status="insufficient",
        )
        return state

    answer = _with_price_reference(_llm_recommendation(state), state)

    selected = products[: int(agent_conf.get("recommendation_limit", 3))]
    insufficient_products = [
        str(item.get("title") or item.get("product_id") or "未知商品")
        for item in selected
        if item.get("evidence_status") == "insufficient"
    ]
    risk_notes = ["回答只基于本地商品表、评分表和评论证据；价格和销量不是实时数据。"]
    if insufficient_products:
        risk_notes.append(
            f"以下候选缺少最终评论证据，仅依据结构化商品信息参与比较：{'、'.join(insufficient_products)}。"
        )
    state["response"] = make_response(
        answer,
        recommended_products=selected,
        comparison_table=_comparison_table(selected),
        review_evidence=evidence,
        risk_notes=risk_notes,
        model_used=True,
        evidence_status="sufficient",
        memory_action=_purchase_memory_action(selected[0], state),
    )
    return state


def final_response(state: AgentState) -> AgentState:
    state.setdefault("response", make_response("本轮没有生成有效回答。"))
    return state


def _llm_recommendation(state: AgentState) -> str:
    messages = [
        {"role": "system", "content": GENERATE_RECOMMENDATION_PROMPT},
        {
            "role": "user",
            "content": json.dumps(
                {
                    "message": state.get("message", ""),
                    "recommendation_request": state.get("recommendation_request") or {},
                    "evidence_bundle": state.get("evidence_bundle") or {},
                    "user_memory": state.get("user_memory") or {},
                },
                ensure_ascii=False,
            ),
        },
    ]
    answer = generate_chat_text(messages, temperature=0.2)
    if not answer:
        raise RuntimeError("Model did not return text for generate_recommendation.")
    return answer


def _comparison_table(products: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "product_id": item.get("product_id"),
            "title": item.get("title"),
            "price": item.get("price"),
            "shop_name": item.get("shop_name"),
            "sales": item.get("sales"),
            "recommendation_score": item.get("recommendation_score"),
            "base_recommendation_score": item.get("base_recommendation_score"),
            "evidence_recall_count": item.get("evidence_recall_count"),
            "evidence_count": item.get("evidence_count"),
            "evidence_status": item.get("evidence_status"),
        }
        for item in products
    ]


def _purchase_memory_action(product: dict[str, Any], state: AgentState) -> dict[str, Any]:
    request = state.get("recommendation_request") or {}
    return {
        "action": "confirm_purchase",
        "label": "确认已购买并记住",
        "product": {
            "product_id": product.get("product_id"),
            "title": product.get("title"),
            "product_type": product.get("product_type") or request.get("product_type"),
            "price": product.get("price"),
            "shop_name": product.get("shop_name"),
            "sales": product.get("sales"),
            "evidence_status": product.get("evidence_status"),
        },
        "requirement_context": {
            "budget_min": request.get("budget_min"),
            "budget_max": request.get("budget_max"),
            "priority_tags": request.get("priority_tags") or [],
            "avoid_tags": request.get("avoid_tags") or [],
        },
    }


def _with_price_reference(answer: str, state: AgentState) -> str:
    reference = (state.get("recommendation_request") or {}).get("budget_reference") or {}
    catalog_min = reference.get("catalog_min")
    catalog_max = reference.get("catalog_max")
    catalog_avg = reference.get("catalog_avg")
    if catalog_min is None or catalog_max is None:
        return answer
    text = f"价格参考：当前知识库为{catalog_min:g}-{catalog_max:g}元，均价约{catalog_avg:g}元。"
    if reference.get("status") == "partial":
        text += f" 你的预算实际可检索部分为{reference.get('effective_min'):g}-{reference.get('effective_max'):g}元。"
    if text in answer:
        return answer
    return f"{answer}\n\n{text}"
