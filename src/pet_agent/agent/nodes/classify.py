from __future__ import annotations

import json
from typing import Any

from pet_agent.agent.nodes.utils import extract_json_object, make_response, supported_product_text
from pet_agent.agent.prompts import CLASSIFY_QUERY_PROMPT
from pet_agent.agent.state import AgentState
from pet_agent.model.factory import generate_chat_text


def classify_query(state: AgentState, agent_conf: dict[str, Any]) -> AgentState:
    decision = _llm_classify(state)
    query_type = decision.get("query_type")
    if query_type not in {"product_recommend", "unsupported_product_direct_answer", "direct_answer"}:
        raise RuntimeError(f"Invalid query_type from model: {query_type!r}")
    state["query_type"] = query_type
    default_intent = "recommendation" if query_type == "product_recommend" else "unsupported_product" if query_type == "unsupported_product_direct_answer" else "non_recommendation"
    state["intent"] = str(decision.get("intent") or default_intent)
    state["agent_decision"] = {
        "query_type": query_type,
        "intent": state["intent"],
        "reason": decision.get("reason", ""),
        "mentioned_product": decision.get("mentioned_product"),
        "direct_answer": decision.get("direct_answer"),
        "memory_query_scope": decision.get("memory_query_scope"),
    }
    return state


def direct_answer(state: AgentState) -> AgentState:
    decision = state.get("agent_decision") or {}
    if state.get("intent") == "memory_query":
        state["response"] = make_response(_confirmed_purchase_answer(state))
        return state
    answer = decision.get("direct_answer")
    if not answer:
        raise RuntimeError("Model classified this as direct_answer but did not provide direct_answer.")
    state["response"] = make_response(answer, model_used=True)
    return state


def _confirmed_purchase_answer(state: AgentState) -> str:
    user_memory = state.get("user_memory") or {}
    history = user_memory.get("confirmed_purchase_history")
    if not isinstance(history, list) or not history:
        return "当前账号还没有确认保存的购买记录。完成商品推荐后，可以点击“确认已购买并记住”保存记录。"

    purchases = [item for item in history if isinstance(item, dict)]
    if not purchases:
        return "当前账号还没有确认保存的购买记录。完成商品推荐后，可以点击“确认已购买并记住”保存记录。"

    scope = str((state.get("agent_decision") or {}).get("memory_query_scope") or "latest")
    if scope == "history":
        rows = [f"当前账号保存了最近{len(purchases)}条确认购买记录："]
        for index, purchase in enumerate(reversed(purchases), start=1):
            title = str(purchase.get("title") or "未知商品")
            product_type = str(purchase.get("product_type") or "未记录品类")
            price = purchase.get("price")
            price_text = f"，记录价格{price:g}元" if isinstance(price, (int, float)) else ""
            confirmed_at = str(purchase.get("confirmed_at") or "")[:10]
            date_text = f"，确认日期{confirmed_at}" if confirmed_at else ""
            rows.append(f"{index}. 「{title}」（{product_type}{price_text}{date_text}）")
        rows.append("这些价格均为确认购买时保存的历史快照。")
        return "\n".join(rows)

    purchase = purchases[-1]

    title = str(purchase.get("title") or "未知商品")
    product_type = str(purchase.get("product_type") or "未记录品类")
    details = [f"你上一次确认记录的已购商品是「{title}」，属于{product_type}。"]
    price = purchase.get("price")
    shop_name = purchase.get("shop_name")
    if price not in (None, ""):
        details.append(f"当时知识库记录的价格是{price:g}元。" if isinstance(price, (int, float)) else f"当时记录的价格是{price}元。")
    if shop_name:
        details.append(f"店铺记录为{shop_name}。")
    priority_tags = purchase.get("priority_tags") or []
    avoid_tags = [item for item in (purchase.get("avoid_tags") or []) if item != "无特别避免"]
    if priority_tags:
        details.append(f"当时重点关注：{'、'.join(str(item) for item in priority_tags)}。")
    if avoid_tags:
        details.append(f"当时希望避免：{'、'.join(str(item) for item in avoid_tags)}。")
    details.append("价格和销量属于当时的历史快照，不代表当前商品页面状态。")
    return "".join(details)


def unsupported_product_direct_answer(state: AgentState) -> AgentState:
    """Explain the knowledge boundary without entering recommendation retrieval."""
    decision = state.get("agent_decision") or {}
    mentioned_product = str(decision.get("mentioned_product") or "该产品类型").strip()
    answer = (
        f"当前知识库暂不包含“{mentioned_product}”的商品与评论数据，因此不能基于证据为它做推荐。\n\n"
        f"目前可推荐的品类是：{supported_product_text()}。\n\n"
        "你可以在下一轮直接选择其中一个品类，例如：我想买智能宠物饮水机。"
    )
    state["response"] = make_response(answer)
    return state


def _llm_classify(state: AgentState) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": CLASSIFY_QUERY_PROMPT},
        {
            "role": "user",
            "content": json.dumps(
                {
                    "message": state.get("message", ""),
                    "history": (state.get("history") or [])[-10:],
                    "supported_products": supported_product_text(),
                    "user_memory": state.get("user_memory") or {},
                },
                ensure_ascii=False,
            ),
        },
    ]
    decision = extract_json_object(
        generate_chat_text(
            messages,
            temperature=0.0,
            response_format={"type": "json_object"},
        )
    )
    if not decision:
        raise RuntimeError("Model did not return valid JSON for classify_query.")
    return decision
