from __future__ import annotations

import json
from typing import Any

from langgraph.types import interrupt

from pet_agent.agent.nodes.utils import (
    allowed_tags,
    extract_json_object,
    make_response,
    normalize_product_type,
    supported_product_text,
    to_float_or_none,
)
from pet_agent.agent.prompts import EXTRACT_REQUIREMENT_PROMPT
from pet_agent.agent.state import AgentState
from pet_agent.agent.tools.langchain_tools import invoke_named_tool
from pet_agent.data.schemas import PRODUCT_TYPES
from pet_agent.model.factory import generate_chat_text


def extract_requirement(state: AgentState, agent_conf: dict[str, Any]) -> AgentState:
    extracted = _llm_extract(state)

    extracted_product_type = normalize_product_type(extracted.get("product_type"))
    # 如果抓取到产品类型就获取产品类型名称
    if extracted_product_type:
        product_type = extracted_product_type
    # 未抓取到就提及产品类型设为空
    elif extracted.get("mentioned_product"):
        product_type = None
    else:
        product_type = normalize_product_type(state.get("product_type"))
    mentioned_product = str(extracted.get("mentioned_product") or extracted.get("product_type") or "").strip() or None
    state["product_type"] = product_type
    state["mentioned_product"] = mentioned_product
    state["budget_min"] = to_float_or_none(extracted.get("budget_min")) if extracted.get("budget_min") is not None else state.get("budget_min")
    state["budget_max"] = to_float_or_none(extracted.get("budget_max")) if extracted.get("budget_max") is not None else state.get("budget_max")
    state["priority_tags"] = allowed_tags(extracted.get("priority_tags")) or list(state.get("priority_tags") or [])
    state["avoid_tags"] = allowed_tags(extracted.get("avoid_tags")) or list(state.get("avoid_tags") or [])
    state["budget_confirmed"] = bool(extracted.get("budget_confirmed")) or bool(state.get("budget_confirmed"))
    state["priority_confirmed"] = bool(extracted.get("priority_confirmed")) or bool(state.get("priority_confirmed"))
    state["avoid_confirmed"] = bool(extracted.get("avoid_confirmed")) or bool(state.get("avoid_confirmed")) or "无特别避免" in state["avoid_tags"]
    return state


def guide_product_type(state: AgentState) -> AgentState:
    mentioned = state.get("mentioned_product")
    if mentioned:
        answer = f"当前知识库暂不支持「{mentioned}」的商品推荐。我现在能处理的品类是：{supported_product_text()}。"
    else:
        answer = f"你可以先选择一个商品品类。当前知识库支持：{supported_product_text()}。"
    state["missing_fields"] = ["产品类型"]
    state["requirement_status"] = "need_product_type"
    state["required_action"] = "select_product_type"
    response = make_response(
        answer,
        required_action="select_product_type",
        action_options=list(PRODUCT_TYPES),
    )
    state["response"] = response
    resume_value = interrupt(_interrupt_payload(state, response))
    product_type = normalize_product_type(_resume_value(resume_value, "product_type"))
    if not product_type:
        product_type = normalize_product_type(_resume_value(resume_value, "value"))
    if not product_type:
        raise ValueError("resume product_type is required for select_product_type.")
    state["product_type"] = product_type
    state["mentioned_product"] = product_type
    state["required_action"] = None
    state["response"] = make_response(f"已选择品类：{product_type}。")
    return state


def load_product_config(state: AgentState, tools: list[Any]) -> AgentState:
    result = invoke_named_tool(tools, "load_product_config", {"product_type": state.get("product_type")})
    state["product_config"] = result
    state.setdefault("tool_plan", []).append({"tool_name": "load_product_config", "args": {"product_type": state.get("product_type")}})
    state.setdefault("tool_results", {})["load_product_config"] = result
    return state


def check_missing_fields(state: AgentState) -> AgentState:
    missing: list[str] = []
    if not state.get("product_type"):
        missing.append("产品类型")
    budget_reference = _build_budget_reference(state)
    state["budget_reference"] = budget_reference
    state["budget_validation"] = str(budget_reference.get("status") or "missing")
    if budget_reference["status"] in {"missing", "invalid", "outside_catalog", "unavailable"}:
        missing.append("预算")
        state["budget_confirmed"] = False
    if not state.get("priority_tags") and not state.get("priority_confirmed"):
        missing.append("核心关注点")
    if not state.get("avoid_tags") and not state.get("avoid_confirmed"):
        missing.append("避免项")
    state["missing_fields"] = missing
    state["requirement_status"] = "ready" if not missing else "needs_clarification"
    return state


def ask_budget(state: AgentState) -> AgentState:
    config = state.get("product_config") or {}
    price_range = config.get("price_range") or {}
    reference = state.get("budget_reference") or _build_budget_reference(state)
    reference_text = _price_reference_text(reference)
    if reference.get("status") in {"invalid", "outside_catalog"}:
        answer = (
            f"你输入的预算{_requested_budget_text(reference)}在当前{state.get('product_type')}知识库中无法检索到商品。\n\n"
            f"{reference_text}\n\n"
            "请从下面 5 个真实价格档位中重新选择。"
        )
    elif reference.get("status") == "unavailable":
        answer = f"当前{state.get('product_type')}缺少可用价格数据，暂时不能生成预算档位。你可以选择暂不确定后继续。"
    else:
        answer = f"{reference_text} 你可以选择预算区间，或告诉我暂不确定。"
    state["required_action"] = "ask_budget"
    response = make_response(
        answer,
        required_action="ask_budget",
        action_options=_budget_options(price_range),
    )
    state["response"] = response
    resume_value = interrupt(_interrupt_payload(state, response))
    state["budget_min"] = to_float_or_none(_resume_value(resume_value, "budget_min"))
    state["budget_max"] = to_float_or_none(_resume_value(resume_value, "budget_max"))
    state["budget_confirmed"] = True
    state["required_action"] = None
    state["response"] = make_response(f"已确认预算：{_resume_value(resume_value, 'display') or _format_budget_resume(state)}。")
    return state


def ask_priority_tags(state: AgentState) -> AgentState:
    options = [item["tag_name"] for item in (state.get("product_config") or {}).get("priority_options", [])]
    answer = (
        f"{_price_reference_text(state.get('budget_reference') or _build_budget_reference(state))}\n\n"
        f"{state.get('product_type')}常见关注点包括：{'、'.join(options) if options else '性价比、销量、评论证据'}。请选择你最在意的点。"
    )
    state["required_action"] = "ask_priority_tags"
    response = make_response(
        answer,
        required_action="ask_priority_tags",
        action_options=options,
    )
    state["response"] = response
    resume_value = interrupt(_interrupt_payload(state, response))
    tags = allowed_tags(_resume_value(resume_value, "priority_tags") or _resume_value(resume_value, "tags"))
    if not tags:
        tags = allowed_tags(options[:3])
    state["priority_tags"] = tags
    state["priority_confirmed"] = True
    state["required_action"] = None
    state["response"] = make_response(f"已确认关注点：{'、'.join(tags) if tags else '综合表现'}。")
    return state


def ask_avoid_tags(state: AgentState) -> AgentState:
    options = [item["tag_name"] for item in (state.get("product_config") or {}).get("avoid_options", [])]
    answer = f"你有没有特别想避免的问题？可选项包括：{'、'.join(options) if options else '暂无高频风险项'}。也可以选择无特别避免。"
    state["required_action"] = "ask_avoid_tags"
    response = make_response(
        answer,
        required_action="ask_avoid_tags",
        action_options=[*options, "无特别避免"],
    )
    state["response"] = response
    resume_value = interrupt(_interrupt_payload(state, response))
    tags = allowed_tags(_resume_value(resume_value, "avoid_tags") or _resume_value(resume_value, "tags"))
    if not tags:
        tags = ["无特别避免"]
    state["avoid_tags"] = tags
    state["avoid_confirmed"] = True
    state["required_action"] = None
    state["response"] = make_response(f"已确认避免项：{'、'.join(tags)}。")
    return state


def requirement_complete(state: AgentState) -> AgentState:
    state["required_action"] = None
    state["recommendation_request"] = {
        "user_id": state.get("user_id"),
        "session_id": state.get("session_id"),
        "product_type": state.get("product_type"),
        "budget_min": state.get("budget_min"),
        "budget_max": state.get("budget_max"),
        "budget_reference": state.get("budget_reference") or {},
        "pet_type": state.get("pet_type"),
        "priority_tags": state.get("priority_tags") or [],
        "avoid_tags": [tag for tag in (state.get("avoid_tags") or []) if tag != "无特别避免"],
        "conversation_history": state.get("history") or [],
    }
    state["requirement_status"] = "ready"
    return state


def _llm_extract(state: AgentState) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": EXTRACT_REQUIREMENT_PROMPT},
        {
            "role": "user",
            "content": json.dumps(
                {
                    "message": state.get("message", ""),
                    "history": (state.get("history") or [])[-10:],
                    "current_requirement": {
                        "product_type": state.get("product_type"),
                        "budget_min": state.get("budget_min"),
                        "budget_max": state.get("budget_max"),
                        "priority_tags": state.get("priority_tags") or [],
                        "avoid_tags": state.get("avoid_tags") or [],
                    },
                    "user_memory": state.get("user_memory") or {},
                    "supported_product_types": list(PRODUCT_TYPES),
                },
                ensure_ascii=False,
            ),
        },
    ]
    extracted = extract_json_object(
        generate_chat_text(
            messages,
            temperature=0.0,
            response_format={"type": "json_object"},
        )
    )
    # 如果模型没有返回json文本LLM抓取需求失败
    if not extracted:
        raise RuntimeError("Model did not return valid JSON for extract_requirement.")
    return extracted


def _fmt_price(value: object) -> str:
    number = to_float_or_none(value)
    if number is None:
        return "未知"
    return str(int(number)) if number.is_integer() else f"{number:.2f}"


def _budget_options(price_range: dict[str, Any]) -> list[dict[str, Any]]:
    min_price = to_float_or_none(price_range.get("min_price")) or 0
    max_price = to_float_or_none(price_range.get("max_price")) or 0
    if max_price <= 0:
        return [{"label": "暂不确定", "budget_min": None, "budget_max": None}]
    if max_price <= min_price:
        return [{"label": f"{_fmt_price(min_price)}元左右", "budget_min": min_price, "budget_max": max_price}]
    step = (max_price - min_price) / 5
    options = []
    for index in range(5):
        start = min_price + step * index
        end = max_price if index == 4 else min_price + step * (index + 1)
        options.append(
            {
                "label": f"{_fmt_price(start)}-{_fmt_price(end)}元",
                "budget_min": round(start, 2),
                "budget_max": round(end, 2),
            }
        )
    return options


def _build_budget_reference(state: AgentState) -> dict[str, Any]:
    """Compare user budget with the real price range for the selected product type."""
    price_range = (state.get("product_config") or {}).get("price_range") or {}
    catalog_min = to_float_or_none(price_range.get("min_price"))
    catalog_max = to_float_or_none(price_range.get("max_price"))
    catalog_avg = to_float_or_none(price_range.get("avg_price"))
    requested_min = to_float_or_none(state.get("budget_min"))
    requested_max = to_float_or_none(state.get("budget_max"))
    reference = {
        "catalog_min": catalog_min,
        "catalog_max": catalog_max,
        "catalog_avg": catalog_avg,
        "requested_min": requested_min,
        "requested_max": requested_max,
        "effective_min": None,
        "effective_max": None,
        "status": "missing",
    }
    if requested_min is None and requested_max is None:
        return reference
    if catalog_min is None or catalog_max is None or catalog_min > catalog_max:
        reference["status"] = "unavailable"
        return reference
    if requested_min is not None and requested_max is not None and requested_min > requested_max:
        reference["status"] = "invalid"
        return reference

    effective_min = max(catalog_min, requested_min) if requested_min is not None else catalog_min
    effective_max = min(catalog_max, requested_max) if requested_max is not None else catalog_max
    if effective_min > effective_max:
        reference["status"] = "outside_catalog"
        return reference
    reference["effective_min"] = effective_min
    reference["effective_max"] = effective_max
    is_partial = (
        (requested_min is not None and requested_min < catalog_min)
        or (requested_max is not None and requested_max > catalog_max)
    )
    reference["status"] = "partial" if is_partial else "matched"
    return reference


def _requested_budget_text(reference: dict[str, Any]) -> str:
    requested_min = reference.get("requested_min")
    requested_max = reference.get("requested_max")
    if requested_min is not None and requested_max is not None:
        return f"{_fmt_price(requested_min)}-{_fmt_price(requested_max)}元"
    if requested_max is not None:
        return f"{_fmt_price(requested_max)}元以内"
    if requested_min is not None:
        return f"{_fmt_price(requested_min)}元以上"
    return ""


def _price_reference_text(reference: dict[str, Any]) -> str:
    catalog_min = reference.get("catalog_min")
    catalog_max = reference.get("catalog_max")
    catalog_avg = reference.get("catalog_avg")
    if catalog_min is None or catalog_max is None:
        return "当前知识库暂未形成可用价格参考。"
    text = f"当前知识库价格参考：{_fmt_price(catalog_min)}-{_fmt_price(catalog_max)}元，均价约{_fmt_price(catalog_avg)}元。"
    if reference.get("status") == "partial":
        text += f" 你的预算可检索部分是：{_fmt_price(reference.get('effective_min'))}-{_fmt_price(reference.get('effective_max'))}元。"
    elif reference.get("status") == "matched":
        text += f" 已记录你的预算：{_requested_budget_text(reference)}。"
    return text


def _interrupt_payload(state: AgentState, response: dict[str, Any]) -> dict[str, Any]:
    return {
        "required_action": response.get("required_action") or state.get("required_action"),
        "action_options": response.get("action_options") or [],
        "response": response,
        "product_type": state.get("product_type"),
        "missing_fields": state.get("missing_fields") or [],
    }


def _resume_value(resume_value: Any, key: str) -> Any:
    if isinstance(resume_value, dict):
        return resume_value.get(key)
    return None


def _format_budget_resume(state: AgentState) -> str:
    min_value = state.get("budget_min")
    max_value = state.get("budget_max")
    if min_value is None and max_value is None:
        return "全价格范围"
    left = _fmt_price(min_value) if min_value is not None else "不限"
    right = _fmt_price(max_value) if max_value is not None else "不限"
    return f"{left}-{right}元"
