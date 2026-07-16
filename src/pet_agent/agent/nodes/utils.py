from __future__ import annotations

import json
import re
from typing import Any

from pet_agent.agent.schemas import ResponsePayload
from pet_agent.agent.tools.product_tools import PRODUCT_KEYWORDS, TAG_KEYWORDS
from pet_agent.data.schemas import PRODUCT_TYPES


PRODUCT_TYPE_ALIASES = {
    "喂食器": "智能宠物喂食器",
    "宠物喂食器": "智能宠物喂食器",
    "智能喂食器": "智能宠物喂食器",
    "饮水机": "智能宠物饮水机",
    "宠物饮水机": "智能宠物饮水机",
    "智能饮水机": "智能宠物饮水机",
    "猫砂盆": "智能宠物猫砂盆",
    "宠物猫砂盆": "智能宠物猫砂盆",
    "智能猫砂盆": "智能宠物猫砂盆",
    "摄像头": "智能宠物摄像头",
    "宠物摄像头": "智能宠物摄像头",
    "智能摄像头": "智能宠物摄像头",
    "项圈": "智能宠物项圈",
    "宠物项圈": "智能宠物项圈",
    "智能项圈": "智能宠物项圈",
    "逗猫器": "智能逗猫器",
    "智能宠物逗猫器": "智能逗猫器",
}


def extract_json_object(text: str | None) -> dict[str, Any]:
    if not text:
        return {}
    raw = text.strip()
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", raw, flags=re.S)
    if not match:
        return {}
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def normalize_product_type(value: object) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text in PRODUCT_TYPES:
        return text
    if text in PRODUCT_TYPE_ALIASES:
        return PRODUCT_TYPE_ALIASES[text]
    for product_type, keywords in PRODUCT_KEYWORDS.items():
        if product_type in text or any(keyword in text for keyword in keywords):
            return product_type
    return None


def to_float_or_none(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def clean_text_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        items = re.split(r"[,，、\s]+", value)
    elif isinstance(value, list):
        items = [str(item) for item in value]
    else:
        return []
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = item.strip()
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def allowed_tags(value: object) -> list[str]:
    allowed = set(TAG_KEYWORDS)
    return [item for item in clean_text_list(value) if item in allowed or item == "无特别避免"]


def supported_product_text() -> str:
    return "、".join(PRODUCT_TYPES)


def make_response(
    answer: str,
    *,
    recommended_products: list[dict[str, Any]] | None = None,
    comparison_table: list[dict[str, Any]] | None = None,
    review_evidence: list[dict[str, Any]] | None = None,
    risk_notes: list[str] | None = None,
    model_used: bool = False,
    **extra: Any,
) -> dict[str, Any]:
    payload = ResponsePayload(
        recommended_products=recommended_products or [],
        comparison_table=comparison_table or [],
        recommendation_reason=answer,
        review_evidence=review_evidence or [],
        risk_notes=risk_notes or [],
    ).to_dict()
    payload["model_used"] = model_used
    payload.update(extra)
    return payload
