from __future__ import annotations

import re
from typing import Any

from pet_agent.data.schemas import PRODUCT_TYPES
from pet_agent.storage.sqlite_store import SQLiteStore
from pet_agent.storage.vector_store import VectorStore


PRODUCT_KEYWORDS = {
    "智能宠物喂食器": (
        "喂食器",
        "喂食机",
        "投食机",
        "投喂器",
        "自动喂食",
        "定时喂",
        "猫粮机",
        "粮桶",
        "出粮",
    ),
    "智能宠物饮水机": (
        "饮水机",
        "饮水器",
        "喂水器",
        "喝水器",
        "流动水",
        "活水",
        "水盆",
        "水碗",
        "水泵",
        "滤芯",
    ),
    "智能宠物猫砂盆": (
        "猫砂盆",
        "猫厕所",
        "自动铲屎",
        "铲屎机",
        "猫砂桶",
        "除臭",
        "集便",
        "清砂",
    ),
    "智能宠物摄像头": (
        "摄像头",
        "监控",
        "看护",
        "宠物摄像",
        "摄像机",
        "云台",
        "夜视",
        "双向对讲",
        "语音",
    ),
    "智能宠物项圈": (
        "项圈",
        "定位器",
        "追踪器",
        "防丢",
        "轨迹",
        "电子围栏",
        "训练器",
        "止吠",
    ),
    "智能逗猫器": (
        "逗猫",
        "逗猫器",
        "逗猫棒",
        "逗猫玩具",
        "玩具",
        "激光",
        "自动球",
        "陪玩",
        "羽毛",
        "互动",
    ),
}

TAG_KEYWORDS = {
    "噪音/静音": (
        "静音",
        "声音",
        "声音小",
        "声音很小",
        "噪音",
        "噪音小",
        "不吵",
        "安静",
        "吵",
        "吵醒",
        "嗡嗡",
    ),
    "容量/空间": (
        "容量",
        "大容量",
        "容量大",
        "够用",
        "够大",
        "多猫",
        "出差",
        "水箱",
        "粮桶",
        "空间",
        "占地方",
    ),
    "清洁/拆洗": (
        "清洁",
        "清洗",
        "好清洗",
        "方便清洗",
        "容易清洗",
        "好洗",
        "拆洗",
        "好拆",
        "滤芯",
        "耗材",
        "换水",
    ),
    "稳定/卡顿": (
        "稳定",
        "很稳定",
        "顺畅",
        "不卡",
        "不卡粮",
        "卡顿",
        "卡粮",
        "卡住",
        "故障",
        "坏了",
        "不动",
        "失灵",
    ),
    "漏水/密封": (
        "漏水",
        "漏",
        "不漏水",
        "不会漏",
        "密封",
        "密封好",
        "防漏",
        "渗水",
        "溢水",
        "溢出来",
        "漏出来",
    ),
    "性价比": (
        "性价比",
        "性价比高",
        "便宜",
        "实惠",
        "划算",
        "不贵",
        "太贵",
        "值得",
        "不值",
        "物美价廉",
    ),
    "联网/APP": (
        "app",
        "APP",
        "手机",
        "联网",
        "远程",
        "远程控制",
        "手机控制",
        "连接",
        "wifi",
        "WiFi",
        "离线",
        "掉线",
        "断网",
        "连不上",
    ),
    "定位/信号": (
        "定位",
        "定位准确",
        "定位准",
        "定位不准",
        "位置",
        "信号",
        "轨迹",
        "防丢",
        "找回",
        "找不到",
        "漂移",
    ),
    "画质/夜视": (
        "画质",
        "画质清晰",
        "清晰",
        "很清晰",
        "看得清",
        "看不清",
        "夜视",
        "像素",
        "模糊",
        "云台",
    ),
    "宠物兴趣": (
        "猫喜欢",
        "猫咪喜欢",
        "宠物喜欢",
        "狗狗喜欢",
        "很喜欢玩",
        "特别喜欢玩",
        "爱玩",
        "陪玩",
        "追着",
        "有兴趣",
        "不玩",
        "不喜欢",
        "害怕",
        "吓到",
    ),
    "佩戴/舒适": (
        "项圈",
        "佩戴",
        "舒服",
        "不舒服",
        "轻便",
        "太重",
        "重量",
        "重量轻",
        "大小合适",
        "脖子",
        "硅胶",
        "不勒",
        "容易掉",
    ),
    "安全风险": (
        "安全",
        "安全可靠",
        "放心",
        "不伤",
        "危险",
        "不安全",
        "电击",
        "夹猫",
        "夹到",
        "防夹",
        "刺激",
        "吓到",
    ),
}

AVOID_HINTS = ("不要", "不想", "避免", "不能", "别", "不希望", "担心", "怕", "害怕", "容易")
AVOID_SUFFIX_HINTS = ("不要", "不行", "不可接受", "受不了", "很介意")

# =================================================1.基本功能获取=================================
# ----解析价格-----
def parse_price(value: object) -> float | None:
    text = str(value or "").replace(",", "")
    match = re.search(r"\d+(?:\.\d+)?", text)
    return float(match.group(0)) if match else None

# -----抓取商品类型------
def extract_product_type(message: str, explicit: str | None = None) -> str | None:
    if explicit:
        return explicit
    for product_type in PRODUCT_TYPES:
        if product_type in message:
            return product_type
    for product_type, keywords in PRODUCT_KEYWORDS.items():
        if product_type in message or any(keyword in message for keyword in keywords):
            return product_type
    return None

# ------抓取预算------
def extract_budget(
    message: str,
    budget_min: float | None,
    budget_max: float | None,
) -> tuple[float | None, float | None]:
    if budget_min is not None or budget_max is not None:
        return budget_min, budget_max
    under = re.search(r"(\d+(?:\.\d+)?)\s*元?(?:以内|以下|内)", message)
    if under:
        return None, float(under.group(1))
    between = re.search(r"(\d+(?:\.\d+)?)\s*[-到至]\s*(\d+(?:\.\d+)?)", message)
    if between:
        return float(between.group(1)), float(between.group(2))
    price = parse_price(message)
    return None, price

# -----抓取标签------
def extract_tags(message: str, explicit: list[str] | None = None) -> list[str]:
    tags = list(explicit or [])
    for tag, keywords in TAG_KEYWORDS.items():
        if tag not in tags and any(keyword.lower() in message.lower() for keyword in keywords):
            tags.append(tag)
    return tags

# -----抓取避免标签------
def extract_avoid_tags(message: str, explicit: list[str] | None = None) -> list[str]:
    tags = list(explicit or [])
    clauses = [clause for clause in re.split(r"[，,。！？!?；;\s]+", message.lower()) if clause]
    for tag, keywords in TAG_KEYWORDS.items():
        for clause in clauses:
            for keyword in keywords:
                keyword_lower = keyword.lower()
                if keyword_lower not in clause:
                    continue
                keyword_index = clause.find(keyword_lower)
                left_context = clause[max(0, keyword_index - 8) : keyword_index]
                right_context = clause[keyword_index + len(keyword_lower) : keyword_index + len(keyword_lower) + 8]
                if any(hint in left_context for hint in AVOID_HINTS) or any(hint in right_context for hint in AVOID_SUFFIX_HINTS):
                    if tag not in tags:
                        tags.append(tag)
                    break
    return tags

# =============================================2.商品技能具体实现==============================
# -------搜索候选商品------
def search_candidate_products(
    sqlite_store: SQLiteStore,
    product_type: str | None,
    budget_min: float | None,
    budget_max: float | None,
    limit: int,
    priority_tags: list[str] | None = None,
    avoid_tags: list[str] | None = None,
    query_mode: str | None = None,
) -> list[dict]:
    if query_mode in {"price_desc", "price_asc"}:
        return sqlite_store.search_products(
            product_type=product_type,
            budget_min=budget_min,
            budget_max=budget_max,
            limit=limit,
            sort_mode=query_mode,
        )
    if priority_tags or avoid_tags:
        return sqlite_store.rank_products_by_tag_needs(
            product_type=product_type,
            budget_min=budget_min,
            budget_max=budget_max,
            priority_tags=priority_tags,
            avoid_tags=avoid_tags,
            limit=limit,
        )
    return sqlite_store.search_products(
        product_type=product_type,
        budget_min=budget_min,
        budget_max=budget_max,
        limit=limit,
    )

# ----------获取商品价格区间--------
def get_product_price_range(sqlite_store: SQLiteStore, product_type: str | None) -> dict:
    return sqlite_store.product_price_range(product_type)

# -----获取评论商品标签-------
def get_common_product_tags(sqlite_store: SQLiteStore, product_type: str, limit: int = 8) -> list[dict]:
    return sqlite_store.common_tags_for_product_type(product_type, limit=limit)

# -----检索商品评论证据------
def retrieve_review_evidence(
    vector_store: VectorStore,
    sqlite_store: SQLiteStore,
    query: str,
    product_type: str | None,
    product_ids: list[str],
    global_user_id: str,
    limit: int,
    tag_names: list[str] | None = None,
) -> list[dict]:
    if not product_ids:
        return []
    results: list[dict] = []
    for product_id in product_ids:
        evidence = vector_store.query(
            query,
            where={
                "doc_type": "review",
                "product_type": product_type,
                "product_id": str(product_id),
                "user_id": global_user_id,
            },
            limit=limit,
        )
        results.extend(evidence)
    return results
