from __future__ import annotations

import re
from collections.abc import Iterable, Mapping

from pet_agent.data.schemas import PRODUCT_TYPES, Product, Review


PRODUCT_KEYWORDS = {
    "智能宠物喂食器": ("喂食器", "喂食机", "投食机", "投喂器", "自动喂食", "猫粮机", "粮桶", "出粮"),
    "智能宠物饮水机": ("饮水机", "饮水器", "喂水器", "喝水器", "流动水", "活水", "水泵", "滤芯"),
    "智能宠物猫砂盆": ("猫砂盆", "猫厕所", "自动铲屎", "铲屎机", "除臭", "集便", "清砂"),
    "智能宠物摄像头": ("摄像头", "监控", "看护", "摄像机", "云台", "夜视", "双向对讲"),
    "智能宠物项圈": ("项圈", "定位器", "追踪器", "防丢", "轨迹", "电子围栏", "训练器", "止吠"),
    "智能逗猫器": ("逗猫", "逗猫器", "逗猫棒", "玩具", "激光", "自动球", "陪玩", "羽毛", "互动"),
}


def parse_price(value: object) -> float | None:
    text = str(value or "").replace(",", "")
    match = re.search(r"\d+(?:\.\d+)?", text)
    return float(match.group(0)) if match else None


def parse_sales(value: object) -> int | None:
    text = str(value or "").replace(",", "").strip()
    match = re.search(r"(\d+(?:\.\d+)?)\s*(万)?", text)
    if not match:
        return None
    number = float(match.group(1))
    if match.group(2):
        number *= 10000
    return int(number)


def infer_product_type(title: str, fallback: str | None = None) -> str:
    if fallback in PRODUCT_TYPES:
        return str(fallback)
    for product_type in PRODUCT_TYPES:
        if product_type in title:
            return product_type
    for product_type, keywords in PRODUCT_KEYWORDS.items():
        if any(keyword in title for keyword in keywords):
            return product_type
    return fallback or "未知类型"


def clean_products(rows: Iterable[Mapping[str, object]]) -> list[Product]:
    products: list[Product] = []
    seen_ids: set[str] = set()
    for row in rows:
        product_id = _clean_text(row.get("product_id"))
        if not product_id or product_id in seen_ids:
            continue
        title = _clean_text(row.get("title"))
        product_type = infer_product_type(title, _clean_text(row.get("product_type")) or None)
        products.append(
            Product(
                product_id=product_id,
                product_type=product_type,
                title=title,
                price=parse_price(row.get("price")),
                shop_name=_clean_text(row.get("shop_name") or row.get("shpo_name")),
                sales=parse_sales(row.get("sales") or row.get("sales_count")),
                source=_clean_text(row.get("source")),
            )
        )
        seen_ids.add(product_id)
    return products


def clean_reviews(rows: Iterable[Mapping[str, object]], products: Iterable[Product]) -> list[Review]:
    product_type_by_id = {product.product_id: product.product_type for product in products}
    reviews: list[Review] = []
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        product_id = _clean_text(row.get("product_id"))
        review_content = _clean_text(row.get("review_content"))
        if not product_id or not review_content or product_id not in product_type_by_id:
            continue
        sku_type = _clean_text(row.get("sku_type"))
        key = (product_id, sku_type, review_content)
        if key in seen:
            continue
        reviews.append(
            Review(
                product_id=product_id,
                product_type=product_type_by_id[product_id],
                purchase_date=_clean_text(row.get("purchase_date")),
                sku_type=sku_type,
                review_content=review_content,
            )
        )
        seen.add(key)
    return reviews


def _clean_text(value: object) -> str:
    text = str(value or "").strip()
    return "" if text.lower() in {"null", "none", "nan"} else text
