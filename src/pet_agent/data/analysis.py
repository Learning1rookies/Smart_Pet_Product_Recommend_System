from __future__ import annotations

from collections import Counter, defaultdict

from pet_agent.data.schemas import Product, Review


REVIEW_TAG_KEYWORDS = {
    "静音": ("静音", "声音小", "声音比较小", "不吵", "安静", "噪音小"),
    "容量": ("容量", "大容量", "够大", "够用", "出差"),
    "易清洁": ("清洗", "清洁", "好洗", "拆洗", "方便清洗"),
    "稳定": ("稳定", "不卡", "不卡粮", "不断网", "顺畅"),
    "防漏": ("漏水", "不漏", "密封", "防漏"),
    "性价比": ("性价比", "便宜", "实惠", "划算", "不贵"),
    "智能联网": ("app", "APP", "联网", "远程", "手机", "wifi", "WiFi"),
    "定位信号": ("定位", "信号", "轨迹", "防丢"),
    "画质夜视": ("画质", "清晰", "夜视", "监控", "云台"),
    "宠物兴趣": ("喜欢玩", "爱玩", "陪玩", "逗猫", "不玩"),
    "佩戴舒适": ("佩戴", "舒服", "轻便", "太重", "项圈"),
    "安全风险": ("安全", "放心", "危险", "夹猫", "电击", "刺激"),
}


def tag_review_content(content: str) -> list[str]:
    text = content or ""
    tags: list[str] = []
    for tag, keywords in REVIEW_TAG_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            tags.append(tag)
    return tags


def product_price_band(price: float | None) -> str:
    if price is None:
        return "未知价格"
    if price < 100:
        return "100元以内"
    if price < 200:
        return "100-200元"
    if price < 500:
        return "200-500元"
    return "500元以上"


def build_product_analysis(products: list[Product], reviews: list[Review]) -> list[dict]:
    reviews_by_product: dict[str, list[Review]] = defaultdict(list)
    tag_counter_by_product: dict[str, Counter[str]] = defaultdict(Counter)
    for review in reviews:
        reviews_by_product[review.product_id].append(review)
        tag_counter_by_product[review.product_id].update(tag_review_content(review.review_content))

    rows: list[dict] = []
    for product in products:
        tag_counter = tag_counter_by_product[product.product_id]
        rows.append(
            {
                "product_id": product.product_id,
                "product_type": product.product_type,
                "title": product.title,
                "shop_name": product.shop_name,
                "price": product.price,
                "sales": product.sales,
                "price_band": product_price_band(product.price),
                "review_count": len(reviews_by_product[product.product_id]),
                "top_review_tags": "、".join(tag for tag, _ in tag_counter.most_common(5)),
            }
        )
    return rows


def build_review_tags(reviews: list[Review]) -> list[dict]:
    rows: list[dict] = []
    for review in reviews:
        tags = tag_review_content(review.review_content)
        rows.append(
            {
                "product_id": review.product_id,
                "product_type": review.product_type,
                "sku_type": review.sku_type,
                "review_content": review.review_content,
                "review_tags": "、".join(tags),
                "tag_count": len(tags),
            }
        )
    return rows
