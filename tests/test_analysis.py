from pet_agent.data.analysis import build_product_analysis, build_review_tags, product_price_band, tag_review_content
from pet_agent.data.cleaning import clean_products, clean_reviews


def test_tag_review_content():
    tags = tag_review_content("声音比较小，清洗也方便，性价比不错")
    assert "静音" in tags
    assert "易清洁" in tags
    assert "性价比" in tags


def test_product_price_band():
    assert product_price_band(None) == "未知价格"
    assert product_price_band(99) == "100元以内"
    assert product_price_band(199) == "100-200元"
    assert product_price_band(399) == "200-500元"


def test_build_analysis_outputs_rows():
    products = clean_products(
        [
            {
                "product_id": "p1",
                "title": "智能宠物饮水机静音循环活水",
                "price": "199",
                "shop_name": "A",
                "sales": "100",
            }
        ]
    )
    reviews = clean_reviews(
        [{"product_id": "p1", "sku_type": "标准", "review_content": "声音比较小，清洗方便"}],
        products,
    )
    product_analysis = build_product_analysis(products, reviews)
    review_tags = build_review_tags(reviews)

    assert product_analysis[0]["review_count"] == 1
    assert "静音" in product_analysis[0]["top_review_tags"]
    assert review_tags[0]["tag_count"] >= 1

