from pet_agent.data.cleaning import clean_products, clean_reviews, parse_price, parse_sales


def test_parse_price():
    assert parse_price("¥199.90") == 199.90
    assert parse_price("到手价 88 元") == 88
    assert parse_price("") is None


def test_parse_sales():
    assert parse_sales("2300人付款") == 2300
    assert parse_sales("1.2万+人付款") == 12000
    assert parse_sales("暂无") is None


def test_clean_products_and_reviews_deduplicates():
    products = clean_products(
        [
            {"product_id": "p1", "title": "智能宠物饮水机", "price": "199", "shop_name": "A", "sales": "1万"},
            {"product_id": "p1", "title": "重复", "price": "299"},
        ]
    )
    reviews = clean_reviews(
        [
            {"product_id": "p1", "sku_type": "标准", "review_content": "安静好用"},
            {"product_id": "p1", "sku_type": "标准", "review_content": "安静好用"},
            {"product_id": "p1", "sku_type": "标准", "review_content": ""},
        ],
        products,
    )
    assert len(products) == 1
    assert products[0].product_type == "智能宠物饮水机"
    assert len(reviews) == 1
    assert reviews[0].product_type == "智能宠物饮水机"
