from pet_agent.data.cleaning import clean_products, clean_reviews
from pet_agent.storage.vector_store import VectorStore, index_knowledge
from pet_agent.utils.content_hash import ContentHashStore, md5_json
from tests.fakes import FakeEmbeddingModel


def test_md5_json_stable_for_same_payload():
    left = md5_json({"b": 2, "a": 1})
    right = md5_json({"a": 1, "b": 2})
    assert left == right


def test_index_knowledge_can_skip_seen_content(tmp_path):
    products = clean_products(
        [
            {
                "product_id": "p1",
                "title": "智能宠物饮水机",
                "price": "199",
                "shop_name": "A",
                "sales": "100",
            }
        ]
    )
    reviews = clean_reviews(
        [{"product_id": "p1", "sku_type": "标准", "review_content": "安静好用"}],
        products,
    )
    vector_store = VectorStore(tmp_path / "vectors", embedding_model=FakeEmbeddingModel())
    hash_store = ContentHashStore(tmp_path / "md5.txt")

    first = index_knowledge(products, reviews, vector_store, hash_store=hash_store, skip_seen=True)
    second = index_knowledge(products, reviews, vector_store, hash_store=hash_store, skip_seen=True)

    assert first["added"] == 2
    assert first["skipped"] == 0
    assert second["added"] == 0
    assert second["skipped"] == 2
