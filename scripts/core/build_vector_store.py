from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pet_agent.config import Settings
from pet_agent.data.schemas import Product, Review, ReviewTagEvidence
from pet_agent.storage.hash_store import ContentHashStore
from pet_agent.storage.vector_store import VectorStore, product_summary
from pet_agent.utils.content_hash import md5_json
from scripts.core.import_to_sqlite import (
    RAW_PRODUCTS_PATH,
    RAW_REVIEWS_PATH,
    REVIEW_TAG_EVIDENCE_PATH,
    product_from_row,
    read_csv_as,
    review_tag_evidence_from_row,
    review_from_row,
)


EVIDENCE_LIMIT_PER_PRODUCT_TAG_TYPE = 1


def build_vector_payload(
    products: list[Product],
    reviews: list[Review],
    evidence_rows: list[ReviewTagEvidence],
) -> tuple[list[str], list[str], list[dict], list[str]]:
    reviews_by_product: dict[str, list[Review]] = defaultdict(list)
    for review in reviews:
        reviews_by_product[review.product_id].append(review)

    ids: list[str] = []
    texts: list[str] = []
    metadatas: list[dict] = []
    digests: list[str] = []

    for product in products:
        metadata = {
            "doc_type": "product_summary",
            "product_id": product.product_id,
            "product_type": product.product_type,
            "user_id": "global",
        }
        text = product_summary(product, reviews_by_product.get(product.product_id, []))
        digest = md5_json({"text": text, "metadata": metadata})
        ids.append(f"product:{product.product_id}:{digest[:12]}")
        texts.append(text)
        metadatas.append(metadata | {"content_md5": digest})
        digests.append(digest)

    grouped_count: dict[tuple[str, str, str], int] = defaultdict(int)
    for row in evidence_rows:
        key = (row.product_id, row.tag_name, row.evidence_type)
        if grouped_count[key] >= EVIDENCE_LIMIT_PER_PRODUCT_TAG_TYPE:
            continue
        grouped_count[key] += 1
        metadata = {
            "doc_type": "review",
            "product_id": row.product_id,
            "product_type": row.product_type,
            "user_id": "global",
            "tag_name": row.tag_name,
            "evidence_type": row.evidence_type,
            "review_id": row.review_id,
        }
        text = (
            f"标签：{row.tag_name}。证据类型：{row.evidence_type}。"
            f"款式：{row.sku_type}。评论证据：{row.evidence_text}"
        )
        digest = md5_json({"text": text, "metadata": metadata})
        ids.append(f"evidence:{row.product_id}:{row.tag_name}:{row.evidence_type}:{digest[:12]}")
        texts.append(text)
        metadatas.append(metadata | {"content_md5": digest})
        digests.append(digest)

    return ids, texts, metadatas, digests


def main() -> None:
    settings = Settings.from_env()
    products = read_csv_as(RAW_PRODUCTS_PATH, product_from_row)
    reviews = read_csv_as(RAW_REVIEWS_PATH, review_from_row)
    evidence_rows = read_csv_as(REVIEW_TAG_EVIDENCE_PATH, review_tag_evidence_from_row)

    vector_store = VectorStore(settings.chroma_path)
    vector_store.reset()

    hash_store = ContentHashStore(settings.app_data_dir / "content_md5.txt")
    ids, texts, metadatas, digests = build_vector_payload(products, reviews, evidence_rows)
    vector_store.add_texts(ids=ids, texts=texts, metadatas=metadatas)
    hash_store.append_many(digests)

    print(f"chroma_path: {settings.chroma_path}")
    print(f"products: {len(products)}")
    print(f"reviews: {len(reviews)}")
    print(f"review_tag_evidence: {len(evidence_rows)}")
    print(f"vector_documents_added: {len(ids)}")
    print(f"evidence_limit_per_product_tag_type: {EVIDENCE_LIMIT_PER_PRODUCT_TAG_TYPE}")


if __name__ == "__main__":
    main()
