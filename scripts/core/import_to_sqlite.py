from __future__ import annotations

import csv
import re
import sys
from pathlib import Path
from typing import Callable, TypeVar


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pet_agent.config import Settings
from pet_agent.data.schemas import Product, ProductTagStats, Review, ReviewTagEvidence
from pet_agent.storage.sqlite_store import SQLiteStore


T = TypeVar("T")


PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
RAW_PRODUCTS_PATH = RAW_DIR / "raw_products.csv"
RAW_REVIEWS_PATH = RAW_DIR / "raw_reviews.csv"
PRODUCT_TAG_STATS_PATH = PROCESSED_DIR / "product_tag_stats.csv"
REVIEW_TAG_EVIDENCE_PATH = PROCESSED_DIR / "review_tag_evidence.csv"


def text_value(row: dict[str, str], key: str) -> str:
    return (row.get(key) or "").strip()


def int_value(row: dict[str, str], key: str) -> int:
    value = text_value(row, key)
    return int(float(value)) if value else 0


def float_value(row: dict[str, str], key: str) -> float:
    value = text_value(row, key)
    return float(value) if value else 0.0


def optional_float_value(row: dict[str, str], key: str) -> float | None:
    value = text_value(row, key)
    if not value:
        return None
    match = re.search(r"\d+(?:\.\d+)?", value.replace(",", ""))
    return float(match.group(0)) if match else None


def optional_int_sales(row: dict[str, str], key: str) -> int | None:
    value = text_value(row, key).replace(",", "")
    if not value:
        return None
    match = re.search(r"\d+(?:\.\d+)?", value)
    if not match:
        return None
    number = float(match.group(0))
    if "万" in value or "w" in value.lower():
        number *= 10000
    return int(number)


def read_csv_as(path: Path, factory: Callable[[dict[str, str]], T]) -> list[T]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [factory(row) for row in csv.DictReader(handle)]


def product_from_row(row: dict[str, str]) -> Product:
    return Product(
        product_id=text_value(row, "product_id"),
        product_type=text_value(row, "product_type"),
        title=text_value(row, "title"),
        price=optional_float_value(row, "price_raw"),
        shop_name=text_value(row, "shop_name"),
        sales=optional_int_sales(row, "sales_raw"),
        source=text_value(row, "source"),
    )


def review_from_row(row: dict[str, str]) -> Review:
    return Review(
        product_id=text_value(row, "product_id"),
        product_type=text_value(row, "product_type"),
        purchase_date=text_value(row, "purchase_date_raw"),
        sku_type=text_value(row, "sku_type"),
        review_content=text_value(row, "review_content"),
    )


def product_tag_stats_from_row(row: dict[str, str]) -> ProductTagStats:
    return ProductTagStats(
        product_id=text_value(row, "product_id"),
        product_type=text_value(row, "product_type"),
        tag_name=text_value(row, "tag_name"),
        product_review_count=int_value(row, "product_review_count"),
        mention_count=int_value(row, "mention_count"),
        advantage_count=int_value(row, "advantage_count"),
        problem_count=int_value(row, "problem_count"),
        mixed_count=int_value(row, "mixed_count"),
        neutral_count=int_value(row, "neutral_count"),
        mention_rate=float_value(row, "mention_rate"),
        smoothed_advantage_rate=float_value(row, "smoothed_advantage_rate"),
        smoothed_problem_rate=float_value(row, "smoothed_problem_rate"),
        source_method=text_value(row, "source_method"),
        confidence=float_value(row, "confidence"),
        advantage_support=float_value(row, "advantage_support"),
        problem_pressure=float_value(row, "problem_pressure"),
    )


def review_tag_evidence_from_row(row: dict[str, str]) -> ReviewTagEvidence:
    return ReviewTagEvidence(
        review_id=text_value(row, "review_id"),
        product_id=text_value(row, "product_id"),
        product_type=text_value(row, "product_type"),
        sku_type=text_value(row, "sku_type"),
        tag_name=text_value(row, "tag_name"),
        evidence_type=text_value(row, "evidence_type"),
        matched_keyword=text_value(row, "matched_keyword"),
        evidence_text=text_value(row, "evidence_text"),
        evidence_quality=text_value(row, "evidence_quality"),
        source_method=text_value(row, "source_method"),
    )


def main() -> None:
    settings = Settings.from_env()
    store = SQLiteStore(settings.sqlite_path)
    store.init_schema()

    products = read_csv_as(RAW_PRODUCTS_PATH, product_from_row)
    reviews = read_csv_as(RAW_REVIEWS_PATH, review_from_row)
    product_tag_stats = read_csv_as(PRODUCT_TAG_STATS_PATH, product_tag_stats_from_row)
    review_tag_evidence = read_csv_as(REVIEW_TAG_EVIDENCE_PATH, review_tag_evidence_from_row)

    store.replace_products(products)
    store.replace_reviews(reviews)
    store.replace_product_tag_stats(product_tag_stats)
    store.replace_review_tag_evidence(review_tag_evidence)

    print(f"sqlite_path: {settings.sqlite_path}")
    print(f"products: {len(products)}")
    print(f"reviews: {len(reviews)}")
    print(f"product_tag_stats: {len(product_tag_stats)}")
    print(f"review_tag_evidence: {len(review_tag_evidence)}")


if __name__ == "__main__":
    main()
