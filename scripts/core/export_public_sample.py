"""Export a small, pseudonymized dataset for the public repository."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SAMPLE_DIR = PROJECT_ROOT / "data" / "sample"
PRODUCTS_PER_TYPE = 1
REVIEWS_PER_PRODUCT = 5
TEXT_LIMIT = 120


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def shorten(text: str) -> str:
    normalized = " ".join(text.split())
    return normalized[:TEXT_LIMIT] + ("..." if len(normalized) > TEXT_LIMIT else "")


def main() -> None:
    products = read_csv(RAW_DIR / "raw_products.csv")
    reviews = read_csv(RAW_DIR / "raw_reviews.csv")
    evidence = read_csv(PROCESSED_DIR / "review_tag_evidence.csv")
    stats = read_csv(PROCESSED_DIR / "product_tag_stats.csv")

    evidence_count_by_product: dict[str, int] = defaultdict(int)
    for row in evidence:
        evidence_count_by_product[row["product_id"]] += 1
    stats_count_by_product: dict[str, int] = defaultdict(int)
    for row in stats:
        stats_count_by_product[row["product_id"]] += 1

    products_by_type: dict[str, list[dict[str, str]]] = defaultdict(list)
    for product in products:
        products_by_type[product["product_type"]].append(product)

    selected_products = []
    for product_type in sorted(products_by_type):
        ranked_products = sorted(
            products_by_type[product_type],
            key=lambda item: (
                stats_count_by_product[item["product_id"]] > 0,
                evidence_count_by_product[item["product_id"]],
                stats_count_by_product[item["product_id"]],
                item["product_id"],
            ),
            reverse=True,
        )
        selected_products.extend(ranked_products[:PRODUCTS_PER_TYPE])
    id_map = {
        product["product_id"]: f"sample_product_{index:02d}"
        for index, product in enumerate(selected_products, start=1)
    }

    product_fields = ["product_id", "product_type", "title", "shop_name", "price_raw", "sales_raw", "source"]
    public_products = [
        {
            "product_id": id_map[product["product_id"]],
            "product_type": product["product_type"],
            "title": f"{product['product_type']}示例商品",
            "shop_name": "已匿名化店铺",
            "price_raw": product["price_raw"],
            "sales_raw": product["sales_raw"],
            "source": "淘宝历史快照（脱敏样本）",
        }
        for product in selected_products
    ]

    reviews_by_product: dict[str, list[dict[str, str]]] = defaultdict(list)
    for review in reviews:
        if review["product_id"] in id_map:
            reviews_by_product[review["product_id"]].append(review)

    selected_reviews = [
        review
        for product_id in id_map
        for review in reviews_by_product[product_id][:REVIEWS_PER_PRODUCT]
    ]
    review_map = {
        review["row_md5"]: f"sample_review_{index:03d}"
        for index, review in enumerate(selected_reviews, start=1)
    }
    review_fields = ["review_id", "product_id", "product_type", "sku_type", "review_content"]
    public_reviews = [
        {
            "review_id": review_map[review["row_md5"]],
            "product_id": id_map[review["product_id"]],
            "product_type": review["product_type"],
            "sku_type": "已匿名化规格",
            "review_content": shorten(review["review_content"]),
        }
        for review in selected_reviews
    ]

    evidence_fields = [
        "review_id",
        "product_id",
        "product_type",
        "tag_name",
        "evidence_type",
        "matched_keyword",
        "evidence_text",
        "evidence_quality",
        "source_method",
    ]
    public_evidence = [
        {
            "review_id": review_map[row["review_id"]],
            "product_id": id_map[row["product_id"]],
            "product_type": row["product_type"],
            "tag_name": row["tag_name"],
            "evidence_type": row["evidence_type"],
            "matched_keyword": row["matched_keyword"],
            "evidence_text": shorten(row["evidence_text"]),
            "evidence_quality": row["evidence_quality"],
            "source_method": row["source_method"],
        }
        for row in evidence
        if row["product_id"] in id_map and row["review_id"] in review_map
    ]

    stats_fields = [
        "product_id",
        "product_type",
        "tag_name",
        "product_review_count",
        "mention_count",
        "advantage_count",
        "problem_count",
        "confidence",
        "advantage_support",
        "problem_pressure",
        "source_method",
    ]
    public_stats = [
        {field: (id_map[row["product_id"]] if field == "product_id" else row[field]) for field in stats_fields}
        for row in stats
        if row["product_id"] in id_map
    ]

    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(SAMPLE_DIR / "products_sample.csv", public_products, product_fields)
    write_csv(SAMPLE_DIR / "reviews_sample.csv", public_reviews, review_fields)
    write_csv(SAMPLE_DIR / "review_tag_evidence_sample.csv", public_evidence, evidence_fields)
    write_csv(SAMPLE_DIR / "product_tag_statistics_sample.csv", public_stats, stats_fields)
    print(f"Exported public sample to {SAMPLE_DIR}")


if __name__ == "__main__":
    main()
