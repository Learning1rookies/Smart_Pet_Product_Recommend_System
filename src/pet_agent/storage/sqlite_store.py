from __future__ import annotations

import json
import sqlite3
import uuid
from pathlib import Path
from typing import Iterable

from pet_agent.data.schemas import Product, ProductTagStats, Review, ReviewTagEvidence


def _product_order_by(sort_mode: str | None) -> str:
    if sort_mode == "price_desc":
        return "COALESCE(price, -1) DESC, COALESCE(sales, 0) DESC"
    if sort_mode == "price_asc":
        return "COALESCE(price, 999999) ASC, COALESCE(sales, 0) DESC"
    return "COALESCE(sales, 0) DESC, COALESCE(price, 999999) ASC"


class SQLiteStore:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def init_schema(self) -> None:
        with self.connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS products (
                    product_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    price REAL,
                    shop_name TEXT,
                    sales INTEGER,
                    source TEXT,
                    product_type TEXT
                );
                CREATE TABLE IF NOT EXISTS reviews (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    product_id TEXT NOT NULL,
                    purchase_date TEXT,
                    sku_type TEXT,
                    review_content TEXT NOT NULL,
                    product_type TEXT,
                    UNIQUE(product_id, sku_type, review_content)
                );
                CREATE TABLE IF NOT EXISTS user_memory (
                    user_id TEXT NOT NULL,
                    memory_key TEXT NOT NULL,
                    memory_value TEXT NOT NULL,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY(user_id, memory_key)
                );
                CREATE TABLE IF NOT EXISTS memory_events (
                    event_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    status TEXT DEFAULT 'active',
                    supersedes_event_id TEXT,
                    valid_from TEXT DEFAULT CURRENT_TIMESTAMP,
                    valid_to TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS user_profiles (
                    user_id TEXT PRIMARY KEY,
                    profile_json TEXT NOT NULL,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS session_summaries (
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    summary_json TEXT NOT NULL,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY(user_id, session_id)
                );
                CREATE TABLE IF NOT EXISTS conversations (
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS app_users (
                    user_id TEXT PRIMARY KEY,
                    display_name TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    last_active_at TEXT DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS product_tag_stats (
                    product_id TEXT NOT NULL,
                    product_type TEXT,
                    tag_name TEXT NOT NULL,
                    product_review_count INTEGER,
                    mention_count INTEGER,
                    advantage_count INTEGER,
                    problem_count INTEGER,
                    mixed_count INTEGER,
                    neutral_count INTEGER,
                    mention_rate REAL,
                    smoothed_advantage_rate REAL,
                    smoothed_problem_rate REAL,
                    confidence REAL,
                    advantage_support REAL,
                    problem_pressure REAL,
                    source_method TEXT,
                    PRIMARY KEY(product_id, tag_name)
                );
                CREATE TABLE IF NOT EXISTS review_tag_evidence (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    review_id TEXT NOT NULL,
                    product_id TEXT NOT NULL,
                    product_type TEXT,
                    sku_type TEXT,
                    tag_name TEXT NOT NULL,
                    evidence_type TEXT NOT NULL,
                    matched_keyword TEXT,
                    evidence_text TEXT NOT NULL,
                    evidence_quality TEXT,
                    source_method TEXT,
                    UNIQUE(review_id, product_id, tag_name, evidence_type, evidence_text)
                );
                CREATE INDEX IF NOT EXISTS idx_product_tag_stats_lookup
                    ON product_tag_stats(product_type, tag_name, product_id);
                CREATE INDEX IF NOT EXISTS idx_product_tag_stats_score
                    ON product_tag_stats(tag_name, advantage_support, problem_pressure);
                CREATE INDEX IF NOT EXISTS idx_review_tag_evidence_lookup
                    ON review_tag_evidence(product_id, tag_name, evidence_type);
                CREATE INDEX IF NOT EXISTS idx_memory_events_user_session
                    ON memory_events(user_id, session_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_memory_events_type
                    ON memory_events(user_id, event_type, created_at);
                """
            )

    def ensure_default_user(self, user_id: str = "demo_user", display_name: str = "演示用户") -> dict:
        users = self.list_app_users()
        if users:
            return users[0]
        return self.create_app_user(display_name=display_name, user_id=user_id)

    def create_app_user(self, display_name: str, user_id: str | None = None) -> dict:
        clean_name = (display_name or "").strip() or "新用户"
        clean_user_id = (user_id or "").strip() or f"user_{uuid.uuid4().hex[:10]}"
        with self.connect() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO app_users(user_id, display_name)
                VALUES(?, ?)
                """,
                (clean_user_id, clean_name),
            )
            connection.execute(
                """
                UPDATE app_users
                SET display_name = ?, last_active_at = CURRENT_TIMESTAMP
                WHERE user_id = ?
                """,
                (clean_name, clean_user_id),
            )
        return {"user_id": clean_user_id, "display_name": clean_name}

    def list_app_users(self) -> list[dict]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT user_id, display_name, created_at, last_active_at
                FROM app_users
                ORDER BY last_active_at DESC, created_at DESC
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def get_app_user(self, user_id: str) -> dict | None:
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT user_id, display_name, created_at, last_active_at
                FROM app_users
                WHERE user_id = ?
                """,
                (user_id,),
            ).fetchone()
        return dict(row) if row else None

    def touch_app_user(self, user_id: str) -> None:
        with self.connect() as connection:
            connection.execute(
                "UPDATE app_users SET last_active_at = CURRENT_TIMESTAMP WHERE user_id = ?",
                (user_id,),
            )

    def replace_products(self, products: Iterable[Product]) -> None:
        with self.connect() as connection:
            connection.execute("DELETE FROM products")
            connection.executemany(
                """
                INSERT OR REPLACE INTO products(product_id, title, price, shop_name, sales, source, product_type)
                VALUES(?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        product.product_id,
                        product.title,
                        product.price,
                        product.shop_name,
                        product.sales,
                        product.source,
                        product.product_type,
                    )
                    for product in products
                ],
            )

    def replace_reviews(self, reviews: Iterable[Review]) -> None:
        with self.connect() as connection:
            connection.execute("DELETE FROM reviews")
            connection.executemany(
                """
                INSERT OR IGNORE INTO reviews(product_id, purchase_date, sku_type, review_content, product_type)
                VALUES(?, ?, ?, ?, ?)
                """,
                [
                    (
                        review.product_id,
                        review.purchase_date,
                        review.sku_type,
                        review.review_content,
                        review.product_type,
                    )
                    for review in reviews
                ],
            )

    def replace_product_tag_stats(self, rows: Iterable[ProductTagStats]) -> None:
        with self.connect() as connection:
            connection.execute("DELETE FROM product_tag_stats")
            connection.executemany(
                """
                INSERT OR REPLACE INTO product_tag_stats(
                    product_id,
                    product_type,
                    tag_name,
                    product_review_count,
                    mention_count,
                    advantage_count,
                    problem_count,
                    mixed_count,
                    neutral_count,
                    mention_rate,
                    smoothed_advantage_rate,
                    smoothed_problem_rate,
                    confidence,
                    advantage_support,
                    problem_pressure,
                    source_method
                )
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        row.product_id,
                        row.product_type,
                        row.tag_name,
                        row.product_review_count,
                        row.mention_count,
                        row.advantage_count,
                        row.problem_count,
                        row.mixed_count,
                        row.neutral_count,
                        row.mention_rate,
                        row.smoothed_advantage_rate,
                        row.smoothed_problem_rate,
                        row.confidence,
                        row.advantage_support,
                        row.problem_pressure,
                        row.source_method,
                    )
                    for row in rows
                ],
            )

    def replace_review_tag_evidence(self, rows: Iterable[ReviewTagEvidence]) -> None:
        with self.connect() as connection:
            connection.execute("DELETE FROM review_tag_evidence")
            connection.executemany(
                """
                INSERT OR IGNORE INTO review_tag_evidence(
                    review_id,
                    product_id,
                    product_type,
                    sku_type,
                    tag_name,
                    evidence_type,
                    matched_keyword,
                    evidence_text,
                    evidence_quality,
                    source_method
                )
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        row.review_id,
                        row.product_id,
                        row.product_type,
                        row.sku_type,
                        row.tag_name,
                        row.evidence_type,
                        row.matched_keyword,
                        row.evidence_text,
                        row.evidence_quality,
                        row.source_method,
                    )
                    for row in rows
                ],
            )

    def search_products(
        self,
        product_type: str | None,
        budget_min: float | None,
        budget_max: float | None,
        limit: int = 8,
        sort_mode: str | None = None,
    ) -> list[dict]:
        where: list[str] = []
        params: list[object] = []
        if product_type:
            where.append("product_type = ?")
            params.append(product_type)
        if budget_min is not None:
            where.append("(price IS NULL OR price >= ?)")
            params.append(budget_min)
        if budget_max is not None:
            where.append("(price IS NULL OR price <= ?)")
            params.append(budget_max)
        clause = f"WHERE {' AND '.join(where)}" if where else ""
        order_by = _product_order_by(sort_mode)
        with self.connect() as connection:
            rows = connection.execute(
                f"""
                SELECT product_id, title, price, shop_name, sales, source, product_type
                FROM products
                {clause}
                ORDER BY {order_by}
                LIMIT ?
                """,
                [*params, limit],
            ).fetchall()
        return [dict(row) for row in rows]

    def product_price_range(self, product_type: str | None) -> dict:
        where = "WHERE product_type = ? AND price IS NOT NULL" if product_type else "WHERE price IS NOT NULL"
        params: list[object] = [product_type] if product_type else []
        with self.connect() as connection:
            row = connection.execute(
                f"""
                SELECT
                    MIN(price) AS min_price,
                    MAX(price) AS max_price,
                    AVG(price) AS avg_price,
                    COUNT(*) AS product_count
                FROM products
                {where}
                """,
                params,
            ).fetchone()
        return dict(row) if row else {"min_price": None, "max_price": None, "avg_price": None, "product_count": 0}

    def common_tags_for_product_type(self, product_type: str, limit: int = 8) -> list[dict]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    tag_name,
                    SUM(mention_count) AS mention_count,
                    AVG(mention_rate) AS avg_mention_rate,
                    AVG(confidence) AS avg_confidence,
                    AVG(advantage_support) AS avg_advantage_support,
                    AVG(problem_pressure) AS avg_problem_pressure
                FROM product_tag_stats
                WHERE product_type = ?
                GROUP BY tag_name
                HAVING mention_count > 0
                ORDER BY mention_count DESC, avg_confidence DESC
                LIMIT ?
                """,
                (product_type, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def reviews_for_products(self, product_ids: list[str], limit_per_product: int = 3) -> list[dict]:
        if not product_ids:
            return []
        placeholders = ",".join("?" for _ in product_ids)
        with self.connect() as connection:
            rows = connection.execute(
                f"""
                SELECT product_id, sku_type, review_content, product_type
                FROM reviews
                WHERE product_id IN ({placeholders})
                ORDER BY id ASC
                """,
                product_ids,
            ).fetchall()
        grouped: dict[str, list[dict]] = {}
        for row in rows:
            item = dict(row)
            grouped.setdefault(item["product_id"], [])
            if len(grouped[item["product_id"]]) < limit_per_product:
                grouped[item["product_id"]].append(item)
        return [item for group in grouped.values() for item in group]

    def tag_scores_for_products(self, product_ids: list[str], tag_names: list[str] | None = None) -> list[dict]:
        if not product_ids:
            return []
        where = [f"product_id IN ({','.join('?' for _ in product_ids)})"]
        params: list[object] = list(product_ids)
        if tag_names:
            where.append(f"tag_name IN ({','.join('?' for _ in tag_names)})")
            params.extend(tag_names)
        with self.connect() as connection:
            rows = connection.execute(
                f"""
                SELECT
                    product_id,
                    product_type,
                    tag_name,
                    product_review_count,
                    mention_count,
                    advantage_count,
                    problem_count,
                    mixed_count,
                    neutral_count,
                    mention_rate,
                    smoothed_advantage_rate,
                    smoothed_problem_rate,
                    confidence,
                    advantage_support,
                    problem_pressure,
                    source_method
                FROM product_tag_stats
                WHERE {' AND '.join(where)}
                ORDER BY product_id, tag_name
                """,
                params,
            ).fetchall()
        return [dict(row) for row in rows]

    def evidence_for_products(
        self,
        product_ids: list[str],
        tag_names: list[str] | None = None,
        evidence_types: list[str] | None = None,
        limit_per_product: int = 3,
    ) -> list[dict]:
        if not product_ids:
            return []
        where = [f"product_id IN ({','.join('?' for _ in product_ids)})"]
        params: list[object] = list(product_ids)
        if tag_names:
            where.append(f"tag_name IN ({','.join('?' for _ in tag_names)})")
            params.extend(tag_names)
        if evidence_types:
            where.append(f"evidence_type IN ({','.join('?' for _ in evidence_types)})")
            params.extend(evidence_types)
        with self.connect() as connection:
            rows = connection.execute(
                f"""
                SELECT
                    review_id,
                    product_id,
                    product_type,
                    sku_type,
                    tag_name,
                    evidence_type,
                    matched_keyword,
                    evidence_text,
                    evidence_quality,
                    source_method
                FROM review_tag_evidence
                WHERE {' AND '.join(where)}
                ORDER BY
                    CASE evidence_type
                        WHEN 'advantage' THEN 1
                        WHEN 'problem' THEN 2
                        WHEN 'mixed' THEN 3
                        ELSE 4
                    END,
                    id ASC
                """,
                params,
            ).fetchall()
        grouped: dict[str, list[dict]] = {}
        for row in rows:
            item = dict(row)
            grouped.setdefault(item["product_id"], [])
            if len(grouped[item["product_id"]]) < limit_per_product:
                grouped[item["product_id"]].append(item)
        return [item for group in grouped.values() for item in group]

    def rank_products_by_tag_needs(
        self,
        product_type: str | None,
        budget_min: float | None,
        budget_max: float | None,
        priority_tags: list[str] | None = None,
        avoid_tags: list[str] | None = None,
        limit: int = 5,
    ) -> list[dict]:
        candidates = self.search_products(
            product_type=product_type,
            budget_min=budget_min,
            budget_max=budget_max,
            limit=100,
        )
        if not candidates:
            return []

        priority_tags = priority_tags or []
        avoid_tags = avoid_tags or []
        tag_names = sorted(set(priority_tags + avoid_tags))
        scores = self.tag_scores_for_products(
            [str(item["product_id"]) for item in candidates],
            tag_names=tag_names or None,
        )
        scores_by_product: dict[str, dict[str, dict]] = {}
        for row in scores:
            scores_by_product.setdefault(str(row["product_id"]), {})[row["tag_name"]] = row

        max_sales = max(float(item.get("sales") or 0) for item in candidates) or 1.0
        max_budget = budget_max if budget_max and budget_max > 0 else None
        ranked: list[dict] = []
        for item in candidates:
            product_id = str(item["product_id"])
            product_scores = scores_by_product.get(product_id, {})
            advantage_score = sum(
                float(product_scores.get(tag, {}).get("advantage_support") or 0.0)
                for tag in priority_tags
            )
            problem_score = sum(
                float(product_scores.get(tag, {}).get("problem_pressure") or 0.0)
                for tag in avoid_tags
            )
            sales_score = float(item.get("sales") or 0) / max_sales
            price = item.get("price")
            price_score = 0.0
            if max_budget and isinstance(price, (int, float)):
                price_score = max(0.0, 1.0 - float(price) / max_budget)

            final_score = (
                0.45 * advantage_score
                - 0.35 * problem_score
                + 0.15 * sales_score
                + 0.05 * price_score
            )
            ranked.append(
                {
                    **item,
                    "recommendation_score": round(final_score, 4),
                    "priority_tag_score": round(advantage_score, 4),
                    "avoid_tag_risk": round(problem_score, 4),
                    "sales_score": round(sales_score, 4),
                    "price_score": round(price_score, 4),
                    "tag_scores": list(product_scores.values()),
                }
            )

        ranked.sort(key=lambda row: row["recommendation_score"], reverse=True)
        return ranked[:limit]

    def save_memory(self, user_id: str, key: str, value: object) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO user_memory(user_id, memory_key, memory_value, updated_at)
                VALUES(?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(user_id, memory_key)
                DO UPDATE SET memory_value = excluded.memory_value, updated_at = CURRENT_TIMESTAMP
                """,
                (user_id, key, json.dumps(value, ensure_ascii=False)),
            )

    def append_bounded_memory_list(
        self,
        user_id: str,
        key: str,
        value: object,
        *,
        max_items: int,
        initial_items: list[object] | None = None,
    ) -> tuple[list[object], object | None]:
        if max_items <= 0:
            raise ValueError("max_items must be positive")
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT memory_value FROM user_memory WHERE user_id = ? AND memory_key = ?",
                (user_id, key),
            ).fetchone()
            items: list[object] = list(initial_items or [])
            if row:
                try:
                    stored = json.loads(row["memory_value"])
                except json.JSONDecodeError:
                    stored = []
                if isinstance(stored, list):
                    items = stored
            items.append(value)
            removed = items[0] if len(items) > max_items else None
            items = items[-max_items:]
            connection.execute(
                """
                INSERT INTO user_memory(user_id, memory_key, memory_value, updated_at)
                VALUES(?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(user_id, memory_key)
                DO UPDATE SET memory_value = excluded.memory_value, updated_at = CURRENT_TIMESTAMP
                """,
                (user_id, key, json.dumps(items, ensure_ascii=False)),
            )
        return items, removed

    def load_memory(self, user_id: str) -> dict[str, object]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT memory_key, memory_value FROM user_memory WHERE user_id = ?",
                (user_id,),
            ).fetchall()
        result: dict[str, object] = {}
        for row in rows:
            try:
                result[row["memory_key"]] = json.loads(row["memory_value"])
            except json.JSONDecodeError:
                result[row["memory_key"]] = row["memory_value"]
        return result

    def append_memory_event(
        self,
        user_id: str,
        session_id: str,
        event_type: str,
        content: str,
        payload: object | None = None,
        status: str = "active",
        supersedes_event_id: str | None = None,
        valid_to: str | None = None,
    ) -> str:
        event_id = f"evt_{uuid.uuid4().hex}"
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO memory_events(
                    event_id,
                    user_id,
                    session_id,
                    event_type,
                    content,
                    payload_json,
                    status,
                    supersedes_event_id,
                    valid_to
                )
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    user_id,
                    session_id,
                    event_type,
                    content,
                    json.dumps(payload or {}, ensure_ascii=False),
                    status,
                    supersedes_event_id,
                    valid_to,
                ),
            )
        return event_id

    def list_memory_events(
        self,
        user_id: str,
        session_id: str | None = None,
        event_type: str | None = None,
        limit: int = 50,
    ) -> list[dict]:
        where = ["user_id = ?"]
        params: list[object] = [user_id]
        if session_id:
            where.append("session_id = ?")
            params.append(session_id)
        if event_type:
            where.append("event_type = ?")
            params.append(event_type)
        with self.connect() as connection:
            rows = connection.execute(
                f"""
                SELECT
                    event_id,
                    user_id,
                    session_id,
                    event_type,
                    content,
                    payload_json,
                    status,
                    supersedes_event_id,
                    valid_from,
                    valid_to,
                    created_at
                FROM memory_events
                WHERE {' AND '.join(where)}
                ORDER BY created_at DESC
                LIMIT ?
                """,
                [*params, limit],
            ).fetchall()
        result: list[dict] = []
        for row in rows:
            item = dict(row)
            try:
                item["payload"] = json.loads(item.pop("payload_json"))
            except json.JSONDecodeError:
                item["payload"] = {}
                item.pop("payload_json", None)
            result.append(item)
        return result

    def get_user_profile(self, user_id: str) -> dict:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT profile_json FROM user_profiles WHERE user_id = ?",
                (user_id,),
            ).fetchone()
        if not row:
            return {}
        try:
            return json.loads(row["profile_json"])
        except json.JSONDecodeError:
            return {}

    def save_user_profile(self, user_id: str, profile: object) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO user_profiles(user_id, profile_json, updated_at)
                VALUES(?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(user_id)
                DO UPDATE SET profile_json = excluded.profile_json, updated_at = CURRENT_TIMESTAMP
                """,
                (user_id, json.dumps(profile or {}, ensure_ascii=False)),
            )

    def get_session_summary(self, user_id: str, session_id: str) -> dict:
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT summary_json
                FROM session_summaries
                WHERE user_id = ? AND session_id = ?
                """,
                (user_id, session_id),
            ).fetchone()
        if not row:
            return {}
        try:
            return json.loads(row["summary_json"])
        except json.JSONDecodeError:
            return {}

    def save_session_summary(self, user_id: str, session_id: str, summary: object) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO session_summaries(user_id, session_id, summary_json, updated_at)
                VALUES(?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(user_id, session_id)
                DO UPDATE SET summary_json = excluded.summary_json, updated_at = CURRENT_TIMESTAMP
                """,
                (user_id, session_id, json.dumps(summary or {}, ensure_ascii=False)),
            )

    def append_message(self, user_id: str, session_id: str, role: str, content: str) -> None:
        with self.connect() as connection:
            connection.execute(
                "INSERT INTO conversations(user_id, session_id, role, content) VALUES(?, ?, ?, ?)",
                (user_id, session_id, role, content),
            )
