from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from pet_agent.data.schemas import Product, Review
from pet_agent.model.factory import get_embedding_model
from pet_agent.storage.hash_store import ContentHashStore
from pet_agent.utils.config_loader import load_chroma_config
from pet_agent.utils.content_hash import md5_json


class VectorStore:
    """ChromaDB vector store backed by the configured embedding model.

    This class intentionally has no JSON/hash alternative path. If ChromaDB or the
    embedding model is unavailable, construction fails so configuration issues
    are visible during startup.
    """

    def __init__(
        self,
        path: Path,
        collection_name: str | None = None,
        embedding_model: Any | None = None,
    ):
        chroma_conf = load_chroma_config()
        self.path = path
        self.collection_name = collection_name or chroma_conf.get("collection_name", "smart_pet_products")
        self.embedding_model = embedding_model or get_embedding_model()
        if self.embedding_model is None:
            raise RuntimeError("Embedding model is not configured. Set MODEL_PROVIDER and embedding model API credentials.")
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError as exc:
            raise RuntimeError("chromadb is required for VectorStore. Install chromadb before building the knowledge base.") from exc

        self.path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(path), settings=Settings(anonymized_telemetry=False))
        self._collection = self._client.get_or_create_collection(
            self.collection_name,
            metadata={
                "hnsw:space": chroma_conf.get("hnsw_space", "cosine"),
                "hnsw:batch_size": int(chroma_conf.get("hnsw_batch_size", 50000)),
                "hnsw:sync_threshold": int(chroma_conf.get("hnsw_sync_threshold", 50000)),
            },
        )

    @property
    def using_chroma(self) -> bool:
        return True

    def reset(self) -> None:
        try:
            self._client.delete_collection(self.collection_name)
        except Exception:
            pass
        chroma_conf = load_chroma_config()
        self._collection = self._client.get_or_create_collection(
            self.collection_name,
            metadata={
                "hnsw:space": chroma_conf.get("hnsw_space", "cosine"),
                "hnsw:batch_size": int(chroma_conf.get("hnsw_batch_size", 50000)),
                "hnsw:sync_threshold": int(chroma_conf.get("hnsw_sync_threshold", 50000)),
            },
        )

    def add_texts(self, ids: list[str], texts: list[str], metadatas: list[dict], batch_size: int = 128) -> None:
        existing = set()
        try:
            existing = set(self._collection.get(ids=ids).get("ids", []))
        except Exception:
            existing = set()

        new_ids = []
        new_texts = []
        new_metadatas = []
        for doc_id, text, metadata in zip(ids, texts, metadatas):
            if doc_id in existing:
                continue
            new_ids.append(doc_id)
            new_texts.append(text)
            new_metadatas.append(metadata)

        if new_ids:
            for start in range(0, len(new_ids), batch_size):
                batch_ids = new_ids[start : start + batch_size]
                batch_texts = new_texts[start : start + batch_size]
                batch_metadatas = new_metadatas[start : start + batch_size]
                self._collection.add(
                    ids=batch_ids,
                    documents=batch_texts,
                    metadatas=batch_metadatas,
                    embeddings=self._embed_documents(batch_texts),
                )

    def upsert_texts(self, ids: list[str], texts: list[str], metadatas: list[dict], batch_size: int = 128) -> None:
        for start in range(0, len(ids), batch_size):
            batch_ids = ids[start : start + batch_size]
            batch_texts = texts[start : start + batch_size]
            batch_metadatas = metadatas[start : start + batch_size]
            self._collection.upsert(
                ids=batch_ids,
                documents=batch_texts,
                metadatas=batch_metadatas,
                embeddings=self._embed_documents(batch_texts),
            )

    def _chroma_where(self, where: dict | None) -> dict | None:
        if not where or len(where) <= 1:
            return where
        return {"$and": [{key: value} for key, value in where.items()]}

    def query(self, text: str, where: dict | None = None, limit: int = 5) -> list[dict]:
        result = self._collection.query(
            query_embeddings=[self._embed_query(text)],
            where=self._chroma_where(where),
            n_results=limit,
        )
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        ids = result.get("ids", [[]])[0]
        distances = result.get("distances", [[]])[0] if result.get("distances") else [None] * len(documents)
        return [
            {"id": doc_id, "document": document, "metadata": metadata, "score": distance}
            for doc_id, document, metadata, distance in zip(ids, documents, metadatas, distances)
        ]

    def _embed_documents(self, texts: list[str]) -> list[list[float]]:
        if hasattr(self.embedding_model, "embed_documents"):
            return self.embedding_model.embed_documents(texts)
        if callable(self.embedding_model):
            return self.embedding_model(texts)
        raise RuntimeError("Embedding model must implement embed_documents(texts).")

    def _embed_query(self, text: str) -> list[float]:
        if hasattr(self.embedding_model, "embed_query"):
            return self.embedding_model.embed_query(text)
        return self._embed_documents([text])[0]


def product_summary(product: Product, reviews: Iterable[Review]) -> str:
    review_texts = "；".join(review.review_content for review in list(reviews)[:5])
    return (
        f"商品：{product.title}。类型：{product.product_type}。价格：{product.price}。"
        f"店铺：{product.shop_name}。销量：{product.sales}。评论摘要：{review_texts}"
    )


def index_knowledge(
    products: list[Product],
    reviews: list[Review],
    vector_store: VectorStore,
    hash_store: ContentHashStore | None = None,
    skip_seen: bool = False,
) -> dict[str, int]:
    reviews_by_product: dict[str, list[Review]] = {}
    for review in reviews:
        reviews_by_product.setdefault(review.product_id, []).append(review)

    ids: list[str] = []
    texts: list[str] = []
    metadatas: list[dict] = []
    digests: list[str] = []
    skipped = 0
    for product in products:
        metadata = {
            "doc_type": "product_summary",
            "product_id": product.product_id,
            "product_type": product.product_type,
            "user_id": "global",
        }
        text = product_summary(product, reviews_by_product.get(product.product_id, []))
        digest = md5_json({"text": text, "metadata": metadata})
        if skip_seen and hash_store and hash_store.seen(digest):
            skipped += 1
            continue
        ids.append(f"product:{product.product_id}:{digest[:12]}")
        texts.append(text)
        metadatas.append(metadata | {"content_md5": digest})
        digests.append(digest)
    for index, review in enumerate(reviews):
        metadata = {
            "doc_type": "review",
            "product_id": review.product_id,
            "product_type": review.product_type,
            "user_id": "global",
        }
        digest = md5_json({"text": review.review_content, "metadata": metadata})
        if skip_seen and hash_store and hash_store.seen(digest):
            skipped += 1
            continue
        ids.append(f"review:{review.product_id}:{index}:{digest[:12]}")
        texts.append(review.review_content)
        metadatas.append(metadata | {"content_md5": digest})
        digests.append(digest)
    if ids:
        vector_store.add_texts(ids, texts, metadatas)
    if hash_store:
        hash_store.append_many(digests)
    return {"added": len(ids), "skipped": skipped}
