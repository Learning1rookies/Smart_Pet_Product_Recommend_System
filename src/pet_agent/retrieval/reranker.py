from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Protocol

from pet_agent.agent.tools.product_tools import TAG_KEYWORDS
from pet_agent.utils.path_tool import get_abs_path


class Reranker(Protocol):
    def rerank(self, query: str, documents: list[str]) -> list[float]:
        """Return one relevance score per document. Higher is better."""


@dataclass(frozen=True)
class EvidenceRerankConfig:
    per_product_recall_limit: int = 20
    rule_keep_limit: int = 20
    final_limit: int = 5
    duplicate_threshold: float = 0.88
    semantic_weight: float = 0.45
    priority_weight: float = 0.25
    product_weight: float = 0.20
    avoid_weight: float = 0.25
    insufficient_evidence_penalty: float = 0.20


class BgeReranker:
    def __init__(self, model_name: str, use_fp16: bool = False, local_files_only: bool = True):
        model_path = _resolve_local_model_path(model_name)
        if model_path:
            model_name = str(model_path)
            local_files_only = False
        elif local_files_only:
            try:
                from huggingface_hub import snapshot_download

                model_name = snapshot_download(model_name, local_files_only=True)
            except Exception as exc:
                raise RuntimeError(
                    "BGE reranker model is not available in the local HuggingFace cache. "
                    f"Download it first or configure a local model path. model={model_name}"
                ) from exc
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("BGE reranker requires torch and transformers.") from exc
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        if use_fp16:
            self.model = self.model.half()
        self.model.eval()

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        scores: list[float] = []
        with self.torch.no_grad():
            for start in range(0, len(documents), 8):
                batch_docs = documents[start : start + 8]
                inputs = self.tokenizer(
                    [query] * len(batch_docs),
                    batch_docs,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )
                logits = self.model(**inputs).logits.reshape(-1)
                scores.extend(self.torch.sigmoid(logits).tolist())
        return [float(score) for score in scores]


class LazyBgeReranker:
    def __init__(self, model_name: str, use_fp16: bool = False, local_files_only: bool = True):
        self.model_name = model_name
        self.use_fp16 = use_fp16
        self.local_files_only = local_files_only
        self._delegate: BgeReranker | None = None
        self._load_lock = Lock()

    def preload(self) -> None:
        self._get_delegate()

    def _get_delegate(self) -> BgeReranker:
        if self._delegate is None:
            with self._load_lock:
                if self._delegate is None:
                    self._delegate = BgeReranker(
                        self.model_name,
                        use_fp16=self.use_fp16,
                        local_files_only=self.local_files_only,
                    )
        return self._delegate

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        return self._get_delegate().rerank(query, documents)


class EvidenceRerankPipeline:
    def __init__(
        self,
        reranker: Reranker,
        config: EvidenceRerankConfig | None = None,
    ):
        self.reranker = reranker
        self.config = config or EvidenceRerankConfig()

    def rerank(
        self,
        query: str,
        evidence: list[dict[str, Any]],
        products: list[dict[str, Any]],
        priority_tags: list[str] | None = None,
        avoid_tags: list[str] | None = None,
        final_limit: int | None = None,
    ) -> list[dict[str, Any]]:
        if not evidence:
            return []
        priority_tags = priority_tags or []
        avoid_tags = avoid_tags or []
        product_scores = _product_scores(products)

        rule_ranked = self._rule_rerank(
            evidence=evidence,
            product_scores=product_scores,
            priority_tags=priority_tags,
            avoid_tags=avoid_tags,
        )
        deduped = _dedupe_by_text(rule_ranked, threshold=self.config.duplicate_threshold)
        kept = _select_with_product_coverage(
            deduped,
            products=products,
            limit=self.config.rule_keep_limit,
            score_key="rule_rerank_score",
        )
        bge_scores = self.reranker.rerank(query, [str(item.get("document") or "") for item in kept])
        enriched: list[dict[str, Any]] = []
        for item, bge_score in zip(kept, bge_scores):
            enriched.append(
                {
                    **item,
                    "bge_rerank_score": round(float(bge_score), 6),
                    "final_evidence_score": round(
                        0.55 * float(bge_score) + 0.45 * float(item.get("rule_rerank_score") or 0.0),
                        6,
                    ),
                    "recall_stage": "bge_rerank",
                }
            )
        enriched.sort(key=lambda item: float(item.get("final_evidence_score") or 0.0), reverse=True)
        return _select_with_product_coverage(
            enriched,
            products=products,
            limit=final_limit or self.config.final_limit,
            score_key="final_evidence_score",
        )

    def preload(self) -> None:
        preload = getattr(self.reranker, "preload", None)
        if callable(preload):
            preload()

    def _rule_rerank(
        self,
        evidence: list[dict[str, Any]],
        product_scores: dict[str, float],
        priority_tags: list[str],
        avoid_tags: list[str],
    ) -> list[dict[str, Any]]:
        ranked = []
        for item in evidence:
            document = str(item.get("document") or "")
            product_id = str((item.get("metadata") or {}).get("product_id") or "")
            semantic_score = _semantic_score(item.get("score"))
            priority_score = _tag_hit_score(document, priority_tags)
            avoid_score = _tag_hit_score(document, avoid_tags)
            product_score = product_scores.get(product_id, 0.0)
            rule_score = (
                self.config.semantic_weight * semantic_score
                + self.config.priority_weight * priority_score
                + self.config.product_weight * product_score
                - self.config.avoid_weight * avoid_score
            )
            ranked.append(
                {
                    **item,
                    "semantic_score": round(semantic_score, 6),
                    "priority_hit_score": round(priority_score, 6),
                    "avoid_hit_score": round(avoid_score, 6),
                    "product_rank_score": round(product_score, 6),
                    "rule_rerank_score": round(rule_score, 6),
                    "recall_stage": "rule_rerank",
                }
            )
        ranked.sort(key=lambda row: float(row.get("rule_rerank_score") or 0.0), reverse=True)
        return ranked


def _product_scores(products: list[dict[str, Any]]) -> dict[str, float]:
    if not products:
        return {}
    raw_scores = [
        float(item.get("recommendation_score") or item.get("sales_score") or item.get("sales") or 0.0)
        for item in products
    ]
    max_score = max(raw_scores) or 1.0
    scores: dict[str, float] = {}
    for item, raw_score in zip(products, raw_scores):
        scores[str(item.get("product_id"))] = max(0.0, raw_score / max_score)
    return scores


def _select_with_product_coverage(
    ranked: list[dict[str, Any]],
    *,
    products: list[dict[str, Any]],
    limit: int,
    score_key: str,
) -> list[dict[str, Any]]:
    """Keep the strongest evidence while reserving one slot per represented product."""
    if limit <= 0 or not ranked:
        return []

    product_ids = {str(item.get("product_id") or "") for item in products}
    selected: list[dict[str, Any]] = []
    selected_object_ids: set[int] = set()
    covered_products: set[str] = set()

    for item in ranked:
        product_id = str((item.get("metadata") or {}).get("product_id") or "")
        if product_id not in product_ids or product_id in covered_products:
            continue
        selected.append(item)
        selected_object_ids.add(id(item))
        covered_products.add(product_id)
        if len(selected) >= limit:
            break

    if len(selected) < limit:
        for item in ranked:
            if id(item) in selected_object_ids:
                continue
            selected.append(item)
            if len(selected) >= limit:
                break

    selected.sort(key=lambda item: float(item.get(score_key) or 0.0), reverse=True)
    return selected


def _semantic_score(distance: object) -> float:
    try:
        value = float(distance)
    except (TypeError, ValueError):
        return 0.0
    return 1.0 / (1.0 + max(0.0, value))


def _tag_hit_score(text: str, tags: list[str]) -> float:
    if not tags:
        return 0.0
    hit_count = 0
    for tag in tags:
        keywords = TAG_KEYWORDS.get(tag, ())
        if any(keyword.lower() in text.lower() for keyword in keywords):
            hit_count += 1
    return hit_count / max(1, len(tags))


def _dedupe_by_text(evidence: list[dict[str, Any]], threshold: float) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    fingerprints: list[set[str]] = []
    for item in evidence:
        tokens = _text_fingerprint(str(item.get("document") or ""))
        if not tokens:
            continue
        if any(_jaccard(tokens, existing) >= threshold for existing in fingerprints):
            continue
        kept.append(item)
        fingerprints.append(tokens)
    return kept


def _text_fingerprint(text: str) -> set[str]:
    compact = re.sub(r"\s+", "", text.lower())
    if not compact:
        return set()
    if len(compact) <= 3:
        return {compact}
    return {compact[index : index + 3] for index in range(len(compact) - 2)}


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _resolve_local_model_path(model_name: str) -> Path | None:
    candidate = Path(model_name)
    if candidate.is_absolute() and candidate.exists():
        return candidate
    project_candidate = get_abs_path(candidate)
    if project_candidate.exists():
        return project_candidate
    return None
