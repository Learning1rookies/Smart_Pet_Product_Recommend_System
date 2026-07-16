from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ResponsePayload:
    """Agent response payload used by Streamlit and FastAPI."""

    recommended_products: list[dict[str, Any]]
    comparison_table: list[dict[str, Any]]
    recommendation_reason: str
    review_evidence: list[dict[str, Any]]
    risk_notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "recommended_products": self.recommended_products,
            "comparison_table": self.comparison_table,
            "recommendation_reason": self.recommendation_reason,
            "review_evidence": self.review_evidence,
            "risk_notes": self.risk_notes,
        }
