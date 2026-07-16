from __future__ import annotations

from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    user_id: str
    session_id: str
    thread_id: str
    message: str
    history: list[dict[str, str]]

    query_type: str
    intent: str
    agent_decision: dict[str, Any]

    product_type: str | None
    mentioned_product: str | None
    budget_min: float | None
    budget_max: float | None
    pet_type: str | None
    budget_reference: dict[str, Any]
    budget_validation: str
    priority_tags: list[str]
    avoid_tags: list[str]
    budget_confirmed: bool
    priority_confirmed: bool
    avoid_confirmed: bool

    product_config: dict[str, Any]
    missing_fields: list[str]
    requirement_status: str
    required_action: str | None
    recommendation_request: dict[str, Any]

    tool_plan: list[dict[str, Any]]
    tool_results: dict[str, Any]
    evidence_bundle: dict[str, Any]
    candidate_products: list[dict[str, Any]]
    review_evidence: list[dict[str, Any]]

    user_memory: dict[str, Any]

    response: dict[str, Any]
