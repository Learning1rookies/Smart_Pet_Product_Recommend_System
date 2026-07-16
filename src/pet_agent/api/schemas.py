from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class FrontendSelectionRequest(BaseModel):
    """Structured filters sent by the UI before the user message reaches the Agent."""

    product_type: str | None = Field(default=None, description="Selected smart pet product type.")
    budget_min: float | None = Field(default=None, description="Minimum acceptable price.")
    budget_max: float | None = Field(default=None, description="Maximum acceptable price.")
    pet_type: str | None = Field(default=None, description="Pet type, such as cat or dog.")
    priority_tags: list[str] = Field(default_factory=list, description="User preferred feature tags.")
    avoid_tags: list[str] = Field(default_factory=list, description="Problems the user wants to avoid.")
    budget_confirmed: bool = Field(default=False, description="Whether budget has been confirmed or intentionally skipped.")
    priority_confirmed: bool = Field(default=False, description="Whether priority tags have been confirmed or defaulted.")
    avoid_confirmed: bool = Field(default=False, description="Whether avoid tags have been confirmed or defaulted.")


class ChatRequest(BaseModel):
    """Request body for the Agent chat endpoint."""

    user_id: str = Field(..., description="User id used for memory isolation.")
    session_id: str = Field(..., description="Session id used for conversation isolation.")
    message: str = Field(..., description="Current user message.")
    frontend_selection: FrontendSelectionRequest | None = Field(
        default=None,
        description="Optional structured selections from the frontend.",
    )
    history: list[dict[str, str]] = Field(
        default_factory=list,
        description="Optional conversation history passed by the caller.",
    )


class ChatResponse(BaseModel):
    """Stable response body returned by the Agent chat endpoint."""

    user_id: str
    session_id: str
    answer: str
    intent: str | None = None
    product_type: str | None = None
    missing_fields: list[str] = Field(default_factory=list)
    requirement_status: str | None = None
    requirement_context: dict[str, Any] = Field(default_factory=dict)
    required_action: str | None = None
    action_options: list[Any] = Field(default_factory=list)
    recommended_products: list[dict[str, Any]] = Field(default_factory=list)
    comparison_table: list[dict[str, Any]] = Field(default_factory=list)
    review_evidence: list[dict[str, Any]] = Field(default_factory=list)
    risk_notes: list[str] = Field(default_factory=list)
    raw_response: dict[str, Any] = Field(default_factory=dict)
