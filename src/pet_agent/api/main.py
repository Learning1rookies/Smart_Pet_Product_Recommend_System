from __future__ import annotations

import os
from pathlib import Path
from threading import Thread
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from pet_agent.api.dependencies import get_agent
from pet_agent.api.schemas import ChatRequest, ChatResponse
from pet_agent.config import Settings


app = FastAPI(title="Smart Pet Product Recommend Agent API")
settings = Settings.from_env()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.api_cors_origins or [],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _preload_api_models() -> None:
    try:
        include_reranker = os.getenv("API_PRELOAD_RERANKER", "false").strip().lower() in {"1", "true", "yes"}
        get_agent().preload_models(include_reranker=include_reranker)
    except Exception:
        # API requests retain normal error reporting if an optional warmup fails.
        return


@app.on_event("startup")
def preload_models_on_startup() -> None:
    Thread(target=_preload_api_models, daemon=True, name="pet-agent-api-preload").start()


def _model_to_dict(model: Any) -> dict[str, Any]:
    """Support both Pydantic v1 and v2 model dict conversion."""
    if model is None:
        return {}
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def _build_agent_state(request: ChatRequest) -> dict[str, Any]:
    selection = request.frontend_selection
    selection_data = _model_to_dict(selection)

    return {
        "user_id": request.user_id,
        "session_id": request.session_id,
        "thread_id": f"{request.user_id}:{request.session_id}",
        "message": request.message,
        "history": request.history,
        "product_type": selection_data.get("product_type"),
        "budget_min": selection_data.get("budget_min"),
        "budget_max": selection_data.get("budget_max"),
        "pet_type": selection_data.get("pet_type"),
        "priority_tags": selection_data.get("priority_tags", []),
        "avoid_tags": selection_data.get("avoid_tags", []),
        "budget_confirmed": bool(selection_data.get("budget_confirmed")),
        "priority_confirmed": bool(selection_data.get("priority_confirmed")),
        "avoid_confirmed": bool(selection_data.get("avoid_confirmed")),
        "selected_filters": selection_data,
        "frontend_context": selection_data,
    }


def _build_chat_response(request: ChatRequest, result: dict[str, Any]) -> ChatResponse:
    response = result.get("response", {}) or {}
    answer = response.get("recommendation_reason") or response.get("answer") or ""
    requirement_context = {
        "product_type": result.get("product_type"),
        "budget_min": result.get("budget_min"),
        "budget_max": result.get("budget_max"),
        "pet_type": result.get("pet_type"),
        "priority_tags": result.get("priority_tags") or [],
        "avoid_tags": result.get("avoid_tags") or [],
        "budget_confirmed": bool(result.get("budget_confirmed")),
        "priority_confirmed": bool(result.get("priority_confirmed")),
        "avoid_confirmed": bool(result.get("avoid_confirmed")),
    }

    return ChatResponse(
        user_id=request.user_id,
        session_id=request.session_id,
        answer=answer,
        intent=result.get("intent"),
        product_type=result.get("product_type"),
        missing_fields=result.get("missing_fields", []),
        requirement_status=result.get("requirement_status"),
        requirement_context={
            key: value
            for key, value in requirement_context.items()
            if value not in (None, [], "")
        },
        required_action=response.get("required_action") or result.get("required_action"),
        action_options=response.get("action_options", []),
        recommended_products=response.get("recommended_products", []),
        comparison_table=response.get("comparison_table", []),
        review_evidence=response.get("review_evidence", []),
        risk_notes=response.get("risk_notes", []),
        raw_response=response,
    )


@app.get("/api/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api-docs", include_in_schema=False)
def api_reference() -> FileResponse:
    return FileResponse(Path(__file__).resolve().parent / "static" / "api_reference.html")


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Call the existing Agent through a stable HTTP JSON interface."""
    try:
        agent = get_agent()
        result = agent.invoke(_build_agent_state(request))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Agent call failed: {exc}") from exc

    return _build_chat_response(request=request, result=result)
