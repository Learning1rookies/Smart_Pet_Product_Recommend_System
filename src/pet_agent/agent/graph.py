from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

try:
    from langgraph.graph import END, StateGraph
except ImportError as exc:  # pragma: no cover - dependency guard
    raise RuntimeError("langgraph is required. Install langgraph before starting the Agent.") from exc

from pet_agent.agent.middleware import log_node
from pet_agent.agent.nodes.classify import classify_query, direct_answer, unsupported_product_direct_answer
from pet_agent.agent.nodes.recommendation import final_response, generate_recommendation, retrieve_evidence
from pet_agent.agent.nodes.requirements import (
    ask_avoid_tags,
    ask_budget,
    ask_priority_tags,
    check_missing_fields,
    extract_requirement,
    guide_product_type,
    load_product_config,
    requirement_complete,
)
from pet_agent.agent.state import AgentState
from pet_agent.agent.tools.langchain_tools import build_product_tools
from pet_agent.memory.user_memory import UserMemory
from pet_agent.model.factory import preload_chat_model
from pet_agent.retrieval.reranker import EvidenceRerankConfig, EvidenceRerankPipeline, LazyBgeReranker
from pet_agent.storage.sqlite_store import SQLiteStore
from pet_agent.storage.vector_store import VectorStore
from pet_agent.utils.config_loader import load_agent_config, load_chroma_config, load_rerank_config
from pet_agent.utils.logger_handler import logger


class PetRecommendationAgent:
    """Graph-orchestrated customer-service Agent for smart pet product decisions."""

    def __init__(
        self,
        sqlite_store: SQLiteStore,
        vector_store: VectorStore,
        evidence_reranker: EvidenceRerankPipeline | None = None,
    ):
        self.sqlite_store = sqlite_store
        self.vector_store = vector_store
        self.user_memory = UserMemory(sqlite_store, vector_store)
        self.agent_conf = load_agent_config()
        self.chroma_conf = load_chroma_config()
        self.checkpointer = InMemorySaver()
        self.evidence_reranker = evidence_reranker or self._build_evidence_reranker()
        self.runtime_tools = build_product_tools(
            sqlite_store=sqlite_store,
            vector_store=vector_store,
            evidence_reranker=self.evidence_reranker,
            global_user_id=self.chroma_conf.get("global_user_id", "global"),
        )
        # The local Streamlit process owns this checkpointer. LangGraph Studio
        # provides its own persistence, so it compiles the same topology without one.
        self.compiled_graph = self._build_graph(with_checkpointer=True)

    def invoke(self, state: AgentState) -> AgentState:
        state = self._prepare_input(state)
        return self.compiled_graph.invoke(
            state,
            config=self._graph_config(state["user_id"], state["session_id"]),
        )

    def resume(self, *, user_id: str, session_id: str, resume_value: dict[str, Any]) -> AgentState:
        return self.compiled_graph.invoke(
            Command(resume=resume_value),
            config=self._graph_config(user_id, session_id),
        )

    def get_thread_values(self, *, user_id: str, session_id: str) -> dict[str, Any]:
        snapshot = self.compiled_graph.get_state(self._graph_config(user_id, session_id))
        return dict(snapshot.values or {})

    def available_tools(self) -> list[Any]:
        return self.runtime_tools

    def confirm_purchase(
        self,
        *,
        user_id: str,
        session_id: str,
        product: dict[str, Any],
        requirement_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        product_id = str(product.get("product_id") or "").strip()
        title = str(product.get("title") or "").strip()
        product_type = str(product.get("product_type") or "").strip()
        if not product_id or not title or not product_type:
            raise ValueError("Confirmed purchase requires product_id, title, and product_type.")

        context = requirement_context or {}
        memory_value = {
            "product_id": product_id,
            "title": title,
            "product_type": product_type,
            "price": product.get("price"),
            "shop_name": product.get("shop_name"),
            "sales": product.get("sales"),
            "evidence_status": product.get("evidence_status"),
            "budget_min": context.get("budget_min"),
            "budget_max": context.get("budget_max"),
            "priority_tags": context.get("priority_tags") or [],
            "avoid_tags": context.get("avoid_tags") or [],
            "confirmed_at": datetime.now(timezone.utc).isoformat(),
            "source_session_id": session_id,
        }
        return self.user_memory.confirm_purchase(user_id, memory_value, max_items=10)

    def preload_models(self, *, include_reranker: bool = False) -> None:
        """Warm reusable model objects before serving the first user request."""
        preload_chat_model()
        if include_reranker:
            self.evidence_reranker.preload()

    def graph_mermaid(self) -> str:
        if hasattr(self.compiled_graph, "get_graph"):
            try:
                return self.compiled_graph.get_graph().draw_mermaid()
            except Exception:
                pass
        return GRAPH_MERMAID

    def build_studio_graph(self) -> Any:
        """Compile the production topology for LangGraph Studio persistence."""
        return self._build_graph(with_checkpointer=False)

    def _build_graph(self, *, with_checkpointer: bool) -> Any:
        graph = StateGraph(AgentState)
        graph.add_node("classify_query", self._classify_query)
        graph.add_node("direct_answer", self._direct_answer)
        graph.add_node("unsupported_product_direct_answer", self._unsupported_product_direct_answer)
        graph.add_node("extract_requirement", self._extract_requirement)
        graph.add_node("guide_product_type", self._guide_product_type)
        graph.add_node("load_product_config", self._load_product_config)
        graph.add_node("check_missing_fields", self._check_missing_fields)
        graph.add_node("ask_budget", self._ask_budget)
        graph.add_node("ask_priority_tags", self._ask_priority_tags)
        graph.add_node("ask_avoid_tags", self._ask_avoid_tags)
        graph.add_node("requirement_complete", self._requirement_complete)
        graph.add_node("retrieve_evidence", self._retrieve_evidence)
        graph.add_node("generate_recommendation", self._generate_recommendation)
        graph.add_node("final_response", self._final_response)

        graph.set_entry_point("classify_query")
        graph.add_conditional_edges(
            "classify_query",
            self._route_after_classify,
            {
                "product": "extract_requirement",
                "direct": "direct_answer",
                "unsupported_product": "unsupported_product_direct_answer",
            },
        )
        graph.add_edge("direct_answer", "final_response")
        graph.add_edge("unsupported_product_direct_answer", "final_response")
        graph.add_conditional_edges(
            "extract_requirement",
            self._route_after_extract_requirement,
            {
                "has_product_type": "load_product_config",
                "missing_product_type": "guide_product_type",
            },
        )
        graph.add_edge("guide_product_type", "load_product_config")
        graph.add_edge("load_product_config", "check_missing_fields")
        graph.add_conditional_edges(
            "check_missing_fields",
            self._route_after_missing_check,
            {
                "ask_budget": "ask_budget",
                "ask_priority_tags": "ask_priority_tags",
                "ask_avoid_tags": "ask_avoid_tags",
                "complete": "requirement_complete",
            },
        )
        graph.add_edge("ask_budget", "check_missing_fields")
        graph.add_edge("ask_priority_tags", "check_missing_fields")
        graph.add_edge("ask_avoid_tags", "check_missing_fields")
        graph.add_edge("requirement_complete", "retrieve_evidence")
        graph.add_edge("retrieve_evidence", "generate_recommendation")
        graph.add_edge("generate_recommendation", "final_response")
        graph.add_edge("final_response", END)
        if with_checkpointer:
            return graph.compile(checkpointer=self.checkpointer)
        return graph.compile()

    def _graph_config(self, user_id: str, session_id: str) -> dict[str, Any]:
        return {
            "configurable": {
                "thread_id": f"{user_id}:{session_id}",
            },
            "run_name": "smart_pet_recommendation_graph",
            "tags": ["smart_pet_agent", user_id, session_id],
            "metadata": {
                "user_id": user_id,
                "session_id": session_id,
            },
        }

    def _build_evidence_reranker(self) -> EvidenceRerankPipeline:
        rerank_conf = load_rerank_config()
        provider = str(rerank_conf.get("provider") or "bge").lower()
        if provider != "bge":
            raise RuntimeError(f"Unsupported rerank provider: {provider}")
        config = EvidenceRerankConfig(
            per_product_recall_limit=int(rerank_conf.get("per_product_recall_limit", 20)),
            rule_keep_limit=int(rerank_conf.get("rule_keep_limit", 20)),
            final_limit=int(rerank_conf.get("final_limit", 5)),
            duplicate_threshold=float(rerank_conf.get("duplicate_threshold", 0.88)),
            semantic_weight=float(rerank_conf.get("semantic_weight", 0.45)),
            priority_weight=float(rerank_conf.get("priority_weight", 0.25)),
            product_weight=float(rerank_conf.get("product_weight", 0.2)),
            avoid_weight=float(rerank_conf.get("avoid_weight", 0.25)),
            insufficient_evidence_penalty=float(rerank_conf.get("insufficient_evidence_penalty", 0.2)),
        )
        bge = LazyBgeReranker(
            model_name=str(rerank_conf.get("model_name") or "BAAI/bge-reranker-base"),
            use_fp16=bool(rerank_conf.get("use_fp16", False)),
            local_files_only=bool(rerank_conf.get("local_files_only", True)),
        )
        return EvidenceRerankPipeline(reranker=bge, config=config)

    def _prepare_input(self, state: AgentState) -> AgentState:
        logger.info(
            "[agent invoke] user_id=%s session_id=%s message=%s",
            state["user_id"],
            state["session_id"],
            state.get("message", ""),
        )
        state["history"] = self._recent_history(state)
        state["user_memory"] = self.user_memory.load(state["user_id"])
        state.setdefault("tool_plan", [])
        state.setdefault("tool_results", {})
        return state

    def _recent_history(self, state: AgentState) -> list[dict[str, str]]:
        history = state.get("history") or []
        limit = int(self.agent_conf.get("history_limit", 10))
        return history[-limit:]

    def _route_after_classify(self, state: AgentState) -> str:
        query_type = state.get("query_type")
        if query_type == "product_recommend":
            return "product"
        if query_type == "unsupported_product_direct_answer":
            return "unsupported_product"
        return "direct"

    def _route_after_extract_requirement(self, state: AgentState) -> str:
        return "has_product_type" if state.get("product_type") else "missing_product_type"

    def _route_after_missing_check(self, state: AgentState) -> str:
        missing = state.get("missing_fields", [])
        if "预算" in missing:
            return "ask_budget"
        if "核心关注点" in missing:
            return "ask_priority_tags"
        if "避免项" in missing:
            return "ask_avoid_tags"
        return "complete"

    @log_node("classify_query")
    def _classify_query(self, state: AgentState) -> AgentState:
        return classify_query(state, self.agent_conf)

    @log_node("direct_answer")
    def _direct_answer(self, state: AgentState) -> AgentState:
        return direct_answer(state)

    @log_node("unsupported_product_direct_answer")
    def _unsupported_product_direct_answer(self, state: AgentState) -> AgentState:
        return unsupported_product_direct_answer(state)

    @log_node("extract_requirement")
    def _extract_requirement(self, state: AgentState) -> AgentState:
        return extract_requirement(state, self.agent_conf)

    @log_node("guide_product_type")
    def _guide_product_type(self, state: AgentState) -> AgentState:
        return guide_product_type(state)

    @log_node("load_product_config")
    def _load_product_config(self, state: AgentState) -> AgentState:
        return load_product_config(state, self.runtime_tools)

    @log_node("check_missing_fields")
    def _check_missing_fields(self, state: AgentState) -> AgentState:
        return check_missing_fields(state)

    @log_node("ask_budget")
    def _ask_budget(self, state: AgentState) -> AgentState:
        return ask_budget(state)

    @log_node("ask_priority_tags")
    def _ask_priority_tags(self, state: AgentState) -> AgentState:
        return ask_priority_tags(state)

    @log_node("ask_avoid_tags")
    def _ask_avoid_tags(self, state: AgentState) -> AgentState:
        return ask_avoid_tags(state)

    @log_node("requirement_complete")
    def _requirement_complete(self, state: AgentState) -> AgentState:
        return requirement_complete(state)

    @log_node("retrieve_evidence")
    def _retrieve_evidence(self, state: AgentState) -> AgentState:
        return retrieve_evidence(state, self.runtime_tools, self.agent_conf)

    @log_node("generate_recommendation")
    def _generate_recommendation(self, state: AgentState) -> AgentState:
        return generate_recommendation(state, self.agent_conf)

    @log_node("final_response")
    def _final_response(self, state: AgentState) -> AgentState:
        return final_response(state)

GRAPH_MERMAID = """flowchart TD
    A([用户输入]) --> B[classify_query]
    B --> C{是否商品推荐相关?}
    C -- 否 --> D[direct_answer]
    D --> P[final_response]
    C -- 不支持的产品品类 --> X[unsupported_product_direct_answer]
    X --> P
    C -- 是 --> E[extract_requirement]
    E --> F{是否明确商品品类?}
    F -- 否 --> G[guide_product_type]
    G --> LC
    F -- 是 --> LC[load_product_config]
    LC --> I[check_missing_fields]
    I --> J{缺预算?}
    J -- 是 --> J1[ask_budget]
    J -- 否 --> K{缺关注点?}
    J1 --> I
    K -- 是 --> K1[ask_priority_tags]
    K -- 否 --> L{缺避免项?}
    K1 --> I
    L -- 是 --> L1[ask_avoid_tags]
    L -- 否 --> M[requirement_complete]
    L1 --> I
    M --> N[retrieve_evidence]
    N --> O[generate_recommendation]
    O --> P
    P --> Z([结束])
"""
