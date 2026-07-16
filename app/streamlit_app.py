from __future__ import annotations

import sys
import time
import uuid
from html import escape
from threading import Thread
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import streamlit as st
import streamlit.components.v1 as components

from pet_agent.agent.graph import PetRecommendationAgent
from pet_agent.config import Settings
from pet_agent.storage.sqlite_store import SQLiteStore
from pet_agent.storage.vector_store import VectorStore
from pet_agent.utils.logger_handler import logger

AGENT_RUNTIME_VERSION = "2026-07-16-purchase-history-v2"


def _preload_streamlit_models(agent: PetRecommendationAgent) -> None:
    try:
        agent.preload_models(include_reranker=True)
        logger.info("[model preload] Streamlit chat model and BGE reranker are ready")
    except Exception:
        logger.exception("[model preload] Streamlit background preload failed")


@st.cache_resource
def load_agent(runtime_version: str) -> PetRecommendationAgent:
    """Cache one Agent instance per compatible graph-runtime version."""
    settings = Settings.from_env()
    sqlite_store = SQLiteStore(settings.sqlite_path)
    sqlite_store.init_schema()
    vector_store = VectorStore(settings.chroma_path)
    agent = PetRecommendationAgent(sqlite_store, vector_store)
    Thread(target=_preload_streamlit_models, args=(agent,), daemon=True, name="pet-agent-preload").start()
    return agent


def stream_text(text: str):
    for char in text:
        yield char
        time.sleep(0.01)


def render_stream_text(text: str) -> None:
    if hasattr(st, "write_stream"):
        st.write_stream(stream_text(text))
        return
    placeholder = st.empty()
    rendered = ""
    for char in text:
        rendered += char
        placeholder.markdown(rendered)
        time.sleep(0.01)


def inject_chat_style() -> None:
    st.markdown(
        """
        <style>
        :root {
            --app-bg: #f6f6f3;
            --sidebar-bg: #f2f3ef;
            --surface: #ffffff;
            --surface-soft: #f6f7f4;
            --ink: #202123;
            --muted: #667085;
            --line: #e0e2da;
            --line-strong: #d2d6cc;
            --accent: #2f7d59;
            --accent-soft: #e4f1ea;
            --shadow-soft: 0 1px 2px rgba(16, 24, 40, .04);
        }
        .stApp {
            background: var(--app-bg);
            color: var(--ink);
        }
        .stApp > div {
            background: var(--app-bg);
        }
        header[data-testid="stHeader"] {
            background: transparent;
        }
        div[data-testid="stToolbar"] {
            right: 14px;
        }
        [data-testid="stSidebar"] {
            background: var(--sidebar-bg);
            border-right: 1px solid var(--line);
        }
        [data-testid="stSidebar"] > div:first-child {
            background: var(--sidebar-bg);
            padding-top: 28px;
            padding-left: 18px;
            padding-right: 18px;
        }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] span {
            color: var(--ink);
        }
        [data-testid="stSidebar"] hr {
            border-color: var(--line);
        }
        .block-container {
            max-width: 1060px;
            padding-top: 46px;
            padding-bottom: 118px;
            padding-left: 42px;
            padding-right: 42px;
        }
        .app-title {
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 18px;
            padding: 0 0 24px;
            border-bottom: 1px solid var(--line);
            margin-bottom: 30px;
        }
        .app-title h1 {
            font-size: 25px;
            line-height: 1.2;
            margin: 0 0 6px;
            letter-spacing: 0;
        }
        .app-title p {
            margin: 0;
            color: var(--muted);
            font-size: 14px;
            line-height: 1.6;
        }
        .status-pill {
            flex: 0 0 auto;
            border: 1px solid #b9d8c6;
            color: #246548;
            background: var(--accent-soft);
            padding: 8px 12px;
            border-radius: 999px;
            font-size: 12px;
            white-space: nowrap;
        }
        .empty-chat {
            border: 1px solid var(--line);
            border-radius: 8px;
            background: rgba(255, 255, 255, .72);
            box-shadow: var(--shadow-soft);
            padding: 28px;
            margin: 44px 0 18px;
        }
        .empty-chat h2 {
            font-size: 18px;
            margin: 0 0 10px;
            letter-spacing: 0;
        }
        .empty-chat p {
            margin: 0;
            color: var(--muted);
            line-height: 1.7;
        }
        [data-testid="stChatMessage"] {
            background: transparent;
            padding: 4px 0;
        }
        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
            max-width: 760px;
        }
        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stMarkdownContainer"] {
            background: #e9ece5;
            border: 1px solid #d8ddd3;
            border-radius: 8px;
            padding: 10px 13px;
            margin-left: auto;
        }
        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) [data-testid="stMarkdownContainer"] {
            background: transparent;
            padding: 2px 0;
        }
        [data-testid="stChatInput"] {
            max-width: 1060px;
            margin: 0 auto;
            padding-left: 42px;
            padding-right: 42px;
            background: transparent;
        }
        div[data-testid="stBottom"] {
            background: linear-gradient(180deg, rgba(246, 246, 243, 0), var(--app-bg) 28%);
            border-top: 0;
            box-shadow: none;
        }
        div[data-testid="stBottom"] > div {
            background: var(--app-bg);
            box-shadow: none;
        }
        .stChatFloatingInputContainer {
            background: linear-gradient(180deg, rgba(246, 246, 243, 0), var(--app-bg) 26%) !important;
            box-shadow: none !important;
        }
        [data-testid="stChatInput"] textarea {
            border-radius: 8px;
            border-color: transparent;
            background: rgba(255, 255, 255, .84);
            box-shadow: 0 0 0 1px var(--line), 0 6px 18px rgba(16, 24, 40, .045);
        }
        div[data-testid="stStatusWidget"] {
            border-radius: 8px;
            border-color: var(--line);
            background: rgba(255, 255, 255, .78);
            box-shadow: var(--shadow-soft);
        }
        .pending-card {
            border: 1px solid var(--line);
            border-radius: 8px;
            background: rgba(255, 255, 255, .86);
            box-shadow: 0 8px 22px rgba(16, 24, 40, .055);
            padding: 16px 18px;
            margin: 18px 0 12px;
        }
        .pending-card strong {
            display: block;
            margin-bottom: 4px;
        }
        .pending-card span {
            color: var(--muted);
            font-size: 13px;
        }
        .pending-lock {
            border: 1px dashed var(--line-strong);
            border-radius: 8px;
            background: rgba(255, 255, 255, .52);
            color: var(--muted);
            padding: 12px 14px;
            margin: 16px 0 0;
            font-size: 13px;
        }
        .result-section {
            margin-top: 16px;
            padding-top: 14px;
            border-top: 1px solid var(--line);
        }
        .result-section-title {
            font-size: 13px;
            font-weight: 700;
            color: #475467;
            margin: 0 0 10px;
        }
        .top-product {
            border-left: 3px solid var(--accent);
            background: var(--accent-soft);
            padding: 12px 14px;
            border-radius: 6px;
            margin: 12px 0;
        }
        .top-product strong {
            display: block;
            color: var(--ink);
            margin-bottom: 6px;
        }
        .top-product span {
            color: #3d5a4b;
            font-size: 13px;
        }
        .side-section {
            border-bottom: 1px solid var(--line);
            padding-bottom: 18px;
            margin-bottom: 18px;
        }
        .side-title {
            font-size: 13px;
            font-weight: 700;
            color: #475467;
            margin-bottom: 12px;
        }
        .context-line {
            display: flex;
            justify-content: space-between;
            gap: 10px;
            font-size: 13px;
            padding: 7px 0;
            border-bottom: 1px solid rgba(224, 226, 218, .8);
        }
        .context-line span:first-child {
            color: var(--muted);
        }
        .context-line span:last-child {
            color: var(--ink);
            text-align: right;
            overflow-wrap: anywhere;
        }
        .small-muted {
            color: var(--muted);
            font-size: 12px;
            line-height: 1.6;
        }
        div.stButton > button,
        div[data-testid="stFormSubmitButton"] > button {
            border-radius: 8px;
            border-color: var(--line-strong);
            background: rgba(255, 255, 255, .58);
        }
        div[data-testid="stFormSubmitButton"] > button[kind="primary"],
        div.stButton > button[kind="primary"] {
            background: var(--accent);
            border-color: var(--accent);
        }
        [data-testid="stSidebar"] div[data-baseweb="select"] > div,
        [data-testid="stSidebar"] input,
        [data-testid="stSidebar"] textarea {
            border-radius: 8px;
            border-color: var(--line);
            background: rgba(255, 255, 255, .7);
        }
        [data-testid="stSidebar"] div[data-testid="stForm"] {
            border-color: var(--line-strong);
            background: rgba(255, 255, 255, .34);
            border-radius: 8px;
            box-shadow: none;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def clear_runtime_context() -> None:
    st.session_state["messages"] = []
    st.session_state["requirement_context"] = {}
    st.session_state["pending_action"] = None
    st.session_state["queued_agent_turn"] = None
    st.session_state["queued_resume_turn"] = None
    st.session_state["memory_decisions"] = {}


def clean_context(context: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in context.items() if value not in (None, [], "")}


def format_budget(context: dict[str, Any]) -> str:
    min_value = context.get("budget_min")
    max_value = context.get("budget_max")
    if min_value is None and max_value is None:
        return "不限" if context.get("budget_confirmed") else "未确认"
    left = f"{min_value:g}" if isinstance(min_value, (int, float)) else "不限"
    right = f"{max_value:g}" if isinstance(max_value, (int, float)) else "不限"
    return f"{left} - {right} 元"


def render_sidebar_context(context: dict[str, Any]) -> None:
    rows = [
        ("品类", context.get("product_type") or "未确认"),
        ("预算", format_budget(context)),
        ("关注", "、".join(context.get("priority_tags") or []) or "未确认"),
        ("避免", "、".join(context.get("avoid_tags") or []) or "未确认"),
    ]
    for label, value in rows:
        st.markdown(
            f'<div class="context-line"><span>{label}</span><span>{value}</span></div>',
            unsafe_allow_html=True,
        )


def render_empty_chat() -> None:
    st.markdown(
        """
        <div class="empty-chat">
            <h2>开始一次产品选购对话</h2>
            <p>你可以直接说想买的智能宠物产品。系统会根据品类、预算、关注点和避免项逐步补齐需求，再基于本地商品表、评论证据和 rerank 结果给出推荐。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def scroll_pending_panel_into_view() -> None:
    components.html(
        """
        <script>
        const root = window.parent.document;
        window.setTimeout(() => {
            const main = root.querySelector("section.main");
            if (main && main.scrollHeight > main.clientHeight) {
                main.scrollTo({ top: main.scrollHeight, behavior: "smooth" });
                return;
            }
            const scrollables = Array.from(root.querySelectorAll("*"))
                .filter((el) => el.scrollHeight > el.clientHeight + 24);
            const target = scrollables.find((el) => String(el.className).includes("main")) || scrollables.at(-1);
            if (target) {
                target.scrollTo({ top: target.scrollHeight, behavior: "smooth" });
            }
        }, 120);
        </script>
        """,
        height=0,
    )


def build_agent_state(user_id: str, message: str) -> dict[str, Any]:
    context = st.session_state.get("requirement_context") or {}
    return {
        "user_id": user_id,
        "session_id": st.session_state["session_id"],
        "thread_id": st.session_state["session_id"],
        "message": message,
        "history": st.session_state["messages"],
        "product_type": context.get("product_type"),
        "budget_min": context.get("budget_min"),
        "budget_max": context.get("budget_max"),
        "pet_type": context.get("pet_type"),
        "priority_tags": context.get("priority_tags") or [],
        "avoid_tags": context.get("avoid_tags") or [],
        "budget_confirmed": bool(context.get("budget_confirmed")),
        "priority_confirmed": bool(context.get("priority_confirmed")),
        "avoid_confirmed": bool(context.get("avoid_confirmed")),
    }


def update_requirement_context(state: dict[str, Any]) -> None:
    st.session_state["requirement_context"] = clean_context(
        {
            "product_type": state.get("product_type"),
            "budget_min": state.get("budget_min"),
            "budget_max": state.get("budget_max"),
            "pet_type": state.get("pet_type"),
            "priority_tags": state.get("priority_tags") or [],
            "avoid_tags": state.get("avoid_tags") or [],
            "budget_confirmed": state.get("budget_confirmed"),
            "priority_confirmed": state.get("priority_confirmed"),
            "avoid_confirmed": state.get("avoid_confirmed"),
        }
    )


def update_pending_action(state: dict[str, Any]) -> None:
    response = state.get("response") or {}
    required_action = response.get("required_action") or state.get("required_action")
    if not required_action:
        st.session_state["pending_action"] = None
        return
    st.session_state["pending_action"] = {
        "required_action": required_action,
        "action_options": response.get("action_options") or [],
        "product_type": state.get("product_type"),
    }


def interrupt_payload(result: dict[str, Any]) -> dict[str, Any] | None:
    interrupts = result.get("__interrupt__") if isinstance(result, dict) else None
    if not interrupts:
        return None
    first = interrupts[0]
    payload = getattr(first, "value", None)
    if payload is None and isinstance(first, dict):
        payload = first.get("value")
    return payload if isinstance(payload, dict) else None


def set_pending_from_interrupt(payload: dict[str, Any]) -> None:
    response = payload.get("response") or {}
    st.session_state["pending_action"] = {
        "required_action": payload.get("required_action") or response.get("required_action"),
        "action_options": payload.get("action_options") or response.get("action_options") or [],
        "product_type": payload.get("product_type"),
    }


def queue_agent_turn(user_text: str, display_text: str | None = None) -> None:
    st.session_state["queued_agent_turn"] = {
        "user_text": user_text,
        "display_text": display_text,
    }


def queue_resume_turn(resume_value: dict[str, Any], display_text: str) -> None:
    st.session_state["queued_resume_turn"] = {
        "resume_value": resume_value,
        "display_text": display_text,
    }


def _format_value(value: Any, suffix: str = "") -> str:
    if value in (None, ""):
        return "暂无"
    if isinstance(value, float):
        return f"{value:g}{suffix}"
    return f"{value}{suffix}"


def _recommendation_detail_rows(response: dict[str, Any]) -> list[dict[str, Any]]:
    rows = response.get("comparison_table") or response.get("recommended_products") or []
    return [
        {
            "商品": item.get("title") or item.get("product_id") or "未知商品",
            "价格": _format_value(item.get("price"), "元"),
            "店铺": item.get("shop_name") or "暂无",
            "销量": _format_value(item.get("sales")),
            "推荐分": _format_value(item.get("recommendation_score")),
            "评论证据": "充分" if item.get("evidence_status") == "sufficient" else "不足",
        }
        for item in rows
    ]


def _evidence_detail_rows(response: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in response.get("review_evidence") or []:
        metadata = item.get("metadata") or {}
        rows.append(
            {
                "关联商品": item.get("candidate_product_title") or metadata.get("product_id") or "未知商品",
                "评论证据": item.get("document") or "暂无评论内容",
                "语义相关度": _format_value(item.get("bge_rerank_score") or item.get("final_evidence_score")),
            }
        )
    return rows


def render_recommendation_details(
    response: dict[str, Any],
    *,
    agent: PetRecommendationAgent,
    user_id: str,
    session_id: str,
) -> None:
    """Render the stable presentation layer from the Agent response contract."""
    products = response.get("recommended_products") or []
    if products:
        top = products[0]
        st.markdown('<div class="result-section"><div class="result-section-title">首推商品</div></div>', unsafe_allow_html=True)
        st.markdown(
            (
                '<div class="top-product">'
                f'<strong>{escape(str(top.get("title") or top.get("product_id") or "未知商品"))}</strong>'
                f'<span>价格：{escape(_format_value(top.get("price"), "元"))}　'
                f'店铺：{escape(str(top.get("shop_name") or "暂无"))}　'
                f'销量：{escape(_format_value(top.get("sales")))}</span>'
                '</div>'
            ),
            unsafe_allow_html=True,
        )

    comparison_rows = _recommendation_detail_rows(response)
    if comparison_rows:
        comparison_title = "候选商品对比" if products else "候选商品参考"
        st.markdown(
            f'<div class="result-section"><div class="result-section-title">{comparison_title}</div></div>',
            unsafe_allow_html=True,
        )
        st.dataframe(comparison_rows, use_container_width=True, hide_index=True)

    evidence_rows = _evidence_detail_rows(response)
    if evidence_rows:
        st.markdown('<div class="result-section"><div class="result-section-title">评论证据</div></div>', unsafe_allow_html=True)
        st.dataframe(evidence_rows, use_container_width=True, hide_index=True)

    risk_notes = response.get("risk_notes") or []
    if risk_notes:
        st.markdown('<div class="result-section"><div class="result-section-title">风险提示</div></div>', unsafe_allow_html=True)
        for note in risk_notes:
            st.warning(str(note))

    memory_action = response.get("memory_action") or {}
    product = memory_action.get("product") or {}
    if memory_action.get("action") != "confirm_purchase" or not product.get("product_id"):
        return

    action_key = f"{user_id}:{session_id}:{product['product_id']}"
    decisions = st.session_state.setdefault("memory_decisions", {})
    decision = decisions.get(action_key)
    if decision == "confirmed":
        st.success("已将该商品保存为当前用户最近一次确认购买记录。")
        return
    if decision == "ignored":
        st.caption("本次推荐未写入长期记忆。")
        return

    st.markdown('<div class="result-section"><div class="result-section-title">购买记忆</div></div>', unsafe_allow_html=True)
    confirm_column, ignore_column = st.columns(2)
    with confirm_column:
        if st.button(
            str(memory_action.get("label") or "确认已购买并记住"),
            key=f"confirm-memory:{action_key}",
            type="primary",
            use_container_width=True,
        ):
            try:
                result = agent.confirm_purchase(
                    user_id=user_id,
                    session_id=session_id,
                    product=product,
                    requirement_context=memory_action.get("requirement_context") or {},
                )
            except Exception as exc:
                st.error(f"购买记忆保存失败：{exc}")
            else:
                decisions[action_key] = "confirmed"
                if result.get("vector_indexed") is False:
                    st.warning("购买记录已保存，语义记忆索引暂未更新。")
                else:
                    st.success(
                        f"已保存最近购买记录（{result.get('purchase_count', 1)}/{result.get('max_items', 10)}）。"
                        "新建会话后可以询问上一次购买或全部购买记录。"
                    )
                if result.get("removed_oldest"):
                    st.caption("购买记录已达到10条，最早的一条记录已自动移除。")
    with ignore_column:
        if st.button(
            "暂不记忆",
            key=f"ignore-memory:{action_key}",
            use_container_width=True,
        ):
            decisions[action_key] = "ignored"
            st.caption("本次推荐未写入长期记忆。")


def render_agent_result(agent: PetRecommendationAgent, user_id: str, result: dict[str, Any]) -> None:
    # A Streamlit process can retain a cached Agent while source files change.
    # Use the current graph state when available, otherwise render this result.
    get_thread_values = getattr(agent, "get_thread_values", None)
    thread_state = (
        get_thread_values(user_id=user_id, session_id=st.session_state["session_id"])
        if callable(get_thread_values)
        else result
    )
    if thread_state:
        update_requirement_context(thread_state)

    payload = interrupt_payload(result)
    if payload:
        set_pending_from_interrupt(payload)
        response = payload.get("response") or {}
        answer = response.get("recommendation_reason") or "请补充当前缺失信息。"
        st.session_state["messages"].append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            render_stream_text(answer)
            st.caption("Graph 已暂停，等待你在下方完成选择后继续同一条线程。")
        return

    update_requirement_context(result)
    update_pending_action(result)
    response = result["response"]
    answer = response["recommendation_reason"]

    st.session_state["messages"].append({"role": "assistant", "content": answer, "details": response})
    with st.chat_message("assistant"):
        render_stream_text(answer)
        render_recommendation_details(
            response,
            agent=agent,
            user_id=user_id,
            session_id=st.session_state["session_id"],
        )
        if response.get("model_used"):
            st.caption("回答由已配置模型生成，并基于本地商品表、评分表和评论证据。")
        else:
            st.caption("本轮为 Graph 工具节点结果。")


def run_agent_turn(agent: PetRecommendationAgent, user_id: str, user_text: str, display_text: str | None = None) -> None:
    shown_text = display_text or user_text
    st.session_state["messages"].append({"role": "user", "content": shown_text})
    with st.chat_message("user"):
        st.markdown(shown_text)

    with st.status("Agent 思考中", expanded=True) as status:
        st.write("读取当前会话上下文")
        st.write("执行 LangGraph 节点编排")
        st.write("检查缺失字段或检索商品证据")
        st.write("生成客服回答")
        try:
            state = agent.invoke(build_agent_state(user_id, user_text))
        except Exception as exc:
            status.update(label="Agent 执行失败", state="error", expanded=False)
            answer = (
                "当前 Agent 没有完成推荐，因为后端检索或模型组件执行失败。\n\n"
                f"错误信息：{exc}\n\n"
                "如果错误提到 BGE reranker 或 HuggingFace cache，说明重排模型还没有下载到本地。"
            )
            st.session_state["messages"].append({"role": "assistant", "content": answer})
            with st.chat_message("assistant"):
                st.error(answer)
            return
        else:
            status.update(label="Agent 思考完成", state="complete", expanded=False)

    render_agent_result(agent, user_id, state)


def run_resume_turn(agent: PetRecommendationAgent, user_id: str, resume_value: dict[str, Any], display_text: str) -> None:
    st.session_state["messages"].append({"role": "user", "content": display_text})
    with st.chat_message("user"):
        st.markdown(display_text)

    with st.status("Graph 恢复执行中", expanded=True) as status:
        st.write("提交前端选择到当前 thread")
        st.write("从暂停节点继续执行")
        st.write("继续检查缺失字段或生成推荐")
        try:
            result = agent.resume(
                user_id=user_id,
                session_id=st.session_state["session_id"],
                resume_value=resume_value,
            )
        except Exception as exc:
            status.update(label="Graph 恢复失败", state="error", expanded=False)
            answer = f"当前 Graph 没有成功恢复。\n\n错误信息：{exc}"
            st.session_state["messages"].append({"role": "assistant", "content": answer})
            with st.chat_message("assistant"):
                st.error(answer)
            return
        else:
            status.update(label="Graph 恢复完成", state="complete", expanded=False)

    render_agent_result(agent, user_id, result)


def render_pending_action(agent: PetRecommendationAgent, user_id: str) -> None:
    pending = st.session_state.get("pending_action")
    if not pending:
        return

    action = pending["required_action"]
    options = pending.get("action_options") or []
    context = dict(st.session_state.get("requirement_context") or {})
    action_titles = {
        "select_product_type": ("选择商品品类", "先确认你要比较的智能宠物产品类型。"),
        "ask_budget": ("确认预算范围", "请选择系统根据当前品类价格生成的 5 个预算档位。"),
        "ask_priority_tags": ("选择核心关注点", "这些标签来自当前品类评论统计，按出现频率排序。"),
        "ask_avoid_tags": ("选择想避免的问题", "系统会在推荐时降低这些风险项明显的商品。"),
    }
    title, description = action_titles.get(action, ("继续补充需求", "完成当前缺失信息后继续推荐。"))

    st.markdown(
        f'<div class="pending-card"><strong>{title}</strong><span>{description}</span></div>',
        unsafe_allow_html=True,
    )
    with st.container():
        if action == "select_product_type":
            with st.form("pending_select_product_type"):
                choice = st.selectbox("选择商品品类", options)
                submitted = st.form_submit_button("确认品类")
                if submitted:
                    st.session_state["pending_action"] = None
                    queue_resume_turn(
                        {"product_type": choice, "display": choice},
                        f"我选择：{choice}",
                    )
                    st.rerun()

        elif action == "ask_budget":
            labels = [item.get("label", "") for item in options]
            with st.form("pending_ask_budget"):
                selected_label = st.selectbox("选择预算档位", labels)
                submitted = st.form_submit_button("确认预算")
                ignored = st.form_submit_button("暂不确定，按全价格范围推荐")
                if submitted:
                    selected = options[labels.index(selected_label)] if labels else {}
                    budget_min = selected.get("budget_min")
                    budget_max = selected.get("budget_max")
                    display = selected.get("label") or "暂不确定"
                    st.session_state["pending_action"] = None
                    queue_resume_turn(
                        {
                            "budget_min": budget_min,
                            "budget_max": budget_max,
                            "budget_confirmed": True,
                            "display": display,
                        },
                        f"我选择预算：{display}",
                    )
                    st.rerun()
                if ignored:
                    st.session_state["pending_action"] = None
                    queue_resume_turn(
                        {
                            "budget_min": None,
                            "budget_max": None,
                            "budget_confirmed": True,
                            "display": "全价格范围",
                        },
                        "我暂不确定预算，按全价格范围推荐",
                    )
                    st.rerun()

        elif action == "ask_priority_tags":
            tag_options = [item for item in options if item]
            with st.form("pending_ask_priority_tags"):
                use_default = st.checkbox("暂不确定，按高频关注点推荐", value=False)
                selected_tags = st.multiselect("选择关注点", tag_options, disabled=use_default)
                submitted = st.form_submit_button("确认关注点")
                ignored = st.form_submit_button("跳过，使用高频关注点")
                if submitted:
                    tags = tag_options[:3] if use_default else selected_tags
                    if not tags:
                        st.warning("请选择至少一个关注点，或勾选按高频关注点推荐。")
                        return
                    st.session_state["pending_action"] = None
                    queue_resume_turn(
                        {"priority_tags": tags, "priority_confirmed": True},
                        f"我选择关注点：{'、'.join(tags)}",
                    )
                    st.rerun()
                if ignored:
                    tags = tag_options[:3]
                    st.session_state["pending_action"] = None
                    if tags:
                        queue_resume_turn(
                            {"priority_tags": tags, "priority_confirmed": True},
                            f"我跳过关注点，默认使用：{'、'.join(tags)}",
                        )
                    else:
                        queue_resume_turn(
                            {"priority_tags": [], "priority_confirmed": True},
                            "我跳过关注点，按综合表现推荐",
                        )
                    st.rerun()

        elif action == "ask_avoid_tags":
            tag_options = [item for item in options if item and item != "无特别避免"]
            with st.form("pending_ask_avoid_tags"):
                no_avoid = st.checkbox("无特别避免", value=False)
                selected_tags = st.multiselect("选择希望避免的问题", tag_options, disabled=no_avoid)
                submitted = st.form_submit_button("确认避免项")
                ignored = st.form_submit_button("跳过，默认无特别避免")
                if submitted:
                    tags = ["无特别避免"] if no_avoid else selected_tags
                    if not tags:
                        st.warning("请选择希望避免的问题，或勾选无特别避免。")
                        return
                    st.session_state["pending_action"] = None
                    queue_resume_turn(
                        {"avoid_tags": tags, "avoid_confirmed": True},
                        f"我选择避免项：{'、'.join(tags)}",
                    )
                    st.rerun()
                if ignored:
                    tags = ["无特别避免"]
                    st.session_state["pending_action"] = None
                    queue_resume_turn(
                        {"avoid_tags": tags, "avoid_confirmed": True},
                        "我跳过避免项，默认无特别避免",
                    )
                    st.rerun()


st.set_page_config(page_title="智能宠物产品客服", page_icon="PET", layout="wide")
inject_chat_style()

agent = load_agent(AGENT_RUNTIME_VERSION)
agent.sqlite_store.ensure_default_user()

if "requirement_context" not in st.session_state:
    st.session_state["requirement_context"] = {}
if "pending_action" not in st.session_state:
    st.session_state["pending_action"] = None
if "queued_agent_turn" not in st.session_state:
    st.session_state["queued_agent_turn"] = None
if "queued_resume_turn" not in st.session_state:
    st.session_state["queued_resume_turn"] = None
if "memory_decisions" not in st.session_state:
    st.session_state["memory_decisions"] = {}
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "session_id" not in st.session_state:
    st.session_state["session_id"] = uuid.uuid4().hex

st.markdown(
    """
    <div class="app-title">
        <div>
            <h1>智能宠物产品客服</h1>
            <p>基于本地商品数据、评论证据和 Agent 工具链，为用户完成智能宠物产品选购推荐。</p>
        </div>
        <div class="status-pill">Agent 在线</div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown('<div class="side-section"><div class="side-title">用户</div>', unsafe_allow_html=True)
    users = agent.sqlite_store.list_app_users()
    if not users:
        users = [agent.sqlite_store.ensure_default_user()]

    user_options = {f"{item['display_name']} ({item['user_id']})": item for item in users}
    current_user_id = st.session_state.get("user_id", users[0]["user_id"])
    current_index = next((index for index, item in enumerate(users) if item["user_id"] == current_user_id), 0)
    selected_label = st.selectbox("切换用户", list(user_options), index=current_index)
    selected_user = user_options[selected_label]
    if selected_user["user_id"] != st.session_state.get("user_id"):
        st.session_state["user_id"] = selected_user["user_id"]
        st.session_state["session_id"] = uuid.uuid4().hex
        clear_runtime_context()
        agent.sqlite_store.touch_app_user(selected_user["user_id"])
        st.rerun()

    user_id = selected_user["user_id"]
    st.session_state["user_id"] = user_id
    agent.sqlite_store.touch_app_user(user_id)

    with st.form("create_user_form", clear_on_submit=True):
        new_display_name = st.text_input("新用户昵称", placeholder="例如：小明")
        submitted = st.form_submit_button("注册并切换")
        if submitted:
            created_user = agent.sqlite_store.create_app_user(new_display_name)
            st.session_state["user_id"] = created_user["user_id"]
            st.session_state["session_id"] = uuid.uuid4().hex
            clear_runtime_context()
            st.rerun()

    st.markdown(f'<div class="small-muted">当前用户：{selected_user["display_name"]}<br>{user_id}</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="side-section"><div class="side-title">当前需求</div>', unsafe_allow_html=True)
    context = st.session_state.get("requirement_context") or {}
    if context:
        render_sidebar_context(context)
    else:
        st.markdown('<div class="small-muted">还没有收集推荐需求。</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="side-section"><div class="side-title">会话</div>', unsafe_allow_html=True)
    if st.button("新建会话", use_container_width=True):
        st.session_state["session_id"] = uuid.uuid4().hex
        clear_runtime_context()
        st.rerun()
    st.markdown(f'<div class="small-muted">Session：{st.session_state["session_id"][:12]}</div></div>', unsafe_allow_html=True)
queued_turn = st.session_state.get("queued_agent_turn")
queued_resume = st.session_state.get("queued_resume_turn")
pending_action = st.session_state.get("pending_action")

if not st.session_state["messages"] and not pending_action and not queued_turn and not queued_resume:
    render_empty_chat()

for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and isinstance(message.get("details"), dict):
            render_recommendation_details(
                message["details"],
                agent=agent,
                user_id=user_id,
                session_id=st.session_state["session_id"],
            )

if queued_turn:
    st.session_state["queued_agent_turn"] = None
    run_agent_turn(
        agent,
        user_id,
        queued_turn["user_text"],
        queued_turn.get("display_text"),
    )
    if st.session_state.get("pending_action"):
        st.rerun()

if queued_resume:
    st.session_state["queued_resume_turn"] = None
    run_resume_turn(
        agent,
        user_id,
        queued_resume["resume_value"],
        queued_resume["display_text"],
    )
    if st.session_state.get("pending_action"):
        st.rerun()

pending_action = st.session_state.get("pending_action")
if pending_action and not queued_turn and not queued_resume:
    render_pending_action(agent, user_id)

if st.session_state.get("pending_action"):
    st.markdown(
        '<div class="pending-lock">请先完成上方选择。完成后我会继续推荐流程，避免新输入打断当前需求。</div>',
        unsafe_allow_html=True,
    )
    scroll_pending_panel_into_view()
else:
    prompt = st.chat_input("例如：我想买 200 元以内的宠物饮水机，最好安静一点")
    if prompt:
        run_agent_turn(agent, user_id, prompt)
        if st.session_state.get("pending_action"):
            st.rerun()
