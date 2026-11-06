"""Streamlit Chat UI for Enterprise Agentic RAG Platform.

Mimics ChatGPT / Microsoft Copilot chat experience with:
- Sidebar: New Chat button, Searchable Previous Conversations
- Main: Thread Message History, Markdown responses, Citations display, Message Feedback
"""

import os
from datetime import datetime
from typing import Any

import httpx
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
DEFAULT_USER_ID = os.getenv("DEFAULT_USER_ID", "dev-user")

st.set_page_config(
    page_title="Enterprise AI Copilot",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern minimalist chat interface
st.markdown(
    """
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .citation-card {
        background-color: rgba(240, 242, 246, 0.6);
        border-left: 3px solid #0066cc;
        padding: 8px 12px;
        margin-top: 6px;
        margin-bottom: 6px;
        border-radius: 4px;
        font-size: 0.85rem;
    }
    .citation-title {
        font-weight: 600;
        color: #1a56db;
    }
    .citation-source {
        color: #6b7280;
        font-size: 0.75rem;
    }
    .citation-snippet {
        margin-top: 4px;
        color: #374151;
        font-style: italic;
    }
    .stChatMessage {
        padding-top: 0.75rem;
        padding-bottom: 0.75rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_headers() -> dict[str, str]:
    return {"X-User-Id": st.session_state.get("user_id", DEFAULT_USER_ID)}


def fetch_conversations(search_query: str | None = None) -> list[dict[str, Any]]:
    try:
        params = {"q": search_query} if search_query else {}
        with httpx.Client(timeout=10.0) as client:
            res = client.get(f"{API_URL}/conversations", headers=get_headers(), params=params)
            if res.status_code == 200:
                return res.json()
    except Exception as exc:
        st.sidebar.caption(f"⚠️ Could not load history: {exc}")
    return []


def fetch_conversation(conversation_id: str) -> dict[str, Any] | None:
    try:
        with httpx.Client(timeout=10.0) as client:
            res = client.get(f"{API_URL}/conversation/{conversation_id}", headers=get_headers())
            if res.status_code == 200:
                return res.json()
    except Exception as exc:
        st.error(f"Failed to load conversation: {exc}")
    return None


def send_chat_message(message: str, conversation_id: str | None) -> dict[str, Any] | None:
    try:
        payload: dict[str, Any] = {"message": message}
        if conversation_id:
            payload["conversation_id"] = conversation_id

        with httpx.Client(timeout=120.0) as client:
            res = client.post(f"{API_URL}/chat", headers=get_headers(), json=payload)
            if res.status_code == 200:
                return res.json()
            st.error(f"Error {res.status_code}: {res.text}")
    except Exception as exc:
        st.error(f"Failed to send message: {exc}")
    return None


def submit_feedback(conversation_id: str, message_id: str, rating: str, comment: str | None = None):
    try:
        payload = {
            "conversation_id": conversation_id,
            "message_id": message_id,
            "rating": rating,
            "comment": comment,
        }
        with httpx.Client(timeout=10.0) as client:
            res = client.post(f"{API_URL}/feedback", headers=get_headers(), json=payload)
            if res.status_code == 204:
                st.toast("Feedback submitted! Thank you.", icon="✅")
    except Exception as exc:
        st.toast(f"Failed to submit feedback: {exc}", icon="⚠️")


# --- State Initialization ---
if "user_id" not in st.session_state:
    st.session_state.user_id = DEFAULT_USER_ID
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_title" not in st.session_state:
    st.session_state.thread_title = "New Chat"

# --- Sidebar ---
with st.sidebar:
    st.title("Enterprise Copilot")

    if st.button("➕ New Chat", use_container_width=True, type="primary"):
        st.session_state.conversation_id = None
        st.session_state.messages = []
        st.session_state.thread_title = "New Chat"
        st.rerun()

    st.markdown("---")
    st.subheader("Previous Chats")

    search_term = st.text_input("🔍 Search chats", placeholder="Filter history...", label_visibility="collapsed")
    conversations = fetch_conversations(search_term if search_term.strip() else None)

    if conversations:
        for conv in conversations:
            c_id = conv["id"]
            title = conv.get("title") or "Untitled Conversation"
            is_active = c_id == st.session_state.conversation_id
            btn_label = f"💬 {'▶ ' if is_active else ''}{title}"

            if st.button(btn_label, key=f"conv_{c_id}", use_container_width=True):
                data = fetch_conversation(c_id)
                if data:
                    st.session_state.conversation_id = data["id"]
                    st.session_state.thread_title = data.get("title", "Conversation")
                    st.session_state.messages = data.get("messages", [])
                    st.rerun()
    else:
        st.caption("No conversations yet.")

    st.markdown("---")
    # Health indicator in bottom sidebar
    try:
        with httpx.Client(timeout=2.0) as client:
            health_check = client.get(f"{API_URL}/health")
            if health_check.status_code == 200:
                st.caption("🟢 Backend Connected")
            else:
                st.caption("🔴 Backend Degraded")
    except Exception:
        st.caption("🔴 Backend Offline")


# --- Main Chat Area ---
st.header(st.session_state.thread_title)

# Render conversation history
for idx, msg in enumerate(st.session_state.messages):
    role = msg.get("role", "user")
    content = msg.get("content", "")
    citations = msg.get("citations", [])
    msg_id = msg.get("id")

    with st.chat_message(role):
        st.markdown(content)

        if citations:
            with st.expander(f"📚 Sources & Citations ({len(citations)})", expanded=False):
                for c in citations:
                    st.markdown(
                        f"""
                        <div class="citation-card">
                            <span class="citation-title">[{c.get('index')}] {c.get('title')}</span> 
                            <span class="citation-source">({c.get('kind', 'doc')}) - {c.get('source')}</span>
                            <div class="citation-snippet">{c.get('snippet', '')}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        if role == "assistant" and msg_id and st.session_state.conversation_id:
            col1, col2, _ = st.columns([0.05, 0.05, 0.9])
            with col1:
                if st.button("👍", key=f"up_{msg_id}_{idx}", help="Helpful"):
                    submit_feedback(st.session_state.conversation_id, msg_id, "up")
            with col2:
                if st.button("👎", key=f"down_{msg_id}_{idx}", help="Not helpful"):
                    submit_feedback(st.session_state.conversation_id, msg_id, "down")

# Chat input
if user_input := st.chat_input("Ask a question about enterprise policies, data, or search the web..."):
    # Append user message immediately to session state
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking & retrieving knowledge..."):
            response = send_chat_message(user_input, st.session_state.conversation_id)

        if response:
            answer = response["answer"]
            citations = response.get("citations", [])
            st.session_state.conversation_id = response["conversation_id"]
            st.session_state.thread_title = response.get("title", st.session_state.thread_title)

            st.markdown(answer)

            if citations:
                with st.expander(f"📚 Sources & Citations ({len(citations)})", expanded=False):
                    for c in citations:
                        st.markdown(
                            f"""
                            <div class="citation-card">
                                <span class="citation-title">[{c.get('index')}] {c.get('title')}</span> 
                                <span class="citation-source">({c.get('kind', 'doc')}) - {c.get('source')}</span>
                                <div class="citation-snippet">{c.get('snippet', '')}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

            # Record in session state
            st.session_state.messages.append(
                {
                    "id": response.get("message_id"),
                    "role": "assistant",
                    "content": answer,
                    "citations": citations,
                }
            )
            st.rerun()
