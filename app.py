"""
Streamlit chat UI for the Signal Support Agent.

Changes from v1:
- Tracks pending_action in session state for multi-turn actions
- Passes conversation history to agent.chat()
- Rebuilds PendingAction from dict each turn
"""

from __future__ import annotations

import streamlit as st

from src.agent.actions import PendingAction
from src.agent.conversation import SupportAgent

st.set_page_config(page_title="Signal Support Agent", layout="centered")


@st.cache_resource(show_spinner="Loading Signal support agent...")
def load_agent() -> SupportAgent:
    return SupportAgent()


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander("Sources"):
        for source in sources:
            title = source.get("title") or "Signal Help Center article"
            url = source.get("url")
            section = source.get("section_heading")
            score = source.get("score")
            source_id = source.get("source_id")
            label = f"{source_id}: {title}" if source_id else title
            if url:
                st.markdown(f"- [{label}]({url})")
            else:
                st.markdown(f"- {label}")
            details = []
            if section:
                details.append(f"section: {section}")
            if source.get("platform"):
                details.append(f"platform: {source['platform']}")
            if isinstance(score, float):
                details.append(f"score: {score:.3f}")
            if details:
                st.caption(", ".join(details))


def build_history() -> list[dict[str, str]]:
    """Convert session messages into the history format expected by the agent."""
    history = []
    for msg in st.session_state.messages:
        history.append({
            "role": msg["role"],
            "content": msg["content"],
        })
    return history


# ---------------------------------------------------------------------------
# App layout
# ---------------------------------------------------------------------------

st.title("Signal Support Agent")
st.caption(
    "Ask about Signal setup, troubleshooting, privacy, backups, transfers, "
    "and verification. You can also create support tickets or check ticket status."
)

agent = load_agent()

platform = st.sidebar.selectbox(
    "Platform filter",
    options=["Auto", "All", "Android", "iOS", "Desktop"],
    index=0,
)

if st.sidebar.button("Clear chat"):
    st.session_state.messages = []
    st.session_state.pending_action = None

# --- Session state init ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hi! I can help with Signal support questions -- "
                "setup, troubleshooting, privacy, backups, transfers, "
                "and verification. I can also create a support ticket "
                "or help with a device transfer. What do you need?"
            ),
            "sources": [],
        }
    ]

if "pending_action" not in st.session_state:
    st.session_state.pending_action = None

# --- Render chat history ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_sources(message.get("sources", []))

# --- Handle new user input ---
user_message = st.chat_input("Ask a Signal support question")

if user_message:
    st.session_state.messages.append({"role": "user", "content": user_message})
    with st.chat_message("user"):
        st.markdown(user_message)

    selected_platform = None if platform == "Auto" else platform

    # Rebuild pending action from session state
    pending_action = None
    if st.session_state.pending_action:
        pending_action = PendingAction.from_dict(st.session_state.pending_action)

    with st.chat_message("assistant"):
        with st.spinner("Searching Signal help content..."):
            response = agent.chat(
                user_message,
                history=build_history(),
                platform_filter=selected_platform,
                pending_action=pending_action,
            )
        st.markdown(response["answer"])
        render_sources(response.get("sources", []))

    # Update pending action in session state
    st.session_state.pending_action = response.get("pending_action")

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response["answer"],
            "sources": response.get("sources", []),
            "metadata": response,
        }
    )
