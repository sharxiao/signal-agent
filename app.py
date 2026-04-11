"""
Streamlit chat UI for the Signal Support Agent.
"""

from __future__ import annotations

import streamlit as st

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


st.title("Signal Support Agent")
st.caption("Ask about Signal setup, troubleshooting, privacy, backups, transfers, and verification.")

agent = load_agent()

platform = st.sidebar.selectbox(
    "Platform filter",
    options=["Auto", "All", "Android", "iOS", "Desktop"],
    index=0,
)

if st.sidebar.button("Clear chat"):
    st.session_state.messages = []

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi, I can help with Signal support questions. What are you trying to do?",
            "sources": [],
        }
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_sources(message.get("sources", []))

user_message = st.chat_input("Ask a Signal support question")
if user_message:
    st.session_state.messages.append({"role": "user", "content": user_message})
    with st.chat_message("user"):
        st.markdown(user_message)

    selected_platform = None if platform == "Auto" else platform
    with st.chat_message("assistant"):
        with st.spinner("Searching Signal help content..."):
            response = agent.chat(user_message, platform_filter=selected_platform)
        st.markdown(response["answer"])
        render_sources(response.get("sources", []))

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response["answer"],
            "sources": response.get("sources", []),
            "metadata": response,
        }
    )
