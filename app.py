"""
Streamlit chat UI for the Signal Support Agent.

Features:
- Signal blue theme with custom CSS
- Demo login (user-scoped ticket/transfer history)
- Quick-action buttons for common tasks
- Multi-turn progress indicator
- Sidebar with ticket/transfer history (per-user)
- Conversation memory
- Confidence badges
- Password protection for deployment
"""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Signal Support Agent",
    page_icon="💬",
    layout="centered",
    initial_sidebar_state="expanded",
)

from src.agent.actions import PendingAction, load_user_store, TICKET_PARAMS, TRANSFER_PARAMS
from src.agent.conversation import SupportAgent

# ---------------------------------------------------------------------------
# Password protection (for deployed version)
# ---------------------------------------------------------------------------

def check_password() -> bool:
    """Returns True if the user has entered the correct password."""
    try:
        app_password = st.secrets.get("APP_PASSWORD", "")
    except Exception:
        return True

    if not app_password:
        return True

    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if st.session_state.authenticated:
        return True

    st.markdown("### 🔒 Signal Support Agent")
    st.caption("Enter the password to access the agent.")
    pwd = st.text_input("Password", type="password", key="password_input")
    if pwd == app_password:
        st.session_state.authenticated = True
        st.rerun()
    elif pwd:
        st.error("Incorrect password. Please try again.")
    return False


if not check_password():
    st.stop()

# ---------------------------------------------------------------------------
# Demo login (user identification)
# ---------------------------------------------------------------------------

def get_user_id() -> str | None:
    """Show a simple login screen and return the user ID."""
    if "user_id" not in st.session_state:
        st.session_state.user_id = None

    if st.session_state.user_id:
        return st.session_state.user_id

    st.markdown(
        """
        <div style="background: linear-gradient(135deg, #3A76F0, #1B4AB5);
                    padding: 2rem; border-radius: 12px; text-align: center; margin-bottom: 1.5rem;">
            <div style="font-size: 40px; margin-bottom: 8px;">💬</div>
            <h1 style="color: white; margin: 0; font-size: 1.8rem;">Signal Support Agent</h1>
            <p style="color: rgba(255,255,255,0.85); margin: 8px 0 0; font-size: 0.9rem;">
                Enter your name to get started
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        name = st.text_input(
            "Your name",
            key="login_name",
            placeholder="e.g. Alice, Bob, ta-demo",
            label_visibility="collapsed",
        )
        if st.button("Start Chat →", use_container_width=True, type="primary"):
            if name.strip():
                st.session_state.user_id = name.strip().lower()
                st.rerun()
            else:
                st.warning("Please enter a name.")

    st.caption(
        "Each user has their own ticket and transfer history. "
        "Try logging in as different users to see isolated data."
    )
    return None


user_id = get_user_id()
if not user_id:
    st.stop()

# ---------------------------------------------------------------------------
# Signal brand theme
# ---------------------------------------------------------------------------

SIGNAL_BLUE = "#3A76F0"
SIGNAL_BLUE_LIGHT = "#EBF1FE"
SIGNAL_BLUE_DARK = "#1B4AB5"

st.markdown(
    f"""
    <style>
    .signal-header {{
        background: linear-gradient(135deg, {SIGNAL_BLUE}, {SIGNAL_BLUE_DARK});
        padding: 1.2rem 1.5rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 12px;
    }}
    .signal-header h1 {{
        color: white;
        font-size: 1.5rem;
        margin: 0;
        font-weight: 600;
    }}
    .signal-header p {{
        color: rgba(255,255,255,0.85);
        font-size: 0.85rem;
        margin: 0;
    }}
    .signal-logo {{
        width: 40px;
        height: 40px;
        background: white;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 22px;
        flex-shrink: 0;
    }}
    .progress-container {{
        background: {SIGNAL_BLUE_LIGHT};
        border: 1px solid {SIGNAL_BLUE}33;
        border-radius: 10px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.8rem;
    }}
    .progress-title {{
        font-size: 0.8rem;
        font-weight: 600;
        color: {SIGNAL_BLUE_DARK};
        margin-bottom: 0.5rem;
    }}
    .progress-steps {{
        display: flex;
        gap: 6px;
        margin-bottom: 0.4rem;
    }}
    .step {{
        flex: 1;
        height: 6px;
        border-radius: 3px;
        background: {SIGNAL_BLUE}33;
    }}
    .step.done {{
        background: {SIGNAL_BLUE};
    }}
    .step.current {{
        background: {SIGNAL_BLUE};
        opacity: 0.6;
        animation: pulse 1.5s infinite;
    }}
    @keyframes pulse {{
        0%, 100% {{ opacity: 0.6; }}
        50% {{ opacity: 1; }}
    }}
    .progress-label {{
        font-size: 0.75rem;
        color: #555;
    }}
    .sidebar-section-title {{
        font-size: 0.75rem;
        font-weight: 600;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.5rem;
    }}
    .ticket-card {{
        background: white;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 0.6rem 0.8rem;
        margin-bottom: 0.5rem;
        font-size: 0.8rem;
    }}
    .ticket-id {{
        font-weight: 600;
        color: {SIGNAL_BLUE};
    }}
    .ticket-status {{
        display: inline-block;
        padding: 1px 8px;
        border-radius: 10px;
        font-size: 0.7rem;
        font-weight: 500;
    }}
    .status-open {{
        background: #FFF3CD;
        color: #856404;
    }}
    .status-pending {{
        background: {SIGNAL_BLUE_LIGHT};
        color: {SIGNAL_BLUE_DARK};
    }}
    .ticket-detail {{
        color: #666;
        font-size: 0.75rem;
        margin-top: 2px;
    }}
    .user-badge {{
        display: inline-block;
        padding: 2px 10px;
        border-radius: 10px;
        font-size: 0.75rem;
        font-weight: 500;
        background: {SIGNAL_BLUE_LIGHT};
        color: {SIGNAL_BLUE_DARK};
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Agent loader
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading Signal support agent...")
def load_agent() -> SupportAgent:
    return SupportAgent()


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def render_header():
    st.markdown(
        """
        <div class="signal-header">
            <div class="signal-logo">💬</div>
            <div>
                <h1>Signal Support Agent</h1>
                <p>Ask about setup, troubleshooting, privacy, backups, transfers, and verification</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander("📎 Sources"):
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


def render_confidence_badge(metadata: dict) -> None:
    if not metadata:
        return
    intent = metadata.get("intent", "")
    if intent in ("greeting", "action", "blocked", "ambiguous", "off_topic"):
        return
    grounded = metadata.get("grounded", False)
    fallback = metadata.get("fallback", False)
    sources = metadata.get("sources", [])
    if fallback:
        level, color, bg = "Low", "#856404", "#FFF3CD"
    elif grounded and len(sources) >= 2:
        level, color, bg = "High", "#0F6E56", "#E1F5EE"
    elif grounded:
        level, color, bg = "Medium", "#1B4AB5", "#EBF1FE"
    else:
        level, color, bg = "Medium", "#1B4AB5", "#EBF1FE"
    retrieval_method = ""
    if sources:
        methods = set()
        for s in sources:
            m = s.get("retrieval_method", "")
            if m:
                methods.add(m)
        if methods:
            retrieval_method = f" · {', '.join(methods)}"
    st.markdown(
        f'<span style="display:inline-block;padding:2px 10px;border-radius:10px;'
        f'font-size:0.7rem;font-weight:500;background:{bg};color:{color};'
        f'margin-top:4px;">Confidence: {level}{retrieval_method}</span>',
        unsafe_allow_html=True,
    )


def render_progress_bar(pending_action_data: dict) -> None:
    if not pending_action_data:
        return
    action_name = pending_action_data.get("action_name", "")
    collected = pending_action_data.get("collected", {})
    remaining = pending_action_data.get("remaining", [])
    param_schemas = {
        "create_ticket": TICKET_PARAMS,
        "device_transfer": TRANSFER_PARAMS,
    }
    schema = param_schemas.get(action_name, [])
    if not schema:
        return
    total = len(schema)
    done_count = len(collected)
    action_labels = {
        "create_ticket": "Creating Support Ticket",
        "device_transfer": "Device Transfer Request",
    }
    title = action_labels.get(action_name, "Action in Progress")
    steps_html = ""
    for i, param in enumerate(schema):
        name = param["name"]
        if name in collected:
            steps_html += '<div class="step done"></div>'
        elif i == done_count:
            steps_html += '<div class="step current"></div>'
        else:
            steps_html += '<div class="step"></div>'
    next_param = remaining[0].replace("_", " ").title() if remaining else "Done"
    label = f"Step {done_count + 1} of {total} — Next: {next_param}"
    st.markdown(
        f"""
        <div class="progress-container">
            <div class="progress-title">📋 {title}</div>
            <div class="progress-steps">{steps_html}</div>
            <div class="progress-label">{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_history():
    """Show ticket and transfer records for the current user only."""
    store = load_user_store(st.session_state.user_id)
    tickets = store.get("tickets", {})
    transfers = store.get("transfers", {})

    st.sidebar.markdown("---")

    if tickets:
        st.sidebar.markdown(
            '<div class="sidebar-section-title">🎫 Support Tickets</div>',
            unsafe_allow_html=True,
        )
        for tid, ticket in sorted(
            tickets.items(),
            key=lambda x: x[1].get("created_at", ""),
            reverse=True,
        ):
            status_class = "status-open" if ticket.get("status") == "open" else "status-pending"
            desc = ticket.get("description", "")
            desc_preview = (desc[:60] + "...") if len(desc) > 60 else desc
            st.sidebar.markdown(
                f"""
                <div class="ticket-card">
                    <span class="ticket-id">{tid}</span>
                    <span class="ticket-status {status_class}">{ticket.get('status', 'open')}</span>
                    <div class="ticket-detail">{ticket.get('issue_type', '')} · {ticket.get('device_os', '')}</div>
                    <div class="ticket-detail">{desc_preview}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    if transfers:
        st.sidebar.markdown(
            '<div class="sidebar-section-title">📱 Transfer Requests</div>',
            unsafe_allow_html=True,
        )
        for trf_id, transfer in sorted(
            transfers.items(),
            key=lambda x: x[1].get("created_at", ""),
            reverse=True,
        ):
            st.sidebar.markdown(
                f"""
                <div class="ticket-card">
                    <span class="ticket-id">{trf_id}</span>
                    <span class="ticket-status status-pending">{transfer.get('status', 'pending')}</span>
                    <div class="ticket-detail">{transfer.get('source_device', '')} → {transfer.get('target_device', '')}</div>
                    <div class="ticket-detail">Type: {transfer.get('transfer_type', '')}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    if not tickets and not transfers:
        st.sidebar.caption("No tickets or transfers yet.")


def build_history() -> list[dict[str, str]]:
    history = []
    for msg in st.session_state.messages:
        history.append({"role": msg["role"], "content": msg["content"]})
    return history


# ---------------------------------------------------------------------------
# App layout
# ---------------------------------------------------------------------------

render_header()

agent = load_agent()

# --- Sidebar ---
st.sidebar.title("📋 Actions & History")

# Show current user with badge
st.sidebar.markdown(
    f'<span class="user-badge">👤 {st.session_state.user_id}</span>',
    unsafe_allow_html=True,
)

col_new, col_switch = st.sidebar.columns(2)
with col_new:
    if st.button("➕ New Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending_action = None
        st.rerun()
with col_switch:
    if st.button("🚪 Switch User", use_container_width=True):
        st.session_state.user_id = None
        st.session_state.messages = []
        st.session_state.pending_action = None
        st.rerun()

render_sidebar_history()

# --- Session state init ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                f"Hi {st.session_state.user_id.title()}! I can help with Signal support questions — "
                "setup, troubleshooting, privacy, backups, transfers, "
                "and verification. I can also create a support ticket "
                "or help with a device transfer. What do you need?"
            ),
            "sources": [],
        }
    ]

if "pending_action" not in st.session_state:
    st.session_state.pending_action = None

# --- Quick action buttons (only show when no pending action) ---
if not st.session_state.pending_action:
    cols = st.columns(3)
    with cols[0]:
        if st.button("🎫 Create Ticket", use_container_width=True):
            st.session_state.messages.append(
                {"role": "user", "content": "I want to create a support ticket"}
            )
            response = agent.chat(
                "I want to create a support ticket",
                history=build_history(),
                user_id=st.session_state.user_id,
            )
            st.session_state.pending_action = response.get("pending_action")
            st.session_state.messages.append({
                "role": "assistant",
                "content": response["answer"],
                "sources": response.get("sources", []),
                "metadata": response,
            })
            st.rerun()
    with cols[1]:
        if st.button("🔍 Check Ticket", use_container_width=True):
            st.session_state.messages.append(
                {"role": "user", "content": "I want to check a ticket status"}
            )
            st.session_state.messages.append({
                "role": "assistant",
                "content": "Sure! Please enter your ticket ID (e.g., SIG-A1B2C3D4).",
                "sources": [],
            })
            st.rerun()
    with cols[2]:
        if st.button("📱 Transfer Device", use_container_width=True):
            st.session_state.messages.append(
                {"role": "user", "content": "I want to transfer to a new phone"}
            )
            response = agent.chat(
                "I want to transfer to a new phone",
                history=build_history(),
                user_id=st.session_state.user_id,
            )
            st.session_state.pending_action = response.get("pending_action")
            st.session_state.messages.append({
                "role": "assistant",
                "content": response["answer"],
                "sources": response.get("sources", []),
                "metadata": response,
            })
            st.rerun()

# --- Multi-turn progress bar ---
if st.session_state.pending_action:
    render_progress_bar(st.session_state.pending_action)

# --- Render chat history ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_confidence_badge(message.get("metadata", {}))
            render_sources(message.get("sources", []))

# --- Handle new user input ---
user_message = st.chat_input("Ask a Signal support question...")

if user_message:
    st.session_state.messages.append({"role": "user", "content": user_message})
    with st.chat_message("user"):
        st.markdown(user_message)

    # Rebuild pending action from session state
    pending_action = None
    if st.session_state.pending_action:
        pending_action = PendingAction.from_dict(st.session_state.pending_action)

    with st.chat_message("assistant"):
        with st.spinner("Searching Signal help content..."):
            response = agent.chat(
                user_message,
                history=build_history(),
                pending_action=pending_action,
                user_id=st.session_state.user_id,
            )
        st.markdown(response["answer"])
        render_confidence_badge(response)
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
    st.rerun()
