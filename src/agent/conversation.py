"""
Conversational orchestration layer for the Signal Support Agent.

Key changes from v1:
- Multi-turn action state machine (pending_action tracking)
- Conversation memory (history passed to QA layer)
- Check-ticket ID extraction from user message
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import asdict, dataclass
from typing import Any, Optional

from src.agent.actions import (
    ActionResult,
    PendingAction,
    continue_pending_action,
    run_single_action,
    start_pending_action,
)
from src.agent.guardrails import (
    GuardrailDecision,
    check_user_message,
    redact_sensitive_text,
)
from src.agent.qa import answer_knowledge_query
from src.agent.router import RouteDecision, detect_platform, route_message

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Conversation turn data
# ---------------------------------------------------------------------------

@dataclass
class ConversationTurn:
    query: str
    answer: str
    intent: str
    route: dict[str, Any]
    guardrail: dict[str, Any]
    grounded: bool
    fallback: bool
    sources: list[dict]
    action: Optional[dict] = None
    citations: Optional[list[str]] = None
    reason_if_fallback: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["citations"] = data["citations"] or []
        return data


# ---------------------------------------------------------------------------
# History formatting for LLM context
# ---------------------------------------------------------------------------

MAX_HISTORY_TURNS = 10  # keep last N exchanges to avoid token overflow


def _format_history_for_prompt(history: list[dict[str, str]]) -> str:
    """
    Convert chat history into a string the QA layer can prepend
    to the user prompt for conversational context.
    """
    if not history:
        return ""

    recent = history[-(MAX_HISTORY_TURNS * 2):]  # each turn = user + assistant
    lines = []
    for msg in recent:
        role = msg.get("role", "user").capitalize()
        content = msg.get("content", "")
        # Truncate long assistant answers to save tokens
        if role == "Assistant" and len(content) > 300:
            content = content[:300] + "..."
        lines.append(f"{role}: {content}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Ticket-ID extraction
# ---------------------------------------------------------------------------

_TICKET_ID_RE = re.compile(r"\bSIG-[A-Z0-9]{8}\b", re.IGNORECASE)


def _extract_ticket_id(text: str) -> Optional[str]:
    match = _TICKET_ID_RE.search(text)
    return match.group(0).upper() if match else None


# ---------------------------------------------------------------------------
# Cancel / topic-switch detection for multi-turn actions
# ---------------------------------------------------------------------------

_CANCEL_PHRASES = {
    "cancel", "stop", "never mind", "nevermind", "forget it",
    "quit", "exit", "abort", "back", "go back", "start over",
    "no thanks", "no thank you", "not anymore", "no longer",
}

_CANCEL_PATTERNS = [
    r"\b(don'?t|do not)\s+(want|need|wish)\s+(to|a|the)\b",
    r"\bnot\s+interested\b",
    r"\b(cancel|stop|end)\s+(this|the|my)\b",
    r"\bi\s+changed\s+my\s+mind\b",
]


def _wants_to_cancel(message: str, pending: PendingAction) -> bool:
    """
    Detect if the user wants to abandon the in-progress action.

    Returns True if:
    - The message matches a cancel phrase or negation pattern
    - The message looks like a completely different topic / question
    - The message content doesn't match the expected parameter at all
    """
    text = re.sub(r"\s+", " ", (message or "").strip().lower())

    # Explicit cancel phrase
    if text in _CANCEL_PHRASES:
        return True
    for phrase in _CANCEL_PHRASES:
        if text.startswith(phrase):
            return True

    # Negation patterns (e.g. "i don't want to create a ticket")
    for pattern in _CANCEL_PATTERNS:
        if re.search(pattern, text):
            return True

    # Looks like a new question (has a question mark and is long enough)
    if "?" in text and len(text.split()) > 4:
        return True

    # Content-mismatch: message looks like a new topic, not a param value
    current_param = pending.next_missing
    if current_param:
        param_name = current_param["name"]

        # If expecting email but input is clearly not an email
        if param_name == "email":
            has_at = "@" in text
            looks_like_sentence = len(text.split()) > 3
            if not has_at and looks_like_sentence:
                return True

        # If expecting device_os but input is a full sentence
        if param_name == "device_os":
            valid_os = {"android", "ios", "desktop", "iphone", "ipad"}
            words = set(text.split())
            if not (words & valid_os) and len(text.split()) > 3:
                return True

        # If expecting transfer_type but input is a full sentence
        if param_name == "transfer_type":
            valid_types = {"messages", "account", "both"}
            words = set(text.split())
            if not (words & valid_types) and len(text.split()) > 3:
                return True

        # If expecting issue_type but input looks like a completely different request
        if param_name == "issue_type":
            # Issue type is flexible, but if the input is a full sentence
            # that looks like a support question, it's probably a topic switch
            if len(text.split()) > 6 and any(
                phrase in text for phrase in [
                    "how do", "how to", "can i", "why is", "what is",
                    "help me", "i need", "my account", "not working",
                ]
            ):
                return True

        # Generic: if expecting a short param but got a long sentence
        if param_name in ("device_os", "transfer_type", "email"):
            if len(text.split()) > 8:
                return True

    return False


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class SupportAgent:
    """
    Routes user turns through guardrails -> multi-turn actions ->
    single-turn actions -> grounded QA.
    """

    def chat(
        self,
        message: str,
        history: Optional[list[dict[str, str]]] = None,
        platform_filter: Optional[str] = None,
        pending_action: Optional[PendingAction] = None,
    ) -> dict[str, Any]:
        """
        Process one user turn.

        Args:
            message:         The user's latest message.
            history:         Previous conversation turns [{role, content}, ...].
            platform_filter: Optional platform to filter retrieval results.
            pending_action:  In-progress multi-turn action state (from session).

        Returns:
            A dict with the full turn data plus a "pending_action" key
            that is non-None when a multi-turn action is still in progress.
        """
        raw_message = message or ""
        safe_message = redact_sensitive_text(raw_message)
        guardrail = check_user_message(safe_message)

        # --- Guardrail block ---
        if not guardrail.allowed:
            result = ConversationTurn(
                query=safe_message,
                answer=guardrail.message,
                intent="blocked",
                route={},
                guardrail=guardrail.to_dict(),
                grounded=False,
                fallback=True,
                sources=[],
                reason_if_fallback=guardrail.reason,
            ).to_dict()
            result["pending_action"] = (
                pending_action.to_dict() if pending_action else None
            )
            return result

        # --- Multi-turn action continuation ---
        if pending_action is not None and not pending_action.is_complete:
            # Check if the user wants to cancel or is switching topics
            if _wants_to_cancel(safe_message, pending_action):
                # Abandon the pending action and fall through to normal routing
                pending_action = None
            else:
                pending_action, action_result = continue_pending_action(
                    pending_action, safe_message
                )
                result = ConversationTurn(
                    query=safe_message,
                    answer=action_result.answer,
                    intent="action",
                    route={},
                    guardrail=guardrail.to_dict(),
                    grounded=False,
                    fallback=False,
                    sources=[],
                    action=action_result.to_dict(),
                ).to_dict()

                # If action just completed and it's device_transfer, also do QA
                if action_result.completed and pending_action.action_name == "device_transfer":
                    platform = platform_filter or detect_platform(safe_message)
                    qa_result = answer_knowledge_query(
                        f"How to transfer Signal {pending_action.collected.get('transfer_type', 'account')} "
                        f"from {pending_action.collected.get('source_device', '')} "
                        f"to {pending_action.collected.get('target_device', '')}",
                        platform_filter=platform,
                        conversation_history=_format_history_for_prompt(history or []),
                    )
                    if qa_result.get("answer"):
                        result["answer"] += f"\n\n{qa_result['answer']}"
                        result["sources"] = qa_result.get("sources", [])
                        result["grounded"] = qa_result.get("grounded", False)
                        result["citations"] = qa_result.get("citations", [])

                result["pending_action"] = (
                    pending_action.to_dict() if not action_result.completed else None
                )
                return result

        # --- Route the message ---
        route = route_message(safe_message)

        if route.intent == "greeting":
            result = self._simple_turn(
                safe_message,
                (
                    "Hi! I can help with Signal support questions -- "
                    "setup, troubleshooting, privacy, backups, transfers, "
                    "and verification. I can also create a support ticket "
                    "or help with a device transfer. What do you need?"
                ),
                route,
                guardrail,
            )
            result["pending_action"] = None
            return result

        if route.intent == "ambiguous":
            result = self._simple_turn(
                safe_message,
                (
                    "I'd like to help, but I'm not sure what you need. "
                    "Could you give me a bit more detail? For example:\n"
                    "- Ask a question about Signal features or troubleshooting\n"
                    "- Say **create a ticket** to open a support ticket\n"
                    "- Say **check ticket SIG-XXXXXXXX** to look up a ticket\n"
                    "- Say **transfer device** to start a device transfer request"
                ),
                route,
                guardrail,
                fallback=False,
                reason="",
            )
            result["pending_action"] = None
            return result

        if route.intent == "off_topic":
            result = self._simple_turn(
                safe_message,
                (
                    "I can help with Signal support topics like setup, "
                    "transfers, backups, privacy, verification, and "
                    "troubleshooting. I can also create a support ticket "
                    "or check an existing ticket status."
                ),
                route,
                guardrail,
                fallback=True,
                reason="Question is outside the Signal support scope.",
            )
            result["pending_action"] = None
            return result

        # --- Action handling ---
        if route.intent == "action" and route.action_name:
            # Multi-turn actions: create_ticket, device_transfer
            if route.action_name in ("create_ticket", "device_transfer"):
                pending_action_new, action_result = start_pending_action(
                    route.action_name, initial_message=safe_message
                )

                # Adjust intro based on whether action completed or has pre-filled params
                if action_result.completed:
                    intro_text = ""
                elif pending_action_new.collected:
                    intro_text = "I'll help you create a support ticket.\n\n" if route.action_name == "create_ticket" else "I'll help you set up a device transfer.\n\n"
                else:
                    intro_text = "I'll help you create a support ticket. I need a few details.\n\n" if route.action_name == "create_ticket" else "I'll help you set up a device transfer request. I need a few details.\n\n"

                result = ConversationTurn(
                    query=safe_message,
                    answer=intro_text + action_result.answer,
                    intent="action",
                    route=route.to_dict(),
                    guardrail=guardrail.to_dict(),
                    grounded=False,
                    fallback=False,
                    sources=[],
                    action=action_result.to_dict(),
                ).to_dict()
                result["pending_action"] = (
                    pending_action_new.to_dict() if not action_result.completed else None
                )
                return result

            # Single-turn: check_ticket
            if route.action_name == "check_ticket":
                ticket_id = _extract_ticket_id(safe_message)
                action_result = run_single_action(
                    "check_ticket", {"ticket_id": ticket_id or ""}
                )
                result = ConversationTurn(
                    query=safe_message,
                    answer=action_result.answer,
                    intent="action",
                    route=route.to_dict(),
                    guardrail=guardrail.to_dict(),
                    grounded=False,
                    fallback=False,
                    sources=[],
                    action=action_result.to_dict(),
                ).to_dict()
                result["pending_action"] = None
                return result

        # --- Knowledge QA ---
        platform = platform_filter or route.platform or detect_platform(safe_message)
        conversation_context = _format_history_for_prompt(history or [])

        qa_result = answer_knowledge_query(
            safe_message,
            platform_filter=platform,
            conversation_history=conversation_context,
        )

        result = ConversationTurn(
            query=safe_message,
            answer=qa_result.get("answer", ""),
            intent=route.intent,
            route=route.to_dict(),
            guardrail=guardrail.to_dict(),
            grounded=bool(qa_result.get("grounded", False)),
            fallback=bool(qa_result.get("fallback", False)),
            sources=qa_result.get("sources", []),
            citations=qa_result.get("citations", []),
            reason_if_fallback=qa_result.get("reason_if_fallback", ""),
        ).to_dict()
        result["pending_action"] = None
        return result

    @staticmethod
    def _simple_turn(
        query: str,
        answer: str,
        route: RouteDecision,
        guardrail: GuardrailDecision,
        fallback: bool = False,
        reason: str = "",
    ) -> dict[str, Any]:
        return ConversationTurn(
            query=query,
            answer=answer,
            intent=route.intent,
            route=route.to_dict(),
            guardrail=guardrail.to_dict(),
            grounded=False,
            fallback=fallback,
            sources=[],
            reason_if_fallback=reason,
        ).to_dict()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def run_cli() -> None:
    parser = argparse.ArgumentParser(description="Chat with the Signal Support Agent")
    parser.add_argument(
        "--platform",
        choices=["All", "Android", "iOS", "Desktop"],
        default=None,
        help="Optional platform filter for retrieval",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print full JSON responses instead of chat text",
    )
    args = parser.parse_args()

    agent = SupportAgent()
    print("Signal Support Agent. Type 'exit' or 'quit' to stop.")

    history: list[dict[str, str]] = []
    pending_action: Optional[PendingAction] = None

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break

        response = agent.chat(
            user_input,
            history=history,
            platform_filter=args.platform,
            pending_action=pending_action,
        )

        # Update pending action state
        pa_data = response.get("pending_action")
        if pa_data:
            pending_action = PendingAction.from_dict(pa_data)
        else:
            pending_action = None

        # Update history
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": response["answer"]})

        if args.json:
            print(json.dumps(response, indent=2, ensure_ascii=False))
            continue

        print(f"\nAgent: {response['answer']}")
        sources = response.get("sources", [])
        if sources:
            print("\nSources:")
            for source in sources:
                source_id = source.get("source_id", "-")
                title = source.get("title", "Signal Help Center article")
                url = source.get("url", "")
                print(f"- {source_id}: {title} {url}")


if __name__ == "__main__":
    run_cli()
