"""
Intent routing for the Signal Support Agent.

Changes from v1:
- Action names aligned to new stateful actions: create_ticket, check_ticket, device_transfer
- Added ambiguous intent handling (asks clarifying question)
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Optional

SIGNAL_TERMS = {
    "signal",
    "message",
    "messages",
    "chat",
    "chats",
    "backup",
    "restore",
    "transfer",
    "pin",
    "registration",
    "verification",
    "code",
    "desktop",
    "android",
    "ios",
    "iphone",
    "ipad",
    "privacy",
    "group",
    "contacts",
    "notification",
    "notifications",
    "sticker",
    "delete",
    "account",
    "phone",
    "ticket",
    "migrate",
    "linked",
    "safety",
    "block",
    "disappearing",
    "media",
    "call",
    "calling",
}


@dataclass(frozen=True)
class RouteDecision:
    intent: str  # greeting | action | knowledge | off_topic | ambiguous
    action_name: Optional[str] = None
    platform: Optional[str] = None
    confidence: float = 0.5
    reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def detect_platform(message: str) -> Optional[str]:
    text = (message or "").lower()
    if any(term in text for term in ["iphone", "ipad", "ios"]):
        return "iOS"
    if "android" in text:
        return "Android"
    if any(term in text for term in ["desktop", "windows", "mac", "linux"]):
        return "Desktop"
    return None


def route_message(message: str) -> RouteDecision:
    text = re.sub(r"\s+", " ", (message or "").strip().lower())
    platform = detect_platform(text)

    # --- Greetings ---
    if text in {"hi", "hello", "hey", "help", "start"}:
        return RouteDecision(
            intent="greeting",
            platform=platform,
            confidence=0.95,
            reason="Short greeting.",
        )

    # --- Actions ---
    action_name = _detect_action(text)
    if action_name:
        return RouteDecision(
            intent="action",
            action_name=action_name,
            platform=platform,
            confidence=0.85,
            reason=f"Matched action: {action_name}.",
        )

    # --- Signal knowledge ---
    if _looks_like_signal_support(text):
        return RouteDecision(
            intent="knowledge",
            platform=platform,
            confidence=0.75,
            reason="Signal support terms detected.",
        )

    # --- Ambiguous: has some signal-adjacent words but not clear ---
    if _is_ambiguous(text):
        return RouteDecision(
            intent="ambiguous",
            platform=platform,
            confidence=0.40,
            reason="Query may be Signal-related but intent is unclear.",
        )

    # --- Off topic ---
    return RouteDecision(
        intent="off_topic",
        platform=platform,
        confidence=0.65,
        reason="No Signal support terms detected.",
    )


def _detect_action(text: str) -> Optional[str]:
    # Create ticket
    if re.search(r"\b(create|open|file|submit|new)\b.*\b(ticket|case|request|issue)\b", text):
        return "create_ticket"
    if re.search(r"\b(ticket|case)\b.*\b(create|open|file|submit|new)\b", text):
        return "create_ticket"
    if "support ticket" in text:
        return "create_ticket"

    # Check ticket
    if re.search(r"\b(check|status|look\s*up|find|track)\b.*\b(ticket|case)\b", text):
        return "check_ticket"
    if re.search(r"\bSIG-[A-Z0-9]+\b", text, re.IGNORECASE):
        return "check_ticket"

    # Device transfer — only match explicit intent to START a transfer,
    # not informational questions like "how do I transfer messages"
    # Exclude patterns that start with "how", "can", "what", "why", "is" (knowledge questions)
    if _is_knowledge_question(text):
        return None
    if re.search(r"\b(transfer|move|migrate|switch)\b.*\b(device|phone|new phone|new device)\b", text):
        return "device_transfer"
    if re.search(r"\b(i want to|i need to|i'd like to|please|help me|start)\b.*\b(transfer|move|migrate|switch)\b", text):
        return "device_transfer"
    if re.search(r"\bnew (phone|device)\b", text) and re.search(r"\b(set up|setup|start|help)\b", text):
        return "device_transfer"

    return None


def _is_knowledge_question(text: str) -> bool:
    """Detect if the message is asking for information rather than requesting an action."""
    knowledge_prefixes = [
        r"^(how (do|can|to|does|should|would))\b",
        r"^(can (i|you|we))\b",
        r"^(what (is|are|does|do|should|happens|if))\b",
        r"^(why (is|are|does|do|can't|cannot|won't|isn't))\b",
        r"^(is (it|there|this|that))\b",
        r"^(where|when|which)\b",
        r"^(tell me|explain|describe)\b",
    ]
    for pattern in knowledge_prefixes:
        if re.search(pattern, text):
            return True
    return False


def _looks_like_signal_support(text: str) -> bool:
    tokens = set(re.findall(r"[a-z0-9]+", text))
    if tokens & SIGNAL_TERMS:
        return True
    support_phrases = [
        "not working",
        "cannot send",
        "can't send",
        "cannot receive",
        "can't receive",
        "not getting",
        "lost my",
        "how do i",
        "how to",
        "set up",
        "sign up",
    ]
    return any(phrase in text for phrase in support_phrases)


def _is_ambiguous(text: str) -> bool:
    """
    Catch borderline queries that might be Signal-related but lack
    enough context — e.g. single words like 'help', vague phrases.
    """
    ambiguous_patterns = [
        r"^(help|problem|issue|error|broken|fix|stuck|trouble)\s*$",
        r"^(it|this|that)\s+(doesn't|does not|won't|isn't|is not)\s+work",
        r"^(can you|could you|please)\s+(help|assist)\b",
    ]
    for pattern in ambiguous_patterns:
        if re.search(pattern, text):
            return True
    return False
