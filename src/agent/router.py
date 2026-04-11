"""
Intent routing for the Signal Support Agent.
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
}


@dataclass(frozen=True)
class RouteDecision:
    intent: str
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

    if text in {"hi", "hello", "hey", "help"}:
        return RouteDecision(
            intent="greeting",
            platform=platform,
            confidence=0.95,
            reason="Short greeting.",
        )

    action_name = _detect_action(text)
    if action_name:
        return RouteDecision(
            intent="action",
            action_name=action_name,
            platform=platform,
            confidence=0.85,
            reason=f"Matched mocked action flow: {action_name}.",
        )

    if _looks_like_signal_support(text):
        return RouteDecision(
            intent="knowledge",
            platform=platform,
            confidence=0.75,
            reason="Signal support terms detected.",
        )

    return RouteDecision(
        intent="off_topic",
        platform=platform,
        confidence=0.65,
        reason="No Signal support terms detected.",
    )


def _detect_action(text: str) -> Optional[str]:
    if re.search(r"\b(delete|deactivate|remove|close)\b.*\b(account)\b", text):
        return "account_delete"

    if re.search(r"\b(transfer|move|migrate|new phone|new device)\b", text):
        return "transfer_device"

    if re.search(r"\b(backup|restore|recover)\b", text):
        return "backup_messages"

    if re.search(r"\b(verification|verify|sms|code|register)\b", text):
        return "verification_code"

    if re.search(r"\b(pin|registration lock)\b", text):
        return "registration_pin"

    if re.search(r"\b(contact|human|agent|support ticket)\b", text):
        return "contact_support"

    return None


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
    ]
    return any(phrase in text for phrase in support_phrases)
