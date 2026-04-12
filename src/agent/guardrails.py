"""
Guardrails for the Signal Support Agent.

Changes from v1:
- Added prompt injection detection patterns
- Added jailbreak attempt detection
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

MAX_MESSAGE_CHARS = 4000

# --- Unsafe intent patterns ---
UNSAFE_PATTERNS = [
    r"\b(read|spy|monitor|intercept)\b.*\b(someone|partner|wife|husband|friend|their)\b.*\b(signal|messages|texts)\b",
    r"\b(hack|break into|bypass)\b.*\b(signal|account|encryption|pin|password)\b",
    r"\bsteal\b.*\b(code|account|messages)\b",
    r"\bdecrypt\b.*\b(signal|messages|backup)\b",
]

# --- Prompt injection patterns ---
INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|above|earlier)\s+(instructions?|prompts?|rules?|directions?)",
    r"disregard\s+(all\s+)?(previous|prior|above|earlier)\s+(instructions?|prompts?|rules?)",
    r"you\s+are\s+now\s+(a|an|my)\s+",
    r"act\s+as\s+(a|an|if)\s+",
    r"pretend\s+(you\s+are|to\s+be|you're)\s+",
    r"(new|override|replace|change)\s+(system\s+)?(prompt|instructions?|persona|role)",
    r"system\s*:\s*",
    r"\bsystem\s+prompt\b",
    r"reveal\s+(your|the)\s+(system\s+)?(prompt|instructions?)",
    r"(what|show|tell|repeat|print)\s+(me\s+)?(your|the)\s+(system\s+)?(prompt|instructions?|rules?)",
    r"jailbreak",
    r"DAN\s+mode",
    r"\bdo\s+anything\s+now\b",
]

# --- Sensitive data redaction ---
SENSITIVE_REDACTIONS = [
    (re.compile(r"\b\d{6}\b"), "[redacted-code]"),
    (re.compile(r"\+?\d[\d\s().-]{7,}\d"), "[redacted-phone]"),
]


@dataclass(frozen=True)
class GuardrailDecision:
    allowed: bool
    reason: str
    message: str

    def to_dict(self) -> dict:
        return asdict(self)


def redact_sensitive_text(text: str) -> str:
    """Redact likely phone numbers and one-time verification codes."""
    redacted = text or ""
    for pattern, replacement in SENSITIVE_REDACTIONS:
        redacted = pattern.sub(replacement, redacted)
    return redacted


def check_user_message(message: str) -> GuardrailDecision:
    """
    Decide whether a user message should be handled by the support agent.
    Checks for: empty, too long, unsafe intent, and prompt injection.
    """
    normalized = re.sub(r"\s+", " ", (message or "").strip())

    if not normalized:
        return GuardrailDecision(
            allowed=False,
            reason="empty_message",
            message="Please enter a Signal support question.",
        )

    if len(normalized) > MAX_MESSAGE_CHARS:
        return GuardrailDecision(
            allowed=False,
            reason="message_too_long",
            message="Please shorten your question so I can help with one Signal issue at a time.",
        )

    lowered = normalized.lower()

    # Check unsafe intent
    for pattern in UNSAFE_PATTERNS:
        if re.search(pattern, lowered):
            return GuardrailDecision(
                allowed=False,
                reason="unsafe_request",
                message=(
                    "I cannot help access, monitor, decrypt, or bypass someone else's "
                    "Signal account or messages. I can help with your own Signal setup, "
                    "privacy, safety, or troubleshooting questions."
                ),
            )

    # Check prompt injection
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, lowered):
            return GuardrailDecision(
                allowed=False,
                reason="prompt_injection",
                message=(
                    "I'm a Signal support assistant and I can only help with "
                    "Signal-related questions. Please ask a support question "
                    "about Signal setup, privacy, backups, transfers, or troubleshooting."
                ),
            )

    return GuardrailDecision(
        allowed=True,
        reason="allowed",
        message="",
    )
