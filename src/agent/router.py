"""
Intent routing for the Signal Support Agent.

Two-tier routing:
1. Fast regex-based classification for clear-cut intents
2. LLM-based fallback for ambiguous or uncertain queries
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass
from typing import Optional

import requests

from src.agent.config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

SIGNAL_TERMS = {
    "signal", "message", "messages", "chat", "chats",
    "backup", "restore", "transfer", "pin", "registration",
    "verification", "code", "desktop", "android", "ios",
    "iphone", "ipad", "privacy", "group", "contacts",
    "notification", "notifications", "sticker", "delete",
    "account", "phone", "ticket", "migrate", "linked",
    "safety", "block", "disappearing", "media", "call", "calling",
}

VALID_INTENTS = {"greeting", "action", "knowledge", "off_topic", "ambiguous"}
VALID_ACTIONS = {"create_ticket", "check_ticket", "device_transfer", None}


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


# ---------------------------------------------------------------------------
# LLM-based intent classifier (fallback)
# ---------------------------------------------------------------------------

_LLM_CLASSIFY_PROMPT = """You are an intent classifier for a Signal messaging app support agent.

Classify the user message into exactly ONE of these intents:
- "knowledge" — user is asking a question about Signal features, setup, troubleshooting, privacy, or how something works
- "action:create_ticket" — user wants to create/open/file a support ticket
- "action:check_ticket" — user wants to check/look up an existing ticket status
- "action:device_transfer" — user wants to start transferring Signal to a new device (not asking how — actually wants to do it)
- "greeting" — user is saying hello or asking for general help
- "off_topic" — question is not related to Signal at all
- "ambiguous" — cannot determine intent, need more information

Respond with ONLY a JSON object:
{"intent": "...", "confidence": 0.0-1.0, "reason": "brief explanation"}

User message: """


def _classify_with_llm(message: str) -> Optional[RouteDecision]:
    """Call the LLM to classify intent when regex is uncertain."""
    try:
        base = LLM_BASE_URL.rstrip("/")
        url = f"{base}/v1/chat/completions"

        payload = {
            "model": LLM_MODEL,
            "messages": [
                {"role": "user", "content": _LLM_CLASSIFY_PROMPT + message}
            ],
            "max_tokens": 150,
            "temperature": 0,
        }

        headers = {}
        if LLM_API_KEY and LLM_API_KEY != "replace_me":
            headers["Authorization"] = f"Bearer {LLM_API_KEY}"

        response = requests.post(url, json=payload, headers=headers, timeout=15)
        response.raise_for_status()
        data = response.json()

        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")

        # Handle list-type content from some models
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(item.get("text", ""))
            content = "\n".join(parts).strip()

        if not content:
            content = data.get("choices", [{}])[0].get("message", {}).get("reasoning_content", "")
            if isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        parts.append(item.get("text", ""))
                    else:
                        parts.append(str(item))
                content = "\n".join(parts).strip()

        # Parse JSON from response
        content = content.strip()
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)

        match = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if not match:
            return None

        result = json.loads(match.group(0))
        raw_intent = result.get("intent", "").strip().lower()
        confidence = float(result.get("confidence", 0.5))
        reason = result.get("reason", "LLM classification")

        platform = detect_platform(message)

        # Parse action intents
        action_name = None
        if raw_intent.startswith("action:"):
            action_name = raw_intent.split(":", 1)[1]
            if action_name not in {"create_ticket", "check_ticket", "device_transfer"}:
                action_name = None
                raw_intent = "knowledge"  # fallback
            else:
                raw_intent = "action"

        if raw_intent not in VALID_INTENTS:
            raw_intent = "knowledge"  # safe default

        return RouteDecision(
            intent=raw_intent,
            action_name=action_name,
            platform=platform,
            confidence=confidence,
            reason=f"LLM: {reason}",
        )

    except Exception as e:
        log.warning(f"LLM intent classification failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Main routing logic
# ---------------------------------------------------------------------------

def route_message(message: str) -> RouteDecision:
    """
    Two-tier routing:
    1. Fast regex match for clear intents
    2. LLM fallback when regex is uncertain
    """
    text = re.sub(r"\s+", " ", (message or "").strip().lower())
    platform = detect_platform(text)

    # --- Greetings (high confidence, no LLM needed) ---
    if text in {"hi", "hello", "hey", "help", "start"}:
        return RouteDecision(
            intent="greeting",
            platform=platform,
            confidence=0.95,
            reason="Short greeting.",
        )

    # --- Actions (regex) ---
    action_name = _detect_action(text)
    if action_name:
        return RouteDecision(
            intent="action",
            action_name=action_name,
            platform=platform,
            confidence=0.85,
            reason=f"Matched action: {action_name}.",
        )

    # --- Clear Signal knowledge ---
    if _looks_like_signal_support(text):
        return RouteDecision(
            intent="knowledge",
            platform=platform,
            confidence=0.75,
            reason="Signal support terms detected.",
        )

    # --- Uncertain: use LLM to classify ---
    llm_result = _classify_with_llm(text)
    if llm_result and llm_result.confidence >= 0.5:
        return llm_result

    # --- Ambiguous (regex fallback) ---
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


# ---------------------------------------------------------------------------
# Regex helpers
# ---------------------------------------------------------------------------

def _detect_action(text: str) -> Optional[str]:
    # --- Negation check: skip action detection if user is refusing/cancelling ---
    if _has_negation(text):
        return None

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

    # Device transfer — only explicit intent, not knowledge questions
    if _is_knowledge_question(text):
        return None
    if re.search(r"\b(transfer|move|migrate|switch)\b.*\b(device|phone|new phone|new device)\b", text):
        return "device_transfer"
    if re.search(r"\b(i want to|i need to|i'd like to|please|help me|start)\b.*\b(transfer|move|migrate|switch)\b", text):
        return "device_transfer"
    if re.search(r"\bnew (phone|device)\b", text) and re.search(r"\b(set up|setup|start|help)\b", text):
        return "device_transfer"

    return None


def _has_negation(text: str) -> bool:
    """Detect if the user is refusing or cancelling rather than requesting."""
    negation_patterns = [
        r"\b(don'?t|do not|doesn'?t|does not)\s+(want|need|wish)\b",
        r"\b(no longer|not anymore|never mind|nevermind)\b",
        r"\b(cancel|stop|abort|forget)\b.*\b(ticket|transfer|action|request)\b",
        r"\b(i\s+)?don'?t\s+(want|need)\s+(to|a|the)\b",
        r"\bnot\s+interested\b",
    ]
    for pattern in negation_patterns:
        if re.search(pattern, text):
            return True
    return False


def _is_knowledge_question(text: str) -> bool:
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
        "not working", "cannot send", "can't send",
        "cannot receive", "can't receive", "not getting",
        "lost my", "how do i", "how to", "set up", "sign up",
    ]
    return any(phrase in text for phrase in support_phrases)


def _is_ambiguous(text: str) -> bool:
    ambiguous_patterns = [
        r"^(help|problem|issue|error|broken|fix|stuck|trouble)\s*$",
        r"^(it|this|that)\s+(doesn't|does not|won't|isn't|is not)\s+work",
        r"^(can you|could you|please)\s+(help|assist)\b",
    ]
    for pattern in ambiguous_patterns:
        if re.search(pattern, text):
            return True
    return False
