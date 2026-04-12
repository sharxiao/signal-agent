"""
Guardrails for the Signal Support Agent.

Two layers:
1. Input guardrails  — check user message before processing
2. Output guardrails — check LLM-generated answer before returning to user
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

MAX_MESSAGE_CHARS = 4000

# ---------------------------------------------------------------------------
# Input guardrail patterns
# ---------------------------------------------------------------------------

UNSAFE_PATTERNS = [
    r"\b(read|spy|monitor|intercept)\b.*\b(someone|partner|wife|husband|friend|their)\b.*\b(signal|messages|texts)\b",
    r"\b(hack|break into|bypass)\b.*\b(signal|account|encryption|pin|password)\b",
    r"\bsteal\b.*\b(code|account|messages)\b",
    r"\bdecrypt\b.*\b(signal|messages|backup)\b",
]

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
    r"forget\s+(your|all|the)\s+(system\s+)?(prompt|instructions?|rules?|role)",
]

# Trick / manipulation patterns — attempts to make the agent produce inappropriate content
TRICK_PATTERNS = [
    r"\bwhat\s+(is\s+the\s+)?word\s+that\s+combines?\b",
    r"\b(combine|put\s+together|merge|join|concat)\b.*\b(letters?|words?|syllables?)\b",
    r"\bspell\s+out\b.*\b(letters?|words?)\b",
    r"\bsay\s+(a\s+)?bad\s+word\b",
    r"\b(what|say)\b.*\b(swear|curse|profan|slur|offensive)\b",
    r"\bcomplete\s+(this|the)\s+(word|sentence)\s*:?\s*[a-z]{1,3}\b",
    r"\bfinish\s+(this|the)\s+word\b",
    r"\brhymes?\s+with\b.*\b(duck|witch|bass|hit|lass|cork|bunt)\b",
]

# ---------------------------------------------------------------------------
# Output guardrail patterns
# ---------------------------------------------------------------------------

# Phrases that suggest the LLM is hallucinating steps not in Signal docs
HALLUCINATION_SIGNALS = [
    r"\b(as of (my|the) (last|latest) (update|training|knowledge))\b",
    r"\b(i('m| am) not sure but)\b",
    r"\b(based on my (general )?knowledge)\b",
    r"\b(i believe|i think|i assume)\b.*\b(signal|app)\b",
    r"\b(typically|usually|generally)\b.*\b(apps? (like|such as))\b",
    r"\b(note:?\s*i (don't|do not) have (access|real-?time))\b",
]

# Phrases that indicate the model leaked system instructions
SYSTEM_LEAK_SIGNALS = [
    r"\b(my (system|initial) (prompt|instructions?))\b",
    r"\b(i was (told|instructed|programmed) to)\b",
    r"\b(my (rules|guidelines|constraints) (are|include|say))\b",
    r"return json only",
    r"you are a support qa assistant",
]

# PII patterns that should never appear in output
OUTPUT_PII_PATTERNS = [
    (re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"), "phone number"),
    (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"), "email address"),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "SSN-like number"),
]

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

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


@dataclass(frozen=True)
class OutputGuardrailResult:
    """Result of checking a generated answer."""
    safe: bool
    issues: list[str]
    sanitized_answer: str

    def to_dict(self) -> dict:
        return {
            "safe": self.safe,
            "issues": self.issues,
        }


# ---------------------------------------------------------------------------
# Input guardrails
# ---------------------------------------------------------------------------

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
                    "I'm not able to help with accessing, monitoring, or intercepting "
                    "someone else's Signal account or messages. "
                    "If you're concerned about your own account security, I can help "
                    "with that — just let me know what's going on."
                ),
            )

    # Check prompt injection
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, lowered):
            return GuardrailDecision(
                allowed=False,
                reason="prompt_injection",
                message=(
                    "It looks like you're trying to modify how I work. "
                    "I'm designed specifically to help with Signal support topics. "
                    "Feel free to ask me about Signal setup, troubleshooting, "
                    "privacy settings, backups, or transfers."
                ),
            )

    # Check manipulation / trick attempts
    for pattern in TRICK_PATTERNS:
        if re.search(pattern, lowered):
            return GuardrailDecision(
                allowed=False,
                reason="manipulation_attempt",
                message=(
                    "That doesn't seem related to Signal support. "
                    "I'm here to help with Signal — things like account setup, "
                    "message backups, device transfers, privacy features, "
                    "and troubleshooting. What can I help you with?"
                ),
            )

    return GuardrailDecision(
        allowed=True,
        reason="allowed",
        message="",
    )


# ---------------------------------------------------------------------------
# Output guardrails
# ---------------------------------------------------------------------------

def check_agent_output(answer: str, query: str = "") -> OutputGuardrailResult:
    """
    Post-generation safety check on the LLM's answer.

    Checks for:
    1. Hallucination signals (model making things up outside documentation)
    2. System prompt leakage
    3. PII in the output
    4. Unsafe content that slipped through

    Returns an OutputGuardrailResult with the (possibly sanitized) answer.
    """
    if not answer:
        return OutputGuardrailResult(safe=True, issues=[], sanitized_answer=answer)

    issues: list[str] = []
    lowered = answer.lower()
    sanitized = answer

    # 1. Check for hallucination signals
    for pattern in HALLUCINATION_SIGNALS:
        if re.search(pattern, lowered):
            issues.append("hallucination_signal")
            break  # one is enough to flag

    # 2. Check for system prompt leakage
    for pattern in SYSTEM_LEAK_SIGNALS:
        if re.search(pattern, lowered):
            issues.append("system_leak")
            sanitized = (
                "I can only help with Signal support questions. "
                "Could you rephrase your question about Signal setup, "
                "privacy, backups, transfers, or troubleshooting?"
            )
            log.warning(f"Output guardrail blocked system leak in answer for query: {query[:80]}")
            return OutputGuardrailResult(
                safe=False,
                issues=issues,
                sanitized_answer=sanitized,
            )

    # 3. Check for PII leakage in output
    for pattern, pii_type in OUTPUT_PII_PATTERNS:
        matches = pattern.findall(sanitized)
        if matches:
            issues.append(f"pii_leak:{pii_type}")
            # Redact PII from output
            for match in matches:
                sanitized = sanitized.replace(match, f"[redacted-{pii_type}]")

    # 4. Check for unsafe content in output
    for pattern in UNSAFE_PATTERNS:
        if re.search(pattern, lowered):
            issues.append("unsafe_content_in_output")
            sanitized = (
                "I cannot provide that information. I can help with your own "
                "Signal setup, privacy, safety, or troubleshooting questions."
            )
            log.warning(f"Output guardrail blocked unsafe content for query: {query[:80]}")
            return OutputGuardrailResult(
                safe=False,
                issues=issues,
                sanitized_answer=sanitized,
            )

    safe = len(issues) == 0 or all("hallucination" in i for i in issues)

    return OutputGuardrailResult(
        safe=safe,
        issues=issues,
        sanitized_answer=sanitized,
    )
