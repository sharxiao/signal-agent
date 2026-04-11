"""
Mock support actions for the conversational agent.

These handlers do not access real Signal accounts. They provide safe,
support-oriented next steps while making that limitation clear.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class ActionResult:
    name: str
    answer: str
    completed: bool = False
    requires_human: bool = False

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "answer": self.answer,
            "completed": self.completed,
            "requires_human": self.requires_human,
        }


def _account_delete() -> ActionResult:
    return ActionResult(
        name="account_delete",
        answer=(
            "I cannot delete a Signal account for you, but I can guide you. "
            "Use Signal on your phone to start the account deletion flow, and make sure you understand "
            "that deleting an account can remove registration and account state. If you are unsure, "
            "review the linked Signal Help Center source before continuing."
        ),
        requires_human=True,
    )


def _transfer_device() -> ActionResult:
    return ActionResult(
        name="transfer_device",
        answer=(
            "I cannot move your messages directly, but I can walk you through the official transfer flow. "
            "Keep both devices nearby, confirm you are transferring on the same platform when required, "
            "and follow the Signal in-app transfer prompts carefully."
        ),
    )


def _backup_messages() -> ActionResult:
    return ActionResult(
        name="backup_messages",
        answer=(
            "I cannot create a backup from here. Signal backup and transfer options depend on your platform, "
            "so I will use the Signal Help Center context to explain the safest available option for your device."
        ),
    )


def _verification_code() -> ActionResult:
    return ActionResult(
        name="verification_code",
        answer=(
            "I cannot view or resend verification codes, and you should not share a verification code in chat. "
            "I can help troubleshoot common reasons a code does not arrive and point you to the official steps."
        ),
    )


def _registration_pin() -> ActionResult:
    return ActionResult(
        name="registration_pin",
        answer=(
            "I cannot recover or reset a Signal PIN for you. I can explain the official options and limits "
            "from Signal support documentation."
        ),
        requires_human=True,
    )


def _contact_support() -> ActionResult:
    return ActionResult(
        name="contact_support",
        answer=(
            "For account-specific issues, contact Signal support through the official help flow. "
            "Do not share verification codes, PINs, or private message contents in this chat."
        ),
        requires_human=True,
    )


ACTION_HANDLERS: dict[str, Callable[[], ActionResult]] = {
    "account_delete": _account_delete,
    "transfer_device": _transfer_device,
    "backup_messages": _backup_messages,
    "verification_code": _verification_code,
    "registration_pin": _registration_pin,
    "contact_support": _contact_support,
}


def run_mock_action(action_name: str | None) -> ActionResult | None:
    """Run a safe mocked action if the router selected one."""
    if not action_name:
        return None

    handler = ACTION_HANDLERS.get(action_name)
    if handler is None:
        return None
    return handler()
