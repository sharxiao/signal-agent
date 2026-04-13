"""
Stateful mock support actions for the Signal Support Agent.

Three actions (per proposal):
1. Create Security Support Ticket  — multi-turn, collects 4 params
2. Check Ticket Status             — single-turn, looks up by ticket ID
3. Start Device Transfer Assistance — collects 3 params

All ticket / transfer data is persisted to a JSON file so state
survives across sessions.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Persistent store — simple JSON file
# ---------------------------------------------------------------------------

_DEFAULT_STORE_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "action_store.json"


def _load_store(path: Path = _DEFAULT_STORE_PATH) -> dict[str, Any]:
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {"tickets": {}, "transfers": {}}


def _save_store(store: dict[str, Any], path: Path = _DEFAULT_STORE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(store, f, indent=2, ensure_ascii=False)


def load_user_store(user_id: str) -> dict[str, dict]:
    """Load only the tickets and transfers belonging to a specific user."""
    store = _load_store()
    user_tickets = {
        tid: t for tid, t in store.get("tickets", {}).items()
        if t.get("user_id", "anonymous") == user_id
    }
    user_transfers = {
        trf_id: t for trf_id, t in store.get("transfers", {}).items()
        if t.get("user_id", "anonymous") == user_id
    }
    return {"tickets": user_tickets, "transfers": user_transfers}


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class ActionResult:
    """Returned to the conversation layer after every action turn."""
    name: str
    answer: str
    completed: bool = False
    requires_human: bool = False
    pending_params: list[str] = field(default_factory=list)
    ticket_id: Optional[str] = None
    transfer_id: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Parameter schemas for multi-turn collection
# ---------------------------------------------------------------------------

TICKET_PARAMS: list[dict[str, str]] = [
    {
        "name": "issue_type",
        "prompt": "What type of issue are you experiencing? (e.g., registration, verification, backup, transfer, notification, other)",
        "validate_hint": "a short issue category",
    },
    {
        "name": "device_os",
        "prompt": "What device or operating system are you using? (Android, iOS, or Desktop)",
        "validate_hint": "Android, iOS, or Desktop",
    },
    {
        "name": "email",
        "prompt": "Please provide a contact email address so the support team can follow up.",
        "validate_hint": "a valid email address",
    },
    {
        "name": "description",
        "prompt": "Please describe the issue in a few sentences.",
        "validate_hint": "a short description of the problem",
    },
]

TRANSFER_PARAMS: list[dict[str, str]] = [
    {
        "name": "source_device",
        "prompt": "What device are you transferring FROM? (e.g., Android phone, iPhone, Desktop)",
        "validate_hint": "a device type",
    },
    {
        "name": "target_device",
        "prompt": "What device are you transferring TO?",
        "validate_hint": "a device type",
    },
    {
        "name": "transfer_type",
        "prompt": "What would you like to transfer? (messages, account, or both)",
        "validate_hint": "messages, account, or both",
    },
]


# ---------------------------------------------------------------------------
# Pending-action state (lives in session, managed by conversation.py)
# ---------------------------------------------------------------------------

@dataclass
class PendingAction:
    """Tracks an in-progress multi-turn action."""
    action_name: str
    params_schema: list[dict[str, str]]
    collected: dict[str, str] = field(default_factory=dict)

    @property
    def next_missing(self) -> Optional[dict[str, str]]:
        for p in self.params_schema:
            if p["name"] not in self.collected:
                return p
        return None

    @property
    def is_complete(self) -> bool:
        return self.next_missing is None

    def to_dict(self) -> dict:
        return {
            "action_name": self.action_name,
            "collected": self.collected,
            "remaining": [
                p["name"] for p in self.params_schema
                if p["name"] not in self.collected
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PendingAction":
        schema_map = {
            "create_ticket": TICKET_PARAMS,
            "device_transfer": TRANSFER_PARAMS,
        }
        return cls(
            action_name=data["action_name"],
            params_schema=schema_map.get(data["action_name"], []),
            collected=data.get("collected", {}),
        )


# ---------------------------------------------------------------------------
# Validators (lightweight)
# ---------------------------------------------------------------------------

def _validate_email(value: str) -> bool:
    return bool(re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", value.strip()))


def _validate_device_os(value: str) -> bool:
    return value.strip().lower() in {"android", "ios", "desktop"}


def _validate_transfer_type(value: str) -> bool:
    return value.strip().lower() in {"messages", "account", "both"}


# ---------------------------------------------------------------------------
# Action executors
# ---------------------------------------------------------------------------

def _generate_ticket_id() -> str:
    short = uuid.uuid4().hex[:8].upper()
    return f"SIG-{short}"


def _generate_transfer_id() -> str:
    short = uuid.uuid4().hex[:8].upper()
    return f"TRF-{short}"


def execute_create_ticket(params: dict[str, str], user_id: str = "anonymous") -> ActionResult:
    """Create a support ticket and persist it."""
    ticket_id = _generate_ticket_id()
    now = datetime.now(timezone.utc).isoformat()

    ticket = {
        "ticket_id": ticket_id,
        "user_id": user_id,
        "issue_type": params["issue_type"],
        "device_os": params["device_os"],
        "email": params["email"],
        "description": params["description"],
        "status": "open",
        "created_at": now,
        "updated_at": now,
    }

    store = _load_store()
    store["tickets"][ticket_id] = ticket
    _save_store(store)

    log.info(f"Created ticket {ticket_id} for user {user_id}")

    return ActionResult(
        name="create_ticket",
        answer=(
            f"Your support ticket has been created.\n\n"
            f"- **Ticket ID**: {ticket_id}\n"
            f"- **Issue type**: {params['issue_type']}\n"
            f"- **Device**: {params['device_os']}\n"
            f"- **Email**: {params['email']}\n"
            f"- **Description**: {params['description']}\n"
            f"- **Status**: Open\n\n"
            f"Please save your ticket ID ({ticket_id}) to check the status later."
        ),
        completed=True,
        ticket_id=ticket_id,
    )


def execute_check_ticket(params: dict[str, str]) -> ActionResult:
    """Look up an existing ticket by ID."""
    ticket_id = params.get("ticket_id", "").strip().upper()

    if not ticket_id:
        return ActionResult(
            name="check_ticket",
            answer="Please provide a ticket ID to look up (e.g., SIG-A1B2C3D4).",
            completed=False,
            pending_params=["ticket_id"],
        )

    store = _load_store()
    ticket = store.get("tickets", {}).get(ticket_id)

    if ticket is None:
        return ActionResult(
            name="check_ticket",
            answer=(
                f"No ticket found with ID **{ticket_id}**. "
                "Please double-check your ticket ID and try again."
            ),
            completed=True,
        )

    return ActionResult(
        name="check_ticket",
        answer=(
            f"Here is the status of your ticket:\n\n"
            f"- **Ticket ID**: {ticket['ticket_id']}\n"
            f"- **Issue type**: {ticket['issue_type']}\n"
            f"- **Device**: {ticket['device_os']}\n"
            f"- **Email**: {ticket['email']}\n"
            f"- **Description**: {ticket['description']}\n"
            f"- **Status**: {ticket['status']}\n"
            f"- **Created**: {ticket['created_at']}\n"
        ),
        completed=True,
        ticket_id=ticket_id,
    )


def execute_device_transfer(params: dict[str, str], user_id: str = "anonymous") -> ActionResult:
    """Store a mock device transfer / migration request."""
    transfer_id = _generate_transfer_id()
    now = datetime.now(timezone.utc).isoformat()

    transfer = {
        "transfer_id": transfer_id,
        "user_id": user_id,
        "source_device": params["source_device"],
        "target_device": params["target_device"],
        "transfer_type": params["transfer_type"],
        "status": "pending",
        "created_at": now,
    }

    store = _load_store()
    store["transfers"][transfer_id] = transfer
    _save_store(store)

    log.info(f"Created transfer request {transfer_id} for user {user_id}")

    return ActionResult(
        name="device_transfer",
        answer=(
            f"Your device transfer request has been created.\n\n"
            f"- **Transfer ID**: {transfer_id}\n"
            f"- **From**: {params['source_device']}\n"
            f"- **To**: {params['target_device']}\n"
            f"- **Transfer type**: {params['transfer_type']}\n"
            f"- **Status**: Pending\n\n"
            f"I will also retrieve relevant transfer instructions from "
            f"the Signal help documentation."
        ),
        completed=True,
        transfer_id=transfer_id,
    )


# ---------------------------------------------------------------------------
# Multi-turn helpers (called by conversation.py)
# ---------------------------------------------------------------------------

def _pre_extract_params(message: str, action_name: str) -> dict[str, str]:
    """
    Try to extract parameter values from the user's initial message.
    E.g. "I want to create a ticket because my phone was stolen. my email is user@example.com"
    should pre-fill issue_type, email, and description.
    """
    extracted: dict[str, str] = {}
    text = message.strip()
    lower = text.lower()

    if action_name == "create_ticket":
        # Extract email
        email_match = re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text)
        if email_match:
            extracted["email"] = email_match.group(0)

        # Extract device_os
        if any(w in lower for w in ["iphone", "ipad", "ios"]):
            extracted["device_os"] = "iOS"
        elif "android" in lower:
            extracted["device_os"] = "Android"
        elif "desktop" in lower:
            extracted["device_os"] = "Desktop"

        # Extract issue_type from keywords
        issue_keywords = {
            "registration": "registration",
            "verification": "verification",
            "verify": "verification",
            "backup": "backup",
            "restore": "backup",
            "transfer": "transfer",
            "notification": "notification",
            "stolen": "security",
            "hacked": "security",
            "compromised": "security",
            "locked": "registration",
            "pin": "registration",
        }
        for keyword, issue in issue_keywords.items():
            if keyword in lower:
                extracted["issue_type"] = issue
                break

        # Extract description — use the part after "because" or the whole message
        because_match = re.search(r"because\s+(.+?)(?:\.|my email|$)", lower, re.IGNORECASE)
        if because_match:
            extracted["description"] = because_match.group(1).strip()

    elif action_name == "device_transfer":
        # Extract devices
        if any(w in lower for w in ["iphone", "ipad", "ios"]):
            if "from" in lower and lower.index("iphone" if "iphone" in lower else "ios") < len(lower) // 2:
                extracted["source_device"] = "iPhone"
            else:
                extracted["target_device"] = "iPhone"
        if "android" in lower:
            if "from" in lower and lower.index("android") < len(lower) // 2:
                extracted["source_device"] = "Android"
            else:
                extracted["target_device"] = "Android"

        # Extract transfer type
        if "messages" in lower and "account" in lower:
            extracted["transfer_type"] = "both"
        elif "messages" in lower:
            extracted["transfer_type"] = "messages"
        elif "account" in lower:
            extracted["transfer_type"] = "account"

    return extracted


def start_pending_action(action_name: str, initial_message: str = "", user_id: str = "anonymous") -> tuple[PendingAction, ActionResult]:
    """
    Begin a new multi-turn action. Pre-extracts any parameters from the
    initial message. Returns the PendingAction state and the next prompt.
    """
    if action_name == "create_ticket":
        pending = PendingAction(action_name="create_ticket", params_schema=TICKET_PARAMS)
    elif action_name == "device_transfer":
        pending = PendingAction(action_name="device_transfer", params_schema=TRANSFER_PARAMS)
    else:
        raise ValueError(f"Unknown multi-turn action: {action_name}")

    # Pre-fill any params we can extract from the initial message
    if initial_message:
        pre_extracted = _pre_extract_params(initial_message, action_name)
        for key, value in pre_extracted.items():
            pending.collected[key] = value

    # Check if all params are now filled
    if pending.is_complete:
        if action_name == "create_ticket":
            result = execute_create_ticket(pending.collected, user_id=user_id)
        elif action_name == "device_transfer":
            result = execute_device_transfer(pending.collected, user_id=user_id)
        else:
            result = ActionResult(name=action_name, answer="Action completed.", completed=True)
        return pending, result

    # Ask for the next missing param
    next_param = pending.next_missing
    # Build a summary of what we already got
    pre_filled_parts = []
    for key, value in pending.collected.items():
        display = key.replace("_", " ").title()
        pre_filled_parts.append(f"  - {display}: {value}")

    if pre_filled_parts:
        pre_summary = "I've noted the following from your message:\n" + "\n".join(pre_filled_parts) + "\n\n"
    else:
        pre_summary = ""

    result = ActionResult(
        name=action_name,
        answer=f"{pre_summary}{next_param['prompt']}",
        completed=False,
        pending_params=[
            p["name"] for p in pending.params_schema
            if p["name"] not in pending.collected
        ],
    )
    return pending, result


def continue_pending_action(
    pending: PendingAction,
    user_input: str,
    user_id: str = "anonymous",
) -> tuple[PendingAction, ActionResult]:
    """
    Feed the next user message into the pending action.
    Validates, stores the param, and either asks for the next one
    or executes the completed action.
    """
    current = pending.next_missing
    if current is None:
        raise ValueError("No missing params — action should already be complete.")

    param_name = current["name"]
    value = user_input.strip()

    # --- lightweight validation ---
    if param_name == "email" and not _validate_email(value):
        result = ActionResult(
            name=pending.action_name,
            answer=f"That doesn't look like a valid email address. {current['prompt']}",
            completed=False,
            pending_params=[
                p["name"] for p in pending.params_schema
                if p["name"] not in pending.collected
            ],
        )
        return pending, result

    if param_name == "device_os" and not _validate_device_os(value):
        result = ActionResult(
            name=pending.action_name,
            answer=f"Please enter Android, iOS, or Desktop. {current['prompt']}",
            completed=False,
            pending_params=[
                p["name"] for p in pending.params_schema
                if p["name"] not in pending.collected
            ],
        )
        return pending, result

    if param_name == "transfer_type" and not _validate_transfer_type(value):
        result = ActionResult(
            name=pending.action_name,
            answer=f"Please enter messages, account, or both. {current['prompt']}",
            completed=False,
            pending_params=[
                p["name"] for p in pending.params_schema
                if p["name"] not in pending.collected
            ],
        )
        return pending, result

    # Store the collected value
    pending.collected[param_name] = value

    # Check if more params needed
    next_param = pending.next_missing
    if next_param is not None:
        result = ActionResult(
            name=pending.action_name,
            answer=f"Got it. {next_param['prompt']}",
            completed=False,
            pending_params=[
                p["name"] for p in pending.params_schema
                if p["name"] not in pending.collected
            ],
        )
        return pending, result

    # All params collected — execute the action
    if pending.action_name == "create_ticket":
        result = execute_create_ticket(pending.collected, user_id=user_id)
    elif pending.action_name == "device_transfer":
        result = execute_device_transfer(pending.collected, user_id=user_id)
    else:
        result = ActionResult(
            name=pending.action_name,
            answer="Action completed.",
            completed=True,
        )

    return pending, result


def run_single_action(action_name: str, params: dict[str, str]) -> ActionResult:
    """
    Execute a single-turn action that doesn't need multi-turn collection.
    Currently only check_ticket.
    """
    if action_name == "check_ticket":
        return execute_check_ticket(params)

    return ActionResult(
        name=action_name,
        answer="Unknown action.",
        completed=False,
    )
