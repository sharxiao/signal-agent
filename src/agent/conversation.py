"""
Conversational orchestration layer for the Signal Support Agent.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from typing import Any, Optional

from src.agent.actions import run_mock_action
from src.agent.guardrails import GuardrailDecision, check_user_message, redact_sensitive_text
from src.agent.qa import answer_knowledge_query
from src.agent.router import RouteDecision, detect_platform, route_message

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


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


class SupportAgent:
    """
    Small support agent that routes a user turn through guardrails, mocked
    actions, and the grounded Signal Help Center QA layer.
    """

    def chat(
        self,
        message: str,
        history: Optional[list[dict[str, str]]] = None,
        platform_filter: Optional[str] = None,
    ) -> dict[str, Any]:
        del history  # Reserved for future multi-turn summarization.

        raw_message = message or ""
        safe_message = redact_sensitive_text(raw_message)
        guardrail = check_user_message(safe_message)

        if not guardrail.allowed:
            return ConversationTurn(
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

        route = route_message(safe_message)
        if route.intent == "greeting":
            return self._simple_turn(
                safe_message,
                "Hi, I can help with Signal support questions. What are you trying to do?",
                route,
                guardrail,
            )

        if route.intent == "off_topic":
            return self._simple_turn(
                safe_message,
                "I can help with Signal support topics like setup, transfers, backups, privacy, verification, and troubleshooting.",
                route,
                guardrail,
                fallback=True,
                reason="Question is outside the Signal support scope.",
            )

        action_result = run_mock_action(route.action_name)
        platform = platform_filter or route.platform or detect_platform(safe_message)
        qa_result = answer_knowledge_query(
            safe_message,
            platform_filter=platform,
        )

        answer = qa_result.get("answer", "")
        action_payload = None
        if action_result is not None:
            action_payload = action_result.to_dict()
            answer = f"{action_result.answer}\n\n{answer}"

        return ConversationTurn(
            query=safe_message,
            answer=answer,
            intent=route.intent,
            route=route.to_dict(),
            guardrail=guardrail.to_dict(),
            grounded=bool(qa_result.get("grounded", False)),
            fallback=bool(qa_result.get("fallback", False)),
            sources=qa_result.get("sources", []),
            action=action_payload,
            citations=qa_result.get("citations", []),
            reason_if_fallback=qa_result.get("reason_if_fallback", ""),
        ).to_dict()

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

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break

        response = agent.chat(user_input, platform_filter=args.platform)
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
