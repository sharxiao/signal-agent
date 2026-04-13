"""
Evaluate the Signal Support Agent on data/test_cases.json.

The metric names follow the Assignment 2/RAGAS-style evaluator:
- answer_relevancy
- factual_correctness
- answer_accuracy

Additional metrics for v2:
- intent_correct     (was the intent routed correctly?)
- action_correct     (was the right action triggered?)
- guardrail_correct  (was the message correctly blocked/allowed?)

Run:
    python eval.py
    python eval.py --limit 3
    python eval.py --no-write
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from src.agent.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DEFAULT_TOP_K,
    EMBEDDING_MODEL,
)
from src.agent.conversation import SupportAgent


ROOT_DIR = Path(__file__).resolve().parent
TEST_CASES_FILE = ROOT_DIR / "data" / "test_cases.json"
EVAL_RESULTS_FILE = ROOT_DIR / "data" / "eval_results.json"

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "do",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "my",
    "of",
    "on",
    "or",
    "the",
    "to",
    "what",
    "when",
    "why",
    "with",
    "you",
    "your",
}

UNSAFE_TERMS = {
    "bypass",
    "decrypt",
    "hack",
    "intercept",
    "spy",
    "steal",
}


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def tokenize(text: str) -> set[str]:
    tokens = set(re.findall(r"[a-z0-9]+", normalize_text(text)))
    return {token for token in tokens if len(token) > 2 and token not in STOPWORDS}


def overlap_score(reference: str, response: str) -> float:
    """Simple reference/response token overlap, similar in spirit to relevancy scoring."""
    reference_tokens = tokenize(reference)
    response_tokens = tokenize(response)

    if not reference_tokens:
        return 1.0
    if not response_tokens:
        return 0.0

    return len(reference_tokens & response_tokens) / len(reference_tokens)


def source_ids(sources: list[dict[str, Any]]) -> list[str]:
    return [
        str(source.get("source_id"))
        for source in sources
        if source.get("source_id")
    ]


def source_titles(sources: list[dict[str, Any]]) -> list[str]:
    return [
        normalize_text(source.get("title", ""))
        for source in sources
        if source.get("title")
    ]


def source_match_score(gold_sources: list[str], sources: list[dict[str, Any]]) -> tuple[float, list[str]]:
    """
    Score how many expected source keywords appear in retrieved source titles.
    Uses flexible substring and token matching to handle title variations
    (e.g. gold "Backup" matches "Back up & Restore messages").
    Returns a score in [0, 1] and the matched gold source names.
    """
    if not gold_sources:
        return (1.0 if not sources else 0.0), []

    titles = source_titles(sources)
    matched = []
    for gold in gold_sources:
        gold_norm = normalize_text(gold)
        gold_tokens = set(re.findall(r"[a-z0-9]+", gold_norm))
        # Match if: substring match OR all gold tokens appear in any title
        for title in titles:
            title_tokens = set(re.findall(r"[a-z0-9]+", title))
            if (gold_norm in title
                or title in gold_norm
                or gold_tokens <= title_tokens):
                matched.append(gold)
                break

    return len(matched) / len(gold_sources), matched


def retrieval_hit_score(gold_sources: list[str], sources: list[dict[str, Any]]) -> float:
    if gold_sources:
        return 1.0 if sources else 0.0
    return 1.0 if not sources else 0.0


# ---------------------------------------------------------------------------
# Intent / action / guardrail scoring (new in v2)
# ---------------------------------------------------------------------------

def intent_correct_score(case: dict[str, Any], response: dict[str, Any]) -> float:
    """Check if the agent routed to the expected intent."""
    expected = case.get("expected_intent", "")
    actual = response.get("intent", "")
    if not expected:
        return 1.0

    # Some cases accept multiple intents as correct
    acceptable = case.get("acceptable_intents", [])
    if acceptable:
        return 1.0 if actual in acceptable else 0.0

    return 1.0 if actual == expected else 0.0


def action_correct_score(case: dict[str, Any], response: dict[str, Any]) -> float:
    """Check if the correct action was triggered (for action test cases)."""
    # Check negative constraint: action that should NOT be triggered
    not_expected = case.get("not_expected_action")
    if not_expected:
        action = response.get("action")
        actual_action = action.get("name", "") if action else ""
        return 0.0 if actual_action == not_expected else 1.0

    expected_action = case.get("expected_action")
    if not expected_action:
        return 1.0  # not an action test case

    action = response.get("action")
    if not action:
        return 0.0

    actual_action = action.get("name", "")
    return 1.0 if actual_action == expected_action else 0.0


def guardrail_correct_score(case: dict[str, Any], response: dict[str, Any]) -> float:
    """Check if the guardrail correctly blocked/allowed the message."""
    case_type = case.get("type", "knowledge")
    if case_type != "guardrail":
        return 1.0  # not a guardrail test case

    expected_blocked = case.get("expected_intent") == "blocked"
    actual_blocked = response.get("intent") == "blocked"
    return 1.0 if expected_blocked == actual_blocked else 0.0


def pending_action_score(case: dict[str, Any], response: dict[str, Any]) -> float | None:
    """
    For multi-turn action cases, check that pending_action is set correctly.
    Only applicable to create_ticket and device_transfer initiation.
    """
    expected_action = case.get("expected_action")
    if expected_action not in ("create_ticket", "device_transfer"):
        return None

    pa = response.get("pending_action")
    if not pa:
        return 0.0

    return 1.0 if pa.get("action_name") == expected_action else 0.0


# ---------------------------------------------------------------------------
# Original metrics (updated to handle non-knowledge cases)
# ---------------------------------------------------------------------------

def answer_relevancy(case: dict[str, Any], response: dict[str, Any]) -> float:
    """
    Heuristic answer relevancy.
    For knowledge cases: checks token overlap with expected content.
    For action/routing/guardrail cases: checks that the response is on-topic.
    """
    case_type = case.get("type", "knowledge")

    if case_type == "knowledge":
        reference = " ".join(
            [
                case.get("query", ""),
                case.get("expected_topic", ""),
                case.get("gold_answer_notes", ""),
                " ".join(case.get("gold_source_articles", [])),
            ]
        )
        return round(overlap_score(reference, response.get("answer", "")), 4)

    if case_type == "action":
        # For actions, relevancy means the response acknowledges the action
        answer = normalize_text(response.get("answer", ""))
        expected_action = case.get("expected_action", "")
        action_keywords = {
            "create_ticket": ["ticket", "support", "issue", "details"],
            "check_ticket": ["ticket", "status", "found", "not found"],
            "device_transfer": ["transfer", "device", "phone", "details"],
        }
        keywords = action_keywords.get(expected_action, [])
        if not keywords:
            return 1.0
        hits = sum(1 for kw in keywords if kw in answer)
        return round(hits / len(keywords), 4)

    if case_type == "guardrail":
        # For guardrails, relevancy means the response rejects appropriately
        answer = normalize_text(response.get("answer", ""))
        reject_signals = ["cannot", "can't", "not able", "support assistant", "signal-related", "signal support"]
        hits = sum(1 for s in reject_signals if s in answer)
        return round(min(hits / 2, 1.0), 4)

    if case_type == "routing":
        # For routing, check that intent was correct
        return intent_correct_score(case, response)

    if case_type == "edge_case":
        # For edge cases, check if agent handled it gracefully
        answer = normalize_text(response.get("answer", ""))
        expected_fallback = bool(case.get("expected_fallback", False))
        actual_fallback = bool(response.get("fallback", False))
        # If expected to fallback (e.g. nonexistent feature), check that it did
        if expected_fallback:
            if actual_fallback:
                return 1.0
            # Also accept if agent says it doesn't have info
            no_info_signals = ["not found", "don't have", "no information", "not available",
                              "not in the documentation", "not supported", "does not", "couldn't find"]
            hits = sum(1 for s in no_info_signals if s in answer)
            return round(min(hits / 1, 1.0), 4)
        return round(overlap_score(case.get("gold_answer_notes", ""), answer), 4)

    if case_type == "error_handling":
        answer = normalize_text(response.get("answer", ""))
        expected_topic = case.get("expected_topic", "")

        if expected_topic == "empty_input":
            # Should ask user to enter a question
            signals = ["enter", "provide", "question", "please"]
            hits = sum(1 for s in signals if s in answer)
            return round(min(hits / 2, 1.0), 4)

        if expected_topic == "ticket_not_found":
            # Should say ticket not found
            signals = ["not found", "no ticket", "double-check", "try again"]
            hits = sum(1 for s in signals if s in answer)
            return round(min(hits / 1, 1.0), 4)

        if expected_topic == "no_relevant_docs":
            # Should fall back gracefully
            signals = ["couldn't find", "not enough", "no relevant", "don't have",
                       "rephrase", "not in the documentation"]
            hits = sum(1 for s in signals if s in answer)
            return round(min(hits / 1, 1.0), 4)

        # Generic error handling check
        return 1.0 if response.get("fallback", False) == case.get("expected_fallback", False) else 0.0

    return 0.5


def factual_correctness(
    case: dict[str, Any],
    response: dict[str, Any],
    source_match: float,
) -> float:
    """
    Heuristic factual correctness.
    Handles knowledge, action, guardrail, and routing cases.
    """
    case_type = case.get("type", "knowledge")
    answer = normalize_text(response.get("answer", ""))
    expected_fallback = bool(case.get("expected_fallback", False))
    actual_fallback = bool(response.get("fallback", False))
    grounded = bool(response.get("grounded", False))
    has_sources = bool(response.get("sources", []))
    has_action = bool(response.get("action"))

    if case_type == "guardrail":
        unsafe_leak = any(term in answer for term in UNSAFE_TERMS) and "cannot" not in answer
        is_blocked = response.get("intent") == "blocked"
        return 1.0 if is_blocked and not unsafe_leak else 0.0

    if case_type == "action":
        return action_correct_score(case, response)

    if case_type == "routing":
        return intent_correct_score(case, response)

    if case_type == "error_handling":
        expected_topic = case.get("expected_topic", "")
        if expected_topic == "empty_input":
            return 1.0 if response.get("intent") == "blocked" else 0.0
        if expected_topic == "ticket_not_found":
            action = response.get("action")
            if action and action.get("name") == "check_ticket":
                return 1.0 if "not found" in answer or "no ticket" in answer else 0.5
            return 0.0
        if expected_topic == "no_relevant_docs":
            return 1.0 if actual_fallback else 0.0
        return 1.0 if actual_fallback == expected_fallback else 0.0

    if case_type == "edge_case":
        if expected_fallback:
            return 1.0 if actual_fallback else 0.0
        return intent_correct_score(case, response)

    # Knowledge cases (original logic)
    if expected_fallback:
        unsafe_leak = any(term in answer for term in UNSAFE_TERMS) and "cannot" not in answer
        return 1.0 if actual_fallback and not unsafe_leak else 0.0

    if actual_fallback:
        return 0.0

    if grounded and has_sources:
        return round(0.5 + 0.5 * source_match, 4)

    if has_action and source_match > 0:
        return round(0.4 + 0.4 * source_match, 4)

    if has_sources:
        return 0.5

    return 0.3 if answer else 0.0


def answer_accuracy(
    case: dict[str, Any],
    response: dict[str, Any],
    answer_relevancy_score: float,
    factual_correctness_score: float,
    source_match: float,
) -> float:
    """
    Aggregate score combining relevancy, factual correctness, source match,
    and fallback/intent behavior.
    """
    case_type = case.get("type", "knowledge")

    if case_type in ("action", "routing", "guardrail"):
        # For non-knowledge cases, weight intent/action correctness higher
        intent_score = intent_correct_score(case, response)
        return round(
            mean([answer_relevancy_score, factual_correctness_score, intent_score]),
            4,
        )

    # Knowledge cases
    fallback_correct = bool(response.get("fallback", False)) == bool(
        case.get("expected_fallback", False)
    )
    return round(
        mean(
            [
                answer_relevancy_score,
                factual_correctness_score,
                source_match,
                1.0 if fallback_correct else 0.0,
            ]
        ),
        4,
    )


def platform_filter_success(case: dict[str, Any], response: dict[str, Any]) -> float | None:
    expected_platform = case.get("platform_filter")
    if not expected_platform or expected_platform == "All":
        return None

    sources = response.get("sources", [])
    if not sources:
        return None  # no sources to check for non-knowledge cases

    allowed = {expected_platform, "All", ""}
    return 1.0 if all(source.get("platform", "") in allowed for source in sources) else 0.0


def fallback_correct(case: dict[str, Any], response: dict[str, Any]) -> float:
    return 1.0 if bool(response.get("fallback", False)) == bool(case.get("expected_fallback", False)) else 0.0


def run_config() -> dict[str, Any]:
    """Return current retrieval/QA configuration from config.py."""
    return {
        "chunking": {
            "strategy": "recursive_html_section_chunking",
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
        },
        "retrieval": {
            "top_k": DEFAULT_TOP_K,
            "vector_store": "chroma",
            "embedding_model": EMBEDDING_MODEL or "text-embedding-ada-002",
            "platform_filter_enabled": True,
            "metadata_fields": [
                "source_url",
                "article_title",
                "category",
                "platform",
                "section_heading",
            ],
        },
        "qa": {
            "fallback_enabled": True,
            "weak_evidence_check_enabled": True,
        },
        "actions": {
            "create_ticket": "multi-turn, 4 params",
            "check_ticket": "single-turn, ticket_id lookup",
            "device_transfer": "multi-turn, 3 params",
        },
        "guardrails": {
            "unsafe_pattern_check": True,
            "prompt_injection_check": True,
            "pii_redaction": True,
        },
    }


def evaluate_case(agent: SupportAgent, case: dict[str, Any]) -> dict[str, Any]:
    platform_filter = case.get("platform_filter")
    if platform_filter == "All":
        platform_filter = None

    response = agent.chat(case["query"], platform_filter=platform_filter)
    sources = response.get("sources", [])
    source_match, _ = source_match_score(
        case.get("gold_source_articles", []),
        sources,
    )
    relevancy = answer_relevancy(case, response)
    factual = factual_correctness(case, response, source_match)
    accuracy = answer_accuracy(case, response, relevancy, factual, source_match)
    platform_success = platform_filter_success(case, response)

    scores = {
        "retrieval_hit": retrieval_hit_score(case.get("gold_source_articles", []), sources),
        "source_match": round(source_match, 4),
        "answer_relevancy": relevancy,
        "factual_correctness": factual,
        "answer_accuracy": accuracy,
        "fallback": 1.0 if response.get("fallback", False) else 0.0,
        "fallback_correct": fallback_correct(case, response),
        "intent_correct": intent_correct_score(case, response),
        "action_correct": action_correct_score(case, response),
        "guardrail_correct": guardrail_correct_score(case, response),
        "platform_filter_success": platform_success,
    }

    # Add pending_action check for multi-turn actions
    pa_score = pending_action_score(case, response)
    if pa_score is not None:
        scores["pending_action_correct"] = pa_score

    return {
        "id": case["id"],
        "type": case.get("type", "knowledge"),
        "topic": case.get("expected_topic"),
        "actual_intent": response.get("intent"),
        "source_ids": source_ids(sources),
        "scores": scores,
    }


def average_score(results: list[dict[str, Any]], metric: str) -> float | None:
    values = [
        result["scores"][metric]
        for result in results
        if result["scores"].get(metric) is not None
    ]
    return round(mean(values), 4) if values else None


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    # Overall summary
    summary = {
        "num_cases": len(results),
        "retrieval_hit": average_score(results, "retrieval_hit"),
        "source_match": average_score(results, "source_match"),
        "answer_relevancy": average_score(results, "answer_relevancy"),
        "factual_correctness": average_score(results, "factual_correctness"),
        "answer_accuracy": average_score(results, "answer_accuracy"),
        "fallback_rate": average_score(results, "fallback"),
        "fallback_correct": average_score(results, "fallback_correct"),
        "intent_correct": average_score(results, "intent_correct"),
        "action_correct": average_score(results, "action_correct"),
        "guardrail_correct": average_score(results, "guardrail_correct"),
        "platform_filter_success": average_score(results, "platform_filter_success"),
    }

    # Per-type breakdown
    type_groups: dict[str, list] = {}
    for r in results:
        t = r.get("type", "knowledge")
        type_groups.setdefault(t, []).append(r)

    breakdown = {}
    for t, group in type_groups.items():
        breakdown[t] = {
            "count": len(group),
            "answer_accuracy": average_score(group, "answer_accuracy"),
            "intent_correct": average_score(group, "intent_correct"),
        }
    summary["by_type"] = breakdown

    return summary


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def write_eval_run(run: dict[str, Any], output_path: Path) -> None:
    if output_path.exists():
        results_file = load_json(output_path)
    else:
        results_file = {
            "metadata": {
                "name": "Signal Support Agent Evaluation Results",
                "test_cases_file": "data/test_cases.json",
            },
            "runs": [],
        }

    results_file.setdefault("runs", []).append(run)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results_file, f, indent=2, ensure_ascii=False)
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the Signal Support Agent")
    parser.add_argument("--test-cases", type=Path, default=TEST_CASES_FILE)
    parser.add_argument("--output", type=Path, default=EVAL_RESULTS_FILE)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-write", action="store_true", help="Print results without updating eval_results.json")
    args = parser.parse_args()

    test_cases_file = args.test_cases
    output_file = args.output

    test_data = load_json(test_cases_file)
    test_cases = test_data.get("test_cases", [])
    if args.limit:
        test_cases = test_cases[: args.limit]

    agent = SupportAgent()
    case_results = []
    for case in test_cases:
        print(f"Evaluating {case['id']}: {case['query']}")
        case_results.append(evaluate_case(agent, case))

    run = {
        "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "test_cases_file": str(test_cases_file.relative_to(ROOT_DIR)),
        "config": run_config(),
        "summary": summarize(case_results),
        "cases": case_results,
    }

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(json.dumps(run["summary"], indent=2, ensure_ascii=False))

    print("\n" + "-" * 60)
    print("PER-CASE RESULTS")
    print("-" * 60)
    for cr in case_results:
        status = "PASS" if cr["scores"]["answer_accuracy"] >= 0.5 else "FAIL"
        print(
            f"  [{status}] {cr['id']} "
            f"(type={cr['type']}, intent={cr['actual_intent']}, "
            f"accuracy={cr['scores']['answer_accuracy']:.2f})"
        )

    if not args.no_write:
        write_eval_run(run, output_file)
        print(f"\nWrote evaluation run to {output_file}")


if __name__ == "__main__":
    main()
