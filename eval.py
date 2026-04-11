"""
Evaluate the Signal Support Agent on data/test_cases.json.

The metric names follow the Assignment 2/RAGAS-style evaluator:
- answer_relevancy
- factual_correctness
- answer_accuracy

This local evaluator keeps the implementation lightweight and deterministic:
it computes heuristic scores from each agent response, retrieved sources, and
the reference metadata in data/test_cases.json.

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
    Score how many expected source article names appear in retrieved source titles.
    Returns a score in [0, 1] and the matched gold source names.
    """
    if not gold_sources:
        return (1.0 if not sources else 0.0), []

    titles = source_titles(sources)
    matched = []
    for gold in gold_sources:
        gold_norm = normalize_text(gold)
        if any(gold_norm in title or title in gold_norm for title in titles):
            matched.append(gold)

    return len(matched) / len(gold_sources), matched


def retrieval_hit_score(gold_sources: list[str], sources: list[dict[str, Any]]) -> float:
    if gold_sources:
        return 1.0 if sources else 0.0
    return 1.0 if not sources else 0.0


def answer_relevancy(case: dict[str, Any], response: dict[str, Any]) -> float:
    """
    Heuristic answer relevancy.
    Mirrors the RAGAS answer_relevancy idea by checking whether the response
    covers tokens from the question, expected topic, expected source titles,
    and human gold-answer notes.
    """
    reference = " ".join(
        [
            case.get("query", ""),
            case.get("expected_topic", ""),
            case.get("gold_answer_notes", ""),
            " ".join(case.get("gold_source_articles", [])),
        ]
    )
    return round(overlap_score(reference, response.get("answer", "")), 4)


def factual_correctness(
    case: dict[str, Any],
    response: dict[str, Any],
    source_match: float,
) -> float:
    """
    Heuristic factual correctness.
    Rewards grounded non-fallback answers with matched sources; rewards expected
    fallback/guardrail behavior when the test case asks for fallback.
    """
    answer = normalize_text(response.get("answer", ""))
    expected_fallback = bool(case.get("expected_fallback", False))
    actual_fallback = bool(response.get("fallback", False))
    grounded = bool(response.get("grounded", False))
    has_sources = bool(response.get("sources", []))
    has_action = bool(response.get("action"))

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
    Aggregate score similar to the Assignment 2 AnswerAccuracy metric.
    Combines relevancy, factual correctness, source match, and fallback behavior.
    """
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
        return 0.0

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
        "platform_filter_success": platform_success,
    }

    return {
        "id": case["id"],
        "topic": case.get("expected_topic"),
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
    return {
        "num_cases": len(results),
        "retrieval_hit": average_score(results, "retrieval_hit"),
        "source_match": average_score(results, "source_match"),
        "answer_relevancy": average_score(results, "answer_relevancy"),
        "factual_correctness": average_score(results, "factual_correctness"),
        "answer_accuracy": average_score(results, "answer_accuracy"),
        "fallback_rate": average_score(results, "fallback"),
        "fallback_correct": average_score(results, "fallback_correct"),
        "platform_filter_success": average_score(results, "platform_filter_success"),
    }


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

    print(json.dumps(run["summary"], indent=2, ensure_ascii=False))
    if not args.no_write:
        write_eval_run(run, output_file)
        print(f"Wrote evaluation run to {output_file}")


if __name__ == "__main__":
    main()
