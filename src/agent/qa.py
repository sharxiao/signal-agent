"""
qa.py

Knowledge QA layer for the Signal Support Agent.

Responsibilities:
1. Retrieve top-k chunks from ChromaDB
2. Build a grounded prompt using only retrieved Signal help content
3. Call the LLM endpoint
4. Return a structured answer with citation-ready sources
5. Fallback safely when evidence is weak or missing
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

import requests

from src.agent.config import (
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MAX_TOKENS,
    LLM_MODEL,
    DEFAULT_TOP_K,
)
from src.agent.embedder import query_collection
from src.agent.pipeline import get_retriever

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# Retrieval quality thresholds — tuned after eval runs.
# Lowered from 0.30/0.22 to reduce false fallbacks on valid but
# cross-platform queries (e.g. iOS backup).

MIN_TOP_SCORE = 0.25
MIN_AVG_TOP3_SCORE = 0.18
REQUEST_TIMEOUT = 60


def _normalize_query(query: str) -> str:
    """Basic cleanup for user input."""
    return re.sub(r"\s+", " ", query.strip())


def _dedupe_sources(results: list[dict]) -> list[dict]:
    """
    Deduplicate retrieved results at the source level for UI display.
    Keeps the highest-scoring occurrence of each (title, url, section).
    """
    seen: dict[tuple[str, str, str], dict] = {}

    for r in results:
        key = (
            r.get("article_title", "").strip(),
            r.get("source_url", "").strip(),
            r.get("section_heading", "").strip(),
        )
        if key not in seen or r.get("score", 0.0) > seen[key].get("score", 0.0):
            seen[key] = {
                "title": r.get("article_title", ""),
                "url": r.get("source_url", ""),
                "section_heading": r.get("section_heading", ""),
                "platform": r.get("platform", ""),
                "category": r.get("category", ""),
                "score": r.get("score", 0.0),
            }

    deduped = list(seen.values())
    deduped.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    return deduped


def _build_context(results: list[dict]) -> str:
    """
    Build retrieval context with stable source IDs.
    These source IDs are what the model should cite.
    """
    blocks = []
    for i, r in enumerate(results, start=1):
        source_id = f"S{i}"
        block = (
            f"[{source_id}]\n"
            f"Title: {r.get('article_title', '')}\n"
            f"URL: {r.get('source_url', '')}\n"
            f"Category: {r.get('category', '')}\n"
            f"Platform: {r.get('platform', '')}\n"
            f"Section: {r.get('section_heading', '')}\n"
            f"Score: {r.get('score', 0.0)}\n"
            f"Content:\n{r.get('text', '').strip()}\n"
        )
        blocks.append(block)
    return "\n---\n".join(blocks)


def _build_messages(query: str, context: str, conversation_history: str = "") -> list[dict[str, str]]:
    """
    OpenAI-compatible chat messages.
    Forces grounded answering and JSON-only output.
    Includes conversation history when available.
    """
    system_prompt = (
        "You are a support QA assistant for Signal.\n"
        "Answer ONLY using the provided official Signal help documentation.\n"
        "Do NOT use outside knowledge.\n"
        "If the documentation is insufficient, say so clearly.\n"
        "Do NOT guess, infer hidden steps, or invent unsupported claims.\n"
        "Keep the answer concise, practical, and support-oriented.\n"
        "If conversation history is provided, use it to understand context "
        "and resolve references like 'it', 'that', 'the same thing', etc.\n"
        "Return JSON only with the following schema:\n"
        "{\n"
        '  "answer": string,\n'
        '  "grounded": boolean,\n'
        '  "fallback": boolean,\n'
        '  "citations": [string],\n'
        '  "reason_if_fallback": string\n'
        "}\n"
        "Rules for citations:\n"
        "- Use only source IDs like S1, S2, S3 from the provided context.\n"
        "- Include only sources that directly support the answer.\n"
        "- If fallback is true, citations can be an empty list.\n"
    )

    history_block = ""
    if conversation_history:
        history_block = f"Previous conversation context:\n{conversation_history}\n\n"

    user_prompt = (
        f"{history_block}"
        f"User question:\n{query}\n\n"
        f"Retrieved Signal help documentation:\n{context}\n\n"
        "Now answer the question using only the retrieved documentation. "
        "Use the conversation context if it helps clarify what the user is asking about."
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _extract_json(text: str) -> dict[str, Any]:
    """
    Best-effort JSON extraction.
    Handles models that may wrap JSON in markdown fences or extra text.
    """
    if text is None:
        return {
            "answer": "",
            "grounded": False,
            "fallback": True,
            "citations": [],
            "reason_if_fallback": "Model returned empty content.",
        }

    text = text.strip()

    # Remove markdown fences if present
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    # Try direct parse first
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # Try to locate first JSON object in the text
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    # Hard fallback
    return {
        "answer": text,
        "grounded": False,
        "fallback": True,
        "citations": [],
        "reason_if_fallback": "Model output was not valid JSON.",
    }


def _call_llm(messages: list[dict[str, str]]) -> dict[str, Any]:
    base = LLM_BASE_URL.rstrip("/")
    url = f"{base}/v1/chat/completions"

    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "max_tokens": LLM_MAX_TOKENS,
        "temperature": 0,
    }

    headers = {}
    if LLM_API_KEY and LLM_API_KEY != "replace_me":
        headers["Authorization"] = f"Bearer {LLM_API_KEY}"

    response = requests.post(
        url,
        json=payload,
        headers=headers,
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    data = response.json()
    choices = data.get("choices", [])
    if not choices:
        raise ValueError(f"No choices returned from LLM: {data}")

    message = choices[0].get("message", {}) or {}

    content = message.get("content")

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        content = "\n".join(parts).strip()

    if not content:
        content = message.get("reasoning_content")

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
            else:
                parts.append(str(item))
        content = "\n".join(parts).strip()

    if not content:
        raise ValueError(f"LLM returned empty content. Full response: {data}")

    try:
        return _extract_json(content)
    except Exception:
        return {
            "answer": content,
            "grounded": False,
            "fallback": True,
            "citations": [],
            "reason_if_fallback": "Model returned non-JSON text.",
        }


def _evidence_is_weak(results: list[dict]) -> bool:
    """
    Very simple first-pass retrieval quality heuristic.
    Conservative on purpose for Stage 1.
    """
    if not results:
        return True

    top_score = results[0].get("score", 0.0)
    top3 = results[:3]
    avg_top3 = sum(r.get("score", 0.0) for r in top3) / len(top3)

    return top_score < MIN_TOP_SCORE or avg_top3 < MIN_AVG_TOP3_SCORE


def answer_knowledge_query(
    query: str,
    n_results: int = DEFAULT_TOP_K,
    platform_filter: Optional[str] = None,
    category_filter: Optional[str] = None,
    conversation_history: str = "",
) -> dict[str, Any]:
    """
    Main entry point for the QA layer.

    Returns:
    {
        "query": str,
        "answer": str,
        "grounded": bool,
        "fallback": bool,
        "reason_if_fallback": str,
        "citations": [str],
        "sources": [...],
        "retrieved_chunks": [...],
    }
    """
    query = _normalize_query(query)
    if not query:
        return {
            "query": query,
            "answer": "Please provide a question.",
            "grounded": False,
            "fallback": True,
            "reason_if_fallback": "Empty query.",
            "citations": [],
            "sources": [],
            "retrieved_chunks": [],
        }

    collection, model = get_retriever()
    results = query_collection(
        collection,
        model,
        query,
        n_results=n_results,
        platform_filter=platform_filter,
        category_filter=category_filter,
    )

    if _evidence_is_weak(results):
        sources = _dedupe_sources(results)
        return {
            "query": query,
            "answer": (
                "I couldn't find enough relevant information in the official Signal "
                "help documentation to answer that confidently."
            ),
            "grounded": False,
            "fallback": True,
            "reason_if_fallback": "Retrieved evidence was weak or insufficient.",
            "citations": [],
            "sources": sources,
            "retrieved_chunks": results,
        }

    context = _build_context(results)
    messages = _build_messages(query, context, conversation_history)

    try:
        llm_output = _call_llm(messages)
    except requests.RequestException as e:
        log.exception("LLM request failed")
        return {
            "query": query,
            "answer": (
                "I found relevant Signal help content, but the language model request failed."
            ),
            "grounded": False,
            "fallback": True,
            "reason_if_fallback": f"LLM request error: {e}",
            "citations": [],
            "sources": _dedupe_sources(results),
            "retrieved_chunks": results,
        }
    except Exception as e:
        log.exception("Unexpected QA generation failure")
        return {
            "query": query,
            "answer": "Something went wrong while generating the answer.",
            "grounded": False,
            "fallback": True,
            "reason_if_fallback": f"Unexpected generation error: {e}",
            "citations": [],
            "sources": _dedupe_sources(results),
            "retrieved_chunks": results,
        }

    # Normalize model output
    answer = str(llm_output.get("answer", "")).strip()
    grounded = bool(llm_output.get("grounded", False))
    fallback = bool(llm_output.get("fallback", False))
    citations = llm_output.get("citations", [])
    reason_if_fallback = str(llm_output.get("reason_if_fallback", "")).strip()

    if not isinstance(citations, list):
        citations = []

    # Keep only valid source IDs that actually exist
    valid_ids = {f"S{i}" for i in range(1, len(results) + 1)}
    citations = [c for c in citations if c in valid_ids]

    # Build citation-ready source objects aligned with S1, S2, ...
    sources = []
    for i, r in enumerate(results, start=1):
        source_id = f"S{i}"
        if source_id in citations:
            sources.append(
                {
                    "source_id": source_id,
                    "title": r.get("article_title", ""),
                    "url": r.get("source_url", ""),
                    "section_heading": r.get("section_heading", ""),
                    "platform": r.get("platform", ""),
                    "category": r.get("category", ""),
                    "score": r.get("score", 0.0),
                }
            )

    # Defensive fallback if model returns empty answer
    if not answer:
        answer = (
            "I couldn't generate a reliable answer from the retrieved Signal help documentation."
        )
        grounded = False
        fallback = True
        if not reason_if_fallback:
            reason_if_fallback = "Model returned an empty answer."

    return {
        "query": query,
        "answer": answer,
        "grounded": grounded,
        "fallback": fallback,
        "reason_if_fallback": reason_if_fallback,
        "citations": citations,
        "sources": sources,
        "retrieved_chunks": results,
    }


if __name__ == "__main__":
    test_queries = [
        "How do I transfer Signal to a new phone?",
        "Can I back up messages on iPhone?",
        "Why am I not getting my verification code on Android?",
    ]

    for q in test_queries:
        print("\n" + "=" * 80)
        print("QUERY:", q)
        result = answer_knowledge_query(q)
        print(json.dumps(result, indent=2, ensure_ascii=False))
