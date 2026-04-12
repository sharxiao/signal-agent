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
        "\n"
        "IMPORTANT rules:\n"
        "- If the user asks MULTIPLE questions in one message, address ALL of them.\n"
        "- If the retrieved documentation does not contain a clear answer for a "
        "sub-question, explicitly say you don't have that information rather than guessing.\n"
        "- If you are not confident in your answer, set grounded to false and "
        "fallback to true.\n"
        "- NEVER fabricate URLs, steps, or feature names not present in the sources.\n"
        "- If the user's question is VAGUE or UNCLEAR (e.g. 'it's not working', "
        "'I have a problem'), ask a clarifying follow-up question to understand "
        "what specific Signal feature or issue they need help with. Include the "
        "follow-up question in your answer.\n"
        "- If the retrieved sources don't seem relevant to the question, say so "
        "and ask the user to rephrase rather than forcing an answer from "
        "unrelated content.\n"
        "\n"
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
    Handles models that may:
    - Wrap JSON in markdown fences
    - Output text before/after the JSON
    - Output label:value format instead of JSON (e.g. "Answer: ...\nGrounded: true")
    - Include thinking/reasoning before the actual JSON
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

    # Try to locate the LAST JSON object in the text
    # (models often think/reason first, then output JSON at the end)
    json_candidates = list(re.finditer(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, flags=re.DOTALL))
    for match in reversed(json_candidates):
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, dict) and "answer" in parsed:
                return parsed
        except json.JSONDecodeError:
            continue

    # Try any JSON object (less strict)
    for match in reversed(json_candidates):
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue

    # Fallback: try to parse label:value format
    # e.g. "Answer: some text\nGrounded: true\nFallback: false\nCitations: [S1]"
    label_match = re.search(
        r"(?:^|\n)\s*[Aa]nswer\s*:\s*(.+?)(?=\n\s*[Gg]rounded\s*:|\n\s*[Ff]allback\s*:|\Z)",
        text,
        flags=re.DOTALL,
    )
    if label_match:
        answer_text = label_match.group(1).strip()
        grounded = bool(re.search(r"[Gg]rounded\s*:\s*true", text))
        fallback = bool(re.search(r"[Ff]allback\s*:\s*true", text))
        citations_match = re.search(r"[Cc]itations?\s*:\s*\[([^\]]*)\]", text)
        citations = []
        if citations_match:
            citations = [c.strip().strip('"\'') for c in citations_match.group(1).split(",") if c.strip()]
        reason_match = re.search(r"[Rr]eason[_ ]?if[_ ]?fallback\s*:\s*(.+?)(?:\n|$)", text)
        reason = reason_match.group(1).strip() if reason_match else ""

        return {
            "answer": answer_text,
            "grounded": grounded,
            "fallback": fallback,
            "citations": citations,
            "reason_if_fallback": reason,
        }

    # Hard fallback — strip any JSON-like metadata from the text
    # Remove lines that look like metadata fields
    cleaned_lines = []
    for line in text.split("\n"):
        line_stripped = line.strip().lower()
        if any(line_stripped.startswith(prefix) for prefix in [
            "grounded:", "fallback:", "citations:", "reason_if_fallback:",
            "reason if fallback:", "**grounded", "**fallback", "**citations",
            "**reason",
        ]):
            continue
        # Also skip lines that are just "Answer:" headers
        if re.match(r"^\*?\*?answer\*?\*?\s*:?\s*$", line_stripped):
            continue
        cleaned_lines.append(line)

    cleaned = "\n".join(cleaned_lines).strip()

    return {
        "answer": cleaned if cleaned else text,
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


def _check_output_quality(
    answer: str,
    citations: list[str],
    results: list[dict],
    grounded: bool,
) -> str:
    """
    Post-generation guardrail. Returns a non-empty reason string if the
    answer should be blocked, empty string if it's OK.

    Checks for:
    1. Answer claims to be grounded but has no citations
    2. Answer contains signs of hallucination (fabricated URLs, steps not in sources)
    3. Answer leaks system prompt or internal instructions
    4. All cited sources have very low retrieval scores
    5. Duplicate sources — all cited sources are identical content
    6. Answer content not supported by source content
    """
    answer_lower = answer.lower()

    # Check 1: Claims grounded but no citations
    if grounded and not citations:
        return "Claims grounded but provides no source citations."

    # Check 2: Hallucination signals
    hallucination_signals = [
        r"https?://(?!support\.signal\.org)\S+",  # URLs not from Signal help
    ]
    for pattern in hallucination_signals:
        if re.search(pattern, answer_lower):
            chunk_texts = " ".join(r.get("text", "") for r in results).lower()
            match = re.search(pattern, answer_lower)
            if match and match.group(0) not in chunk_texts:
                return "Possible hallucination: content not found in retrieved sources."

    # Check 3: System prompt leakage
    leak_patterns = [
        r"system\s*prompt",
        r"my\s+instructions\s+(are|say)",
        r"i\s+was\s+(told|instructed|programmed)\s+to",
        r"as\s+an?\s+ai\s+(language\s+)?model",
    ]
    for pattern in leak_patterns:
        if re.search(pattern, answer_lower):
            return "Possible system prompt leakage in response."

    # Check 4: All cited sources have very low scores
    if citations and results:
        cited_scores = []
        for i, r in enumerate(results, start=1):
            if f"S{i}" in citations:
                cited_scores.append(r.get("score", 0.0))
        if cited_scores and max(cited_scores) < 0.20:
            return "All cited sources have very low relevance scores."

    # Check 5: All sources are duplicates (same article title)
    if len(results) >= 2:
        titles = [r.get("article_title", "").strip().lower() for r in results if r.get("article_title")]
        if titles and len(set(titles)) == 1:
            # All sources from the same article — check if answer content
            # has reasonable overlap with the source content
            source_text = " ".join(r.get("text", "") for r in results).lower()
            answer_words = set(re.findall(r"[a-z]{4,}", answer_lower))
            source_words = set(re.findall(r"[a-z]{4,}", source_text))
            if answer_words and source_words:
                overlap = len(answer_words & source_words) / len(answer_words)
                if overlap < 0.15:
                    return "Answer content has very low overlap with source documents."

    return ""  # OK


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

    # --- OUTPUT GUARDRAIL: check for hallucination / low-quality answers ---
    guardrail_issue = _check_output_quality(answer, citations, results, grounded)
    if guardrail_issue:
        log.warning(f"Output guardrail triggered: {guardrail_issue}")
        answer = (
            "I found some related Signal help content, but I'm not confident "
            "enough in the answer to share it. Could you rephrase your question "
            "or provide more details so I can help you better?"
        )
        grounded = False
        fallback = True
        reason_if_fallback = f"Output guardrail: {guardrail_issue}"
        citations = []

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
