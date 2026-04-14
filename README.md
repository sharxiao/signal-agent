# Signal Support Agent

A conversational support agent for Signal Messenger, built with RAG (Retrieval-Augmented Generation), multi-turn action flows, and safety guardrails.

**Course:** RSM8430H — Applications of Large Language Models  
**Team:** 8

---

## Overview

This agent answers Signal support questions using official help center documentation, creates support tickets, initiates device transfers, and handles safety/guardrail scenarios — all through a Streamlit chat interface.

### Key Capabilities

- **Knowledge QA** — Answers questions grounded in ~150 Signal help center articles with source attribution
- **Intent Classification** — Two-tier routing: fast regex matching + LLM fallback for ambiguous queries
- **3 Mock Actions** — Create Support Ticket (multi-turn, 4 params), Check Ticket Status, Device Transfer Request (multi-turn, 3 params)
- **Multi-turn Conversations** — Sequential parameter collection with validation, cancel detection, and parameter pre-extraction from initial message
- **Persistent State** — Tickets and transfers stored in JSON, survive across sessions
- **Per-User History Isolation** — Ticket and transfer history is scoped to a user session code rather than a shared global history
- **Session Resume Support** — Users can start a new session or resume a previous one with a saved session code
- **Conversation Memory** — Chat history passed to the LLM for contextual follow-ups
- **Guardrails** — Input guardrails (unsafe requests, prompt injection, manipulation tricks) + output guardrails (hallucination detection, system prompt leakage, source quality checks)
- **Semantic Retrieval** — Embedding-based search with ChromaDB over Signal help center chunks
- **Robust Fallbacks** — Graceful fallback when retrieval evidence is weak or the knowledge base is temporarily unavailable
- **Password-Protected Public Deployment** — Streamlit app can be deployed to a public URL with password protection for demo access
- **Evaluation** — Automated scoring across knowledge, action, guardrail, routing, edge-case, and error-handling categories

---

## Architecture

```
User Message
    │
    ▼
┌─────────────────┐
│  Input Guardrails │ ── Block unsafe / injection / tricks
└────────┬────────┘
         │
    ▼ (if allowed)
┌─────────────────┐
│  Multi-turn Check │ ── Continue pending action? Cancel detection?
└────────┬────────┘
         │
    ▼ (if no pending action)
┌─────────────────┐
│  Intent Router    │ ── Regex → LLM fallback
└────────┬────────┘
         │
    ┌────┴────┬──────────┬──────────┐
    ▼         ▼          ▼          ▼
 Greeting   Action    Knowledge   Off-topic
            │          │
            ▼          ▼
       ┌─────────┐  ┌──────────────┐
       │ Actions  │ │ Semantic RAG │
       │ (stateful│ │ (ChromaDB +  │
       │  JSON)   │ │  embeddings) │
       └─────────┘  └──────┬───────┘
                           ▼
                    ┌──────────────┐
                    │  LLM (Qwen3) │
                    │  Grounded QA │
                    └──────┬───────┘
                           ▼
                    ┌────────────────┐
                    │Output Guardrail│
                    └──────┬─────────┘
                           ▼
                      Response + Sources
```

---

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.10+ |
| Agent Framework | Custom (no LangChain) |
| Vector Database | ChromaDB (persistent, cosine similarity) |
| Retrieval | Semantic vector search over embedded chunks |
| LLM | Qwen3-30B-A3B (course endpoint) |
| Embeddings | Course embedding endpoint |
| UI | Streamlit |
| Data Source | Signal Help Center (Zendesk API) |
| State Persistence | JSON file (`data/action_store.json`) |
| Session Isolation | Streamlit session state + resumable session code |

---

## Project Structure

```
signal-support-agent/
├── .streamlit/
│   └── secrets.toml.example   # Template for local / Streamlit Cloud secrets
├── src/agent/
│   ├── config.py          # Paths, model names, chunking params
│   ├── ingest.py          # Fetch articles from Zendesk API
│   ├── chunker.py         # Clean HTML, detect platform, chunk text
│   ├── embedder.py        # Embed chunks, store them in ChromaDB, and run semantic retrieval
│   ├── pipeline.py        # End-to-end ingestion pipeline
│   ├── router.py          # Two-tier intent classification (regex + LLM)
│   ├── actions.py         # Stateful mock actions with multi-turn param collection
│   ├── guardrails.py      # Input + output safety guardrails
│   ├── qa.py              # Grounded QA with source attribution
│   └── conversation.py    # Orchestration layer with memory + state machine
├── data/
│   ├── raw/               # Raw JSON from Zendesk API
│   ├── chunks/            # Processed text chunks
│   ├── chromadb/          # ChromaDB persistent storage
│   ├── action_store.json  # Persistent ticket/transfer data
│   ├── test_cases.json    # 17 evaluation test cases
│   └── eval_results.json  # Evaluation run history
├── app.py                 # Streamlit chat UI
├── eval.py                # Evaluation framework
├── ID.txt                 # API authentication (not committed)
├── requirements.txt       # Python dependencies
└── README.md
```

---

## Setup & Installation

### Prerequisites

- Python 3.10+
- Course API access (student ID in `ID.txt`)

### Steps

```bash
## Setup & Installation

### Prerequisites

- Python 3.10+
- Course API access

### Steps

```bash
## Setup & Installation

### Prerequisites

- Python 3.10+
- Course API access

### Steps

```bash
# 1. Clone the repository
git clone <repo-url>
cd Signal_Agent

# 2. Install dependencies
pip install -r requirements.txt

# 3. Create ID.txt with your student ID
echo "YOUR_STUDENT_ID" > ID.txt

# 4. Create a local Streamlit secrets file
mkdir -p .streamlit
cp .streamlit/secrets.toml.example .streamlit/secrets.toml

# 5. Run the ingestion pipeline (fetches articles, chunks, embeds)
python -m src.agent.pipeline

# 6. Launch the Streamlit app
streamlit run app.py
```

### Skip API fetch (if raw data already exists)

```bash
python -m src.agent.pipeline --skip-ingest
```

### Rebuild ChromaDB from scratch

```bash
python -m src.agent.pipeline --skip-ingest --reset-db
```

---

## Usage

### Streamlit UI

The chat interface provides:

- **Quick action buttons** — Create Ticket, Check Ticket, Transfer Device
- **Multi-turn progress bar** — Visual indicator of parameter collection progress
- **Confidence badges** — High / Medium / Low confidence on each answer
- **Source attribution** — Expandable sources with article links and relevance scores
- **Sidebar** — Ticket and transfer history scoped to the current user session
- **Session codes** — Users can start a new session or resume a previous one using a saved session code
- **Password gate** — Optional password protection for public deployment

### CLI

```bash
# Interactive chat
python -m src.agent.conversation

# With JSON output
python -m src.agent.conversation --json
```
---

## Demo & Deployment

- **Backup demo video:** https://www.dropbox.com/scl/fi/uyajtjdrqb266u2zzadnb/Final-Project-Backup-Demo-Team-8.mp4?rlkey=jg8fhbmyavaugh15lt4rr8k0b&st=bqry86d2&dl=0
- **Public Streamlit app:** https://signalagent-zjweeyqsjssqjjwqufnxz4.streamlit.app/

---

## Evaluation

### Running the evaluation

```bash
# Run all all test cases
python eval.py

# Run first N cases only
python eval.py --limit 5

# Print results without saving
python eval.py --no-write
```

### Test Case Coverage

The evaluation set covers:

| Category | What's Tested |
|----------|---------------|
| Knowledge QA | Transfer, backup, verification, PIN, desktop linking, notifications, account deletion, safety number, disappearing messages, group chat, block user |
| Actions | Create ticket, check ticket, device transfer, alternative ticket phrasing, ticket ID extraction |
| Guardrails | Unsafe requests, prompt injection, manipulation tricks |
| Routing | Off-topic queries, ambiguous queries, greeting, negation handling |
| Edge Cases | Nonexistent features |
| Error Handling | Empty input, ticket not found, weak retrieval fallback |

### Metrics

- **answer_relevancy** — Token overlap between answer and expected content
- **factual_correctness** — Grounding quality + source matching
- **answer_accuracy** — Aggregate of relevancy, factual correctness, and intent
- **intent_correct** — Was the intent routed correctly?
- **action_correct** — Was the right action triggered (or not triggered)?
- **guardrail_correct** — Was the message correctly blocked/allowed?

---

## Safety & Guardrails

### Input Guardrails

| Type | Example | Behavior |
|------|---------|----------|
| Unsafe request | "spy on someone's messages" | Blocked with specific explanation |
| Prompt injection | "ignore all previous instructions" | Blocked, explains it's a Signal support agent |
| Manipulation trick | "what word combines s and hit" | Blocked, redirects to Signal topics |
| PII redaction | Phone numbers, verification codes | Automatically redacted from input |

### Output Guardrails

- Hallucination detection (fabricated URLs, unsupported claims)
- System prompt leakage detection
- Source quality validation (low-score citations, duplicate sources)
- Content-source overlap verification

---

## Known Limitations

- **LLM output format** — The Qwen3 model occasionally outputs label:value format instead of JSON, requiring robust parsing
- **iOS backup** — Signal does not support standalone backups on iOS; the agent correctly reports this but users may find it unhelpful
- **Multi-question handling** — When asking multiple questions, retrieval is optimized for the combined query which may not be ideal for each sub-question individually
- **Conversation memory** — Limited to last 10 turns to avoid token overflow; very long conversations may lose early context
- **Public deployment session model** — The app uses lightweight session codes rather than a full authentication system, which is appropriate for demo access but not a production-grade account system

---

## Team Responsibilities

| Member | Responsibilities |
|--------|-----------------|
| Member 1 | Data ingestion, chunking, embedding pipeline |
| Member 2 | QA layer, grounded answering, source attribution |
| Member 3 | Intent routing, action system, multi-turn flows |
| Member 4 | Guardrails, safety, evaluation framework |
| Member 5 | Streamlit UI, integration, deployment |
