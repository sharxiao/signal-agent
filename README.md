# Signal_Agent

This is the final deliverable for RSM8430

## Project Structure

```
signal-support-agent/
├── src/
│   └── agent/
│       ├── __init__.py
│       ├── config.py          # Central configuration for paths, model names, & chunking params
│       ├── ingest.py          # Fetch articles and sections from Zendesk API
│       ├── chunker.py         # Clean HTML, detect platform, & chunk text
│       ├── embedder.py        # Embed chunks + store in ChromaDB
│       ├── retriever.py       # Query ChromaDB
│       ├── pipeline.py        # Full ingestion pipeline (calls ingest -> chunk -> embed)
│       ├── router.py          # Intent classification
│       ├── actions.py         # Mock actions
│       └── guardrails.py      # Safety layer
├── data/
│   ├── raw/                   # Raw JSON from Zendesk API
│   ├── chunks/                # Processed chunks as JSON
│   └── chromadb/              # ChromaDB persistent storage
├── tests/
│   └── test_pipeline.py
├── app.py                     # Streamlit UI
├── requirements.txt
└── README.md                  # This file
```

## Setup

1. Clone the repo and navigate to the project root.
2. Create and activate a virtual environment:

```bash
   python -m venv venv
   source venv/bin/activate
```

3. Install dependencies:

```bash
   pip install -r requirements.txt
```

## Running the pipeline

First, replace the API key with your own student number in `config.py`.
Then, build the knowledge base (fetch from Zendesk API, chunk, embed, store in ChromaDB):

```bash
   python -m src.agent.pipeline
```

If the raw data has already been fetched (skip API call):

```bash
   python -m src.agent.pipeline --skip-ingest
```

Wipe and rebuild ChromaDB from scratch:

```bash
   python -m src.agent.pipeline --reset-db
```

## Configuration

If you need to change any configuration settings, you can find all configurable parameters in `src/agent/config.py`:

- chunk size and overlap
- embedding model name
- ChromaDB collection name
- LLM endpoint

## Interface

To query the knowledge base:

```python
   from src.agent.pipeline import get_retriever
   from src.agent.embedder import query_collection

   collection, embedding_client = get_retriever()
   results = query_collection(collection, embedding_client, "how to backup")

   # each result will have text, source_url, article_title, category, platform,
   # section_heading, and score
```

**Chunk metadata fields:**
| Field | Type | Description |
|-------|------|-------------|
| `chunk_id` | str | Unique ID e.g. `360007059212-003` |
| `text` | str | Plain text content |
| `article_title` | str | Title of source article |
| `source_url` | str | Public Signal help centre URL |
| `category` | str | e.g. `Troubleshooting`, `Security` |
| `platform` | str | `Android`, `iOS`, `Desktop`, or `All` |
| `section_heading` | str | Nearest heading above this chunk |
| `score` | float | Cosine similarity (0–1, higher is better) |

## Notes (delete later)

- `data/` is gitignored - run the pipeline locally to regenerate it
