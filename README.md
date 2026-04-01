# Signal_Agent

This is the final deliverable for RSM8430

## Project Structure

```
signal-support-agent/
├── src/
│   └── agent/
│       ├── __init__.py
│       ├── config.py          # Paths, model names, chunking params
│       ├── ingest.py          # Fetch articles from Zendesk API
│       ├── chunker.py         # Clean HTML + chunk text
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
