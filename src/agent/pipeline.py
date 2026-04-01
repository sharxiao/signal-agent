"""
this file features the end-to-end ingestion pipeline.

in order, these steps are run...
     1. fetch articles + sections from zendesk api
     2. clean html and chunk text
     3. embed chunks and store in chromadb
"""

import argparse
import logging
 
from sentence_transformers import SentenceTransformer
 
# import relevant constants and functions
from src.agent.config import (
    EMBEDDING_MODEL,
    RAW_ARTICLES_FILE,
)
from src.agent.ingest import (
    fetch_all_articles,
    fetch_sections,
    save_raw_data,
    load_raw_articles,
)
from src.agent.chunker import build_chunks, save_chunks, load_chunks
from src.agent.embedder import (
    get_chroma_client,
    get_or_create_collection,
    embed_and_store,
    query_collection,
)
 
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)
 
 
def run_pipeline(skip_ingest: bool = False, reset_db: bool = False) -> None:
    """
    Run the full ingestion pipeline.
 
    Args:
        skip_ingest: If True, load raw data from disk instead of calling API.
        reset_db:    If True, delete and rebuild the ChromaDB collection.
    """
 
    # step 1: ingest
    if skip_ingest and RAW_ARTICLES_FILE.exists():
        log.info("── Step 1: Loading raw data from disk (--skip-ingest) ──")
        articles, section_map = load_raw_articles()
    else:
        log.info("Step 1: Fetching articles from Zendesk API")
        articles    = fetch_all_articles()
        section_map = fetch_sections()
        save_raw_data(articles, section_map)
 
    log.info(f"   Articles: {len(articles)} | Sections: {len(section_map)}")
 
    # step 2: chunk
    log.info("Step 2: Cleaning HTML and chunking text")
    chunks = build_chunks(articles, section_map)
    save_chunks(chunks)
    log.info(f"   Chunks: {len(chunks)}")
 
    # step 3: embed + store
    log.info("Step 3: Embedding and storing in ChromaDB")
    model      = SentenceTransformer(EMBEDDING_MODEL)
    client     = get_chroma_client()
    collection = get_or_create_collection(client, reset=reset_db)
 
    # skip embedding if collection already populated and not resetting
    if collection.count() > 0 and not reset_db:
        log.info(
            f"   Collection already has {collection.count()} chunks. "
            "Use --reset-db to rebuild."
        )
    else:
        embed_and_store(chunks, collection, model)
 
    # fin
    log.info("Pipeline complete")
    log.info(f"   ChromaDB collection: {collection.count()} chunks")
 
    # test
    log.info("\Quick test — querying: 'notifications not working android'")
    results = query_collection(
        collection, model, "notifications not working android", n_results=3
    )
    for i, r in enumerate(results, 1):
        log.info(
            f"  [{i}] score={r['score']:.3f} | {r['article_title']} "
            f"| platform={r['platform']}"
        )
 
 
def get_retriever():
    """
    Return a (collection, model) tuple for use by other modules
    (retriever.py, router.py, etc.).
 
    Example:
        from src.agent.pipeline import get_retriever
        from src.agent.embedder import query_collection
 
        collection, model = get_retriever()
        results = query_collection(collection, model, "how to backup")
    """
    model      = SentenceTransformer(EMBEDDING_MODEL)
    client     = get_chroma_client()
    collection = get_or_create_collection(client, reset=False)
    return collection, model
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Signal RAG ingestion pipeline")
    parser.add_argument(
        "--skip-ingest",
        action="store_true",
        help="Load raw data from disk instead of calling the Zendesk API",
    )
    parser.add_argument(
        "--reset-db",
        action="store_true",
        help="Delete and rebuild the ChromaDB collection from scratch",
    )
    args = parser.parse_args()
    run_pipeline(skip_ingest=args.skip_ingest, reset_db=args.reset_db)