"""
this file embeds text chunks and stores them in ChromaDB.
ChromaDB is stored persistently at data/chromadb/
"""

import logging
from typing import Optional
 
import chromadb
from chromadb.config import Settings
from openai import OpenAI

 
# import relevant constant vars
from src.agent.config import (
    CHROMADB_DIR,
    CHROMA_COLLECTION_NAME,
    EMBEDDING_BASE_URL,
    EMBEDDING_API_KEY,
    EMBEDDING_MODEL,
)
from src.agent.chunker import load_chunks
 
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)
logging.getLogger("chromadb").setLevel(logging.CRITICAL)
 
# batch size for embedding to keep memory usage reasonable
EMBED_BATCH_SIZE = 32


def get_embedding_client() -> OpenAI:
    """Return an OpenAI client pointed at the course embedding endpoint."""
    return OpenAI(
        base_url=EMBEDDING_BASE_URL,
        api_key=EMBEDDING_API_KEY,
    )
 

def get_chroma_client() -> chromadb.PersistentClient:
    """Return a persistent ChromaDB client pointed at data/chromadb/."""
    return chromadb.PersistentClient(
        path=str(CHROMADB_DIR),
        settings=Settings(anonymized_telemetry=False),
    )
 
 
def get_or_create_collection(client: chromadb.PersistentClient, reset: bool = False) -> chromadb.Collection:
    """
    Get the ChromaDB collection, optionally resetting it.
    Set reset=True to rebuild the index from scratch.
    """
    if reset:
        try:
            client.delete_collection(CHROMA_COLLECTION_NAME)
            log.info(f"Deleted existing collection '{CHROMA_COLLECTION_NAME}'")
        except Exception:
            pass  # collection didn't exist yet
 
    collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},  # cosine similarity for retrieval
    )
    log.info(
        f"Collection '{CHROMA_COLLECTION_NAME}' ready. "
        f"Current count: {collection.count()}"
    )
    return collection
 

def embed_texts(texts: list[str], embedding_client: OpenAI) -> list[list[float]]:
    """
    Call the course embedding API and return a list of embedding vectors.
    One vector per input text.
    """
    kwargs = {"input": texts}
    if EMBEDDING_MODEL:
        kwargs["model"] = EMBEDDING_MODEL
    else:
        kwargs["model"] = "text-embedding-ada-002"  # OpenAI library requires this field
    response = embedding_client.embeddings.create(**kwargs)
    return [item.embedding for item in response.data]


def embed_and_store(chunks: list[dict], collection: chromadb.Collection, embedding_client: OpenAI) -> None:
    """
    Embed all chunks in batches and upsert them into ChromaDB.
    Uses upsert so re-running is safe (no duplicates).
    """
    total = len(chunks)
    log.info(f"Embedding {total} chunks in batches of {EMBED_BATCH_SIZE}...")
 
    for batch_start in range(0, total, EMBED_BATCH_SIZE):
        batch = chunks[batch_start : batch_start + EMBED_BATCH_SIZE]
 
        texts     = [c["text"] for c in batch]
        ids       = [c["chunk_id"] for c in batch]
        metadatas = [
            {
                "article_title":   c["article_title"],
                "source_url":      c["source_url"],
                "category":        c["category"],
                "platform":        c["platform"],
                "section_heading": c["section_heading"],
                "chunk_index":     c["chunk_index"],
                "word_count":      c["word_count"],
            }
            for c in batch
        ]
 
        embeddings = embed_texts(texts, embedding_client)
 
        collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )
 
        log.info(
            f"  Upserted batch {batch_start // EMBED_BATCH_SIZE + 1} "
            f"({batch_start + len(batch)}/{total})"
        )
 
    log.info(f"Done. Collection now contains {collection.count()} chunks.")
 
 
def query_collection(
    collection: chromadb.Collection,
    embedding_client: OpenAI,
    query: str,
    n_results: int = 5,
    platform_filter: Optional[str] = None,
    category_filter: Optional[str] = None,
) -> list[dict]:
    """
    Retrieve the top-n most relevant chunks for a query.
    Optionally filter by platform and/or category metadata.
 
    Returns a list of result dicts with keys:
        text, source_url, article_title, category, platform,
        section_heading, score
    """
    query_embedding = embed_texts([query], embedding_client)[0]
 
    # build optional metadata filter
    where = {}
    if platform_filter and platform_filter != "All":
        where["platform"] = {"$in": [platform_filter, "All"]}
    if category_filter:
        where["category"] = category_filter
 
    kwargs = {
        "query_embeddings": [query_embedding],
        "n_results": n_results,
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        kwargs["where"] = where
 
    results = collection.query(**kwargs)
 
    # flatten ChromaDB's nested response format
    output = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        output.append({
            "text":            doc,
            "source_url":      meta.get("source_url", ""),
            "article_title":   meta.get("article_title", ""),
            "category":        meta.get("category", ""),
            "platform":        meta.get("platform", ""),
            "section_heading": meta.get("section_heading", ""),
            "score":           round(1 - dist, 4),  # convert distance -> similarity
        })
 
    return output
 
 
if __name__ == "__main__":
    chunks     = load_chunks()
    embedding_client = get_embedding_client()
    chroma_client    = get_chroma_client()
    collection = get_or_create_collection(chroma_client, reset=True)
 
    embed_and_store(chunks, collection, embedding_client)
 
    # quick test
    log.info("\nQuick test — querying: 'how do I backup my messages'")
    results = query_collection(collection, embedding_client, "how do I backup my messages")
    for i, r in enumerate(results, 1):
        log.info(f"  [{i}] score={r['score']} | {r['article_title']} "
                 f"| {r['platform']} | {r['category']}")
        log.info(f"       {r['text'][:120]}...")