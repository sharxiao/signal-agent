"""
this file features paths, chunking parameters, and model names
all in one place for easy modification
"""

from pathlib import Path
import os

from dotenv import load_dotenv

load_dotenv()

os.environ["ANONYMIZED_TELEMETRY"] = "False"

# project root and data directories
ROOT_DIR = Path(__file__).resolve()         # signal_agent/
while ROOT_DIR.name != "Signal_Agent":
    ROOT_DIR = ROOT_DIR.parent   
     
DATA_DIR        = ROOT_DIR / "data"
RAW_DIR         = DATA_DIR / "raw"          # raw JSON from Zendesk API
CHUNKS_DIR      = DATA_DIR / "chunks"       # processed chunks as JSON
CHROMADB_DIR    = DATA_DIR / "chromadb"     # ChromaDB persistent storage

# create directories if they don't exist
for _dir in [RAW_DIR, CHUNKS_DIR, CHROMADB_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# zendesk api
ZENDESK_ARTICLES_URL = (
    "https://support.signal.org/api/v2/help_center/en-us/articles.json"
)
ZENDESK_SECTIONS_URL = (
    "https://support.signal.org/api/v2/help_center/en-us/sections.json"
)
ARTICLES_PER_PAGE = 30   # zendesk default; 154 articles = 6 pages
REQUEST_DELAY     = 0.5  # seconds between API requests

# raw data file names
RAW_ARTICLES_FILE = RAW_DIR / "articles.json"
RAW_SECTIONS_FILE = RAW_DIR / "sections.json"
CHUNKS_FILE       = CHUNKS_DIR / "chunks.json"

# chunking parameters
CHUNK_SIZE    = 400
CHUNK_OVERLAP = 50

# embedding model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # sentence-transformers model

# ChromaDB
CHROMA_COLLECTION_NAME = "signal_support"

# LLM endpoint
LLM_MODEL = "qwen3-30b-a3b-fp8"
LLM_BASE_URL = "https://rsm-8430-finalproject.bjlkeng.io" # course project endpoint
LLM_MAX_TOKENS = 1024
LLM_API_KEY = os.getenv("LLM_API_KEY", "")