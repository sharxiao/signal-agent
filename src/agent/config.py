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

def _read_student_id(id_file: str = "ID.txt") -> str:
    """Read student ID from ID.txt (line 1)."""
    id_path = Path(id_file)
    if not id_path.exists():
        raise FileNotFoundError(
            "ID.txt not found. Create it at the project root with:\n"
            "  Line 1: student ID"
        )
    with open(id_path, "r") as f:
        lines = f.readlines()
        if len(lines) < 1:
            raise ValueError("Please add your student ID on line 1.")
        student_id = lines[0].strip()
        if not student_id:
            raise ValueError("Student ID cannot be empty!")
        return student_id

# api authentication
STUDENT_ID = _read_student_id()

# embedding endpoint
EMBEDDING_BASE_URL = "https://rsm-8430-a2.bjlkeng.io"
EMBEDDING_API_KEY  = STUDENT_ID
EMBEDDING_MODEL = ""

# ChromaDB
CHROMA_COLLECTION_NAME = "signal_support"

# LLM endpoint
LLM_BASE_URL   = "https://rsm-8430-finalproject.bjlkeng.io"
LLM_API_KEY    = STUDENT_ID
LLM_MODEL      = "qwen3-30b-a3b-fp8"
LLM_MAX_TOKENS = 1024