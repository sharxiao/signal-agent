"""
this file features paths, chunking parameters, and model names
all in one place for easy modification
"""

from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()
os.environ["ANONYMIZED_TELEMETRY"] = "False"

# project root detection — works both locally and on Streamlit Cloud
# Local: .../Signal_Agent/src/agent/config.py
# Cloud: /mount/src/signal_agent/src/agent/config.py
def _find_root() -> Path:
    """Find project root by looking for src/agent/ directory."""
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "src" / "agent").is_dir() and (current / "app.py").exists():
            return current
        current = current.parent
    # Fallback: assume config.py is at src/agent/config.py
    return Path(__file__).resolve().parent.parent.parent

ROOT_DIR = _find_root()

DATA_DIR        = ROOT_DIR / "data"
RAW_DIR         = DATA_DIR / "raw"
CHUNKS_DIR      = DATA_DIR / "chunks"
CHROMADB_DIR    = DATA_DIR / "chromadb"

for _dir in [RAW_DIR, CHUNKS_DIR, CHROMADB_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# zendesk api
ZENDESK_ARTICLES_URL = (
    "https://support.signal.org/api/v2/help_center/en-us/articles.json"
)
ZENDESK_SECTIONS_URL = (
    "https://support.signal.org/api/v2/help_center/en-us/sections.json"
)

ARTICLES_PER_PAGE = 30
REQUEST_DELAY     = 0.5

RAW_ARTICLES_FILE = RAW_DIR / "articles.json"
RAW_SECTIONS_FILE = RAW_DIR / "sections.json"
CHUNKS_FILE       = CHUNKS_DIR / "chunks.json"

CHUNK_SIZE    = 400
CHUNK_OVERLAP = 100
DEFAULT_TOP_K = 5

def _read_student_id(id_file: str = "ID.txt") -> str:
    """Read student ID from Streamlit secrets (cloud) or ID.txt (local)."""
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "STUDENT_ID" in st.secrets:
            return st.secrets["STUDENT_ID"]
    except Exception:
        pass

    id_path = ROOT_DIR / id_file
    if not id_path.exists():
        id_path = Path(id_file)
    if not id_path.exists():
        raise FileNotFoundError(
            "ID.txt not found. Create it at the project root with:\n"
            "  Line 1: student ID\n"
            "Or set STUDENT_ID in Streamlit secrets for cloud deployment."
        )
    with open(id_path, "r") as f:
        lines = f.readlines()
    if len(lines) < 1:
        raise ValueError("Please add your student ID on line 1.")
    student_id = lines[0].strip()
    if not student_id:
        raise ValueError("Student ID cannot be empty!")
    return student_id

STUDENT_ID = _read_student_id()

EMBEDDING_BASE_URL = "https://rsm-8430-a2.bjlkeng.io"
EMBEDDING_API_KEY  = STUDENT_ID
EMBEDDING_MODEL = ""

CHROMA_COLLECTION_NAME = "signal_support"

LLM_BASE_URL   = "https://rsm-8430-finalproject.bjlkeng.io"
LLM_API_KEY    = STUDENT_ID
LLM_MODEL      = "qwen3-30b-a3b-fp8"
LLM_MAX_TOKENS = 2048
