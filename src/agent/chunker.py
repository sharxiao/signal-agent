"""
this file cleans article HTML, detects platform, and splits it into overlapping text chunks

each chunk is a dict with the following fields:
- chunk_id            str        -> unique id e.g. 123456789-00
- text                str        -> plain text content of the chunk
- article_title       str        -> title of the source article
- source_url          str        -> public signal help centre url
- category            str        -> section name
- platform            str        -> android, ios, desktop (or all)
- section_heading     str        -> nearest heading above this chunk (or "")
- chunk_index         int        -> position of chunk within article
- word_count          int        -> number of words in the chunk
"""

import json
import re
import logging
from html.parser import HTMLParser

# import relevant constant vars
from src.agent.config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    CHUNKS_FILE
)
from src.agent.ingest import load_raw_articles

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# html cleaning!
# minimal html parser that extracts plain text while preserving whitespace
class _HTMLStripper(HTMLParser):
 
    def __init__(self):
        super().__init__()
        self._parts = []
        # tags that should add a newline to preserve structure
        self._block_tags = {
            "p", "li", "h1", "h2", "h3", "h4", "h5", "h6",
            "tr", "br", "hr", "div",
        }
 
    def handle_starttag(self, tag, attrs):
        if tag in self._block_tags:
            self._parts.append("\n")
 
    def handle_data(self, data):
        self._parts.append(data)
 
    def get_text(self) -> str:
        raw = "".join(self._parts)
        # collapse multiple blank lines and strip leading/trailing whitespace
        raw = re.sub(r"\n{3,}", "\n\n", raw)
        return raw.strip()
 
 
def strip_html(html: str) -> str:
    """Remove HTML tags and return clean plain text."""
    stripper = _HTMLStripper()
    stripper.feed(html or "")
    return stripper.get_text()


# platform detection
def detect_platform(text: str, title: str) -> str:
    """
    Infer platform from article text and title.
    Returns "Android", "iOS", "Desktop", or "All".
    """
    combined = (title + " " + text).lower()
 
    has_android = "android" in combined
    has_ios     = any(kw in combined for kw in ["ios", "iphone", "ipad"])
    has_desktop = "desktop" in combined
 
    platforms = sum([has_android, has_ios, has_desktop])
    if platforms > 1 or platforms == 0:
        return "All"
    if has_android:
        return "Android"
    if has_ios:
        return "iOS"
    return "Desktop"


# section heading extraction
def extract_headings(html: str) -> list[tuple[int, str]]:
    """
    Return a list of (char_position, heading_text) from h2/h3/h4 tags.
    Used to tag each chunk with the nearest preceding heading.
    """
    pattern = re.compile(
        r"<h[234][^>]*>(.*?)</h[234]>", re.IGNORECASE | re.DOTALL
    )

    headings = []
    for match in pattern.finditer(html):
        heading_text = strip_html(match.group(1)).strip()
        headings.append((match.start(), heading_text))
    return headings


# word-based chunking
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split text into overlapping word-based chunks.
    Short articles (< chunk_size words) are returned as a single chunk.
    """
    words = text.split()
    if len(words) <= chunk_size:
        return [text]
 
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += chunk_size - overlap
 
    return chunks


# main processing
def process_article(article: dict, section_map: dict[int, str]) -> list[dict]:
    """
    Convert a single raw Zendesk article into a list of chunk dicts.
    """
    article_id    = str(article["id"])
    title         = article.get("title", "")
    source_url    = article.get("html_url", "")
    body_html     = article.get("body", "")
    section_id    = article.get("section_id")
    category      = section_map.get(section_id, "General")
 
    plain_text    = strip_html(body_html)
    platform      = detect_platform(plain_text, title)
    headings      = extract_headings(body_html)
 
    raw_chunks    = chunk_text(plain_text)
    result        = []
 
    for idx, chunk_text_str in enumerate(raw_chunks):
        # find the nearest heading that precedes this chunk
        # we use a rough heuristic: map chunk index to a character offset.
        approx_char_pos = (idx / max(len(raw_chunks), 1)) * len(body_html)
        section_heading = ""
        for char_pos, heading in reversed(headings):
            if char_pos <= approx_char_pos:
                section_heading = heading
                break
 
        chunk = {
            "chunk_id":        f"{article_id}-{idx:03d}",
            "text":            chunk_text_str,
            "article_title":   title,
            "source_url":      source_url,
            "category":        category,
            "platform":        platform,
            "section_heading": section_heading,
            "chunk_index":     idx,
            "word_count":      len(chunk_text_str.split()),
        }
        result.append(chunk)
 
    return result
 
 
def build_chunks(articles: list[dict], section_map: dict[int, str]) -> list[dict]:
    """Process all articles and return a flat list of chunks."""
    all_chunks = []
    for article in articles:
        chunks = process_article(article, section_map)
        all_chunks.extend(chunks)
 
    log.info(
        f"Built {len(all_chunks)} chunks from {len(articles)} articles."
    )
    return all_chunks
 
 
def save_chunks(chunks: list[dict]) -> None:
    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    log.info(f"Saved {len(chunks)} chunks -> {CHUNKS_FILE}")
 
 
def load_chunks() -> list[dict]:
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(
            "Chunks not found. Run chunker first: python -m src.agent.chunker"
        )
    with open(CHUNKS_FILE, encoding="utf-8") as f:
        chunks = json.load(f)
    log.info(f"Loaded {len(chunks)} chunks from {CHUNKS_FILE}")
    return chunks
 
 
if __name__ == "__main__":
    articles, section_map = load_raw_articles()
    chunks = build_chunks(articles, section_map)
    save_chunks(chunks)
 
    # sanity check
    log.info("Sample chunk:")
    sample = chunks[0]
    for key, val in sample.items():
        preview = str(val)[:80] + "..." if len(str(val)) > 80 else str(val)
        log.info(f"  {key}: {preview}")