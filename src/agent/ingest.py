"""
this file fetches all the Signal help centre articles and sections
from the Zendesk public API and saves them to the data/raw/ folder
"""

import json
import time
import logging
import requests
from pathlib import Path

# import relevant constant vars
from src.agent.config import (
    ZENDESK_ARTICLES_URL,
    ZENDESK_SECTIONS_URL,
    RAW_ARTICLES_FILE,
    RAW_SECTIONS_FILE,
    REQUEST_DELAY,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def fetch_all_articles() -> list[dict]:
    """
    Paginate through the Zendesk API and return all English articles.
    The API returns 30 articles per page; 154 articles = 6 pages.
    """
    articles = []
    url = ZENDESK_ARTICLES_URL
    page = 1
 
    while url:
        log.info(f"Fetching page {page}: {url}")
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
 
        articles.extend(data["articles"])
        log.info(f"  → {len(data['articles'])} articles fetched "
                 f"(total so far: {len(articles)})")
 
        url = data.get("next_page")  # none on last page
        page += 1
        if url:
            time.sleep(REQUEST_DELAY)
 
    log.info(f"Done. Total articles fetched: {len(articles)}")
    return articles


def fetch_sections() -> dict[int, str]:
    """
    Fetch all sections and return a mapping of section_id -> section_name.
    Used to resolve category metadata for each article.
    """
    log.info(f"Fetching sections from {ZENDESK_SECTIONS_URL}")
    response = requests.get(ZENDESK_SECTIONS_URL, timeout=10)
    response.raise_for_status()
    data = response.json()
 
    section_map = {s["id"]: s["name"] for s in data["sections"]}
    log.info(f"Done. Total sections found: {len(section_map)}")
    return section_map


def save_raw_data(articles: list[dict], sections: dict[int, str]) -> None:
    """Save raw API responses to data/raw/ as JSON files."""
    with open(RAW_ARTICLES_FILE, "w", encoding="utf-8") as f:
        json.dump(articles, f, indent=2, ensure_ascii=False)

    log.info(f"Saved {len(articles)} articles -> {RAW_ARTICLES_FILE}")
 
    # save sections as a list of {id, name} dicts for readability
    sections_list = [{"id": k, "name": v} for k, v in sections.items()]
    with open(RAW_SECTIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(sections_list, f, indent=2, ensure_ascii=False)
        
    log.info(f"Saved {len(sections_list)} sections -> {RAW_SECTIONS_FILE}")


def load_raw_articles() -> tuple[list[dict], dict[int, str]]:
    """
    Load previously saved raw data from disk.
    Returns (articles, section_map).
    Raises FileNotFoundError if ingest has not been run yet.
    """
    if not RAW_ARTICLES_FILE.exists() or not RAW_SECTIONS_FILE.exists():
        raise FileNotFoundError(
            "Raw data not found. Run ingest first: python -m src.agent.ingest"
        )
 
    with open(RAW_ARTICLES_FILE, encoding="utf-8") as f:
        articles = json.load(f)
 
    with open(RAW_SECTIONS_FILE, encoding="utf-8") as f:
        sections_list = json.load(f)
    section_map = {s["id"]: s["name"] for s in sections_list}
 
    log.info(f"Loaded {len(articles)} articles and "
             f"{len(section_map)} sections from disk.")
    return articles, section_map
 
 
if __name__ == "__main__":
    articles = fetch_all_articles()
    sections = fetch_sections()
    save_raw_data(articles, sections)