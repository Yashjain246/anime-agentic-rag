"""
ingestion/anime_synopsis_scraper.py
====================================
Source notebook: Anime description JSON creator.ipynb

Pulls top-ranked anime from Jikan's API (unofficial MyAnimeList API,
already used elsewhere in this project by src/tools/jikan.py) and writes
{title, synopsis, score, genres} records to data/anime_desc (1).jsonl —
the file scripts/rebuild_recs_db.py indexes into the recs vector store.

No LLM involved — this is a straight paginated API pull, no Gemini calls,
no cost beyond Jikan's own rate limit.

Run standalone:
    python -m notebooks.ingestion.anime_synopsis_scraper --count 500

Sections
--------
1. Configuration
2. Fetch + clean
3. JSONL writer (resume-aware)
4. CLI entry-point
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import requests

from config.settings import settings

# =============================================================================
# 1. Configuration
# =============================================================================

JIKAN_TOP_ANIME_URL = "https://api.jikan.moe/v4/top/anime"
ANIME_PER_PAGE = 25
OUTPUT_JSONL = settings.DATA_DIR / "anime_desc (1).jsonl"


# =============================================================================
# 2. Fetch + Clean
# =============================================================================

def clean_record(anime: dict) -> dict | None:
    synopsis = anime.get("synopsis")
    if not synopsis:
        return None

    clean_synopsis = synopsis.replace("\n", " ").strip()
    clean_synopsis = clean_synopsis.replace("[Written by MAL Rewrite]", "").strip()

    return {
        "title": anime.get("title", "unknown"),
        "synopsis": clean_synopsis,
        "score": anime.get("score", 0),
        "genres": [g["name"] for g in anime.get("genres", [])],
    }


# =============================================================================
# 3. JSONL Writer (resume-aware)
# =============================================================================

def load_saved_titles(output_file: Path) -> set[str]:
    saved = set()
    if not output_file.exists():
        return saved
    with output_file.open(encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line.strip())
                if d.get("title"):
                    saved.add(d["title"].strip().lower())
            except Exception:
                pass
    return saved


def extract_anime_synopses(
    count: int = 500,
    output_file: Path = OUTPUT_JSONL,
    sleep_between: float = 1.5,
) -> None:
    total_pages = count // ANIME_PER_PAGE
    saved_titles = load_saved_titles(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("a", encoding="utf-8") as f:
        for page in range(1, total_pages + 1):
            print(f"Fetching page {page}/{total_pages}...")
            try:
                response = requests.get(JIKAN_TOP_ANIME_URL, params={"page": page}, timeout=15)

                if response.status_code != 200:
                    print(f"  Error: HTTP {response.status_code}, sleeping 10s")
                    time.sleep(10)
                    continue

                for anime in response.json().get("data", []):
                    record = clean_record(anime)
                    if not record:
                        continue
                    if record["title"].strip().lower() in saved_titles:
                        continue

                    json.dump(record, f, ensure_ascii=False)
                    f.write("\n")
                    saved_titles.add(record["title"].strip().lower())

                f.flush()
                time.sleep(sleep_between)

            except Exception as e:
                print(f"  Error: {e}")

    print("Done.")


# =============================================================================
# 4. CLI Entry-Point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pull top-anime synopses from Jikan, appending to "
                     "data/anime_desc (1).jsonl."
    )
    parser.add_argument("--count", type=int, default=500, help="How many anime to fetch")
    args = parser.parse_args()
    extract_anime_synopses(count=args.count)
