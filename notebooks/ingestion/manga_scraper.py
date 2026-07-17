"""
ingestion/manga_scraper.py
==========================
Source notebooks: manga_chapters_v2.ipynb

Scrapes MediaWiki wikitext for each One Piece manga chapter, cleans it,
and uses the Gemini API to produce a structured summary. Results are
appended incrementally to data/manga_chapters (3).jsonl.

Run standalone:
    python -m notebooks.ingestion.manga_scraper [--start N] [--end N]

Sections
--------
1. Configuration
2. Wikitext utilities  (cleaning, section extraction)
3. Gemini extraction   (prompt, API call, retry logic)
4. JSONL writer        (resume-aware append)
5. Main extraction loop
6. CLI entry-point
"""

from __future__ import annotations
import argparse, json, os, re, time
from pathlib import Path

# TODO: uncomment once dependencies are confirmed
# import google.generativeai as genai
# import requests

# =============================================================================
# 1. Configuration
# =============================================================================

GEMINI_MODEL     = "gemini-1.5-flash"   # TODO: verify against notebook
MAX_RETRIES      = 3
RETRY_DELAY_SEC  = 5
OUTPUT_JSONL     = Path("data/manga_chapters (3).jsonl")
WIKI_API_URL     = "https://onepiece.fandom.com/api.php"  # TODO: verify

# TODO: paste API key setup / genai.configure() from the notebook here


# =============================================================================
# 2. Wikitext Utilities
# =============================================================================

def clean_wikitext(raw: str) -> str:
    """
    Strip MediaWiki markup from *raw* and return plain text suitable for
    LLM summarisation.

    Removes: templates {{...}}, tags <ref>/<gallery>, HTML entities, etc.

    TODO: Replace with actual cleaning logic from manga_chapters_v2.ipynb.
    """
    raise NotImplementedError


def extract_plot_section(wikitext: str) -> str:
    """
    Pull out only the plot/synopsis section from cleaned wikitext.

    TODO: Replace with actual section-extraction code from the notebook.
    """
    raise NotImplementedError


def fetch_wikitext(chapter_num: int) -> str | None:
    """
    Call the MediaWiki API and return raw wikitext for *chapter_num*.
    Returns None if the page is missing or the request fails.

    TODO: Replace with actual MediaWiki fetch code from the notebook.
    """
    raise NotImplementedError


# =============================================================================
# 3. Gemini Extraction
# =============================================================================

SUMMARY_PROMPT_TEMPLATE = """
You are an expert One Piece wiki editor.
Given the following plot text from Chapter {chapter_num}: "{chapter_title}",
write a concise, factual summary covering key events, characters, and lore.

Plot text:
{plot_text}

Summary:
"""
# TODO: replace with the exact prompt from manga_chapters_v2.ipynb


def summarise_chapter(chapter_num: int, title: str, plot_text: str) -> str | None:
    """
    Call Gemini to summarise *plot_text* for the given chapter.
    Retries up to MAX_RETRIES times on failure.

    TODO: Replace with actual Gemini call + retry logic from the notebook.
    """
    raise NotImplementedError


# =============================================================================
# 4. JSONL Writer (resume-aware)
# =============================================================================

def load_completed_chapters(path: Path) -> set[int]:
    """Return the set of chapter numbers already written to *path*."""
    if not path.exists():
        return set()
    completed = set()
    with path.open() as f:
        for line in f:
            try:
                completed.add(json.loads(line)["chapter"])
            except (json.JSONDecodeError, KeyError):
                pass
    return completed


def append_record(record: dict, path: Path) -> None:
    """Append a single chapter *record* as a JSONL line to *path*."""
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# =============================================================================
# 5. Main Extraction Loop
# =============================================================================

def run_extraction(
    start: int = 1,
    end: int = 843,
    output: Path = OUTPUT_JSONL,
    resume: bool = True,
) -> None:
    """
    Iterate chapters [start, end], extract summaries, and write to *output*.

    Parameters
    ----------
    start, end : int
        Inclusive chapter range.
    output : Path
        Destination JSONL file.
    resume : bool
        Skip chapters already in *output* when True.

    TODO: Replace the body with the actual extraction loop from the notebook.
    """
    raise NotImplementedError


# =============================================================================
# 6. CLI Entry-Point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scrape and summarise One Piece manga chapters via Gemini."
    )
    parser.add_argument("--start",     type=int, default=1,   help="First chapter")
    parser.add_argument("--end",       type=int, default=843, help="Last chapter")
    parser.add_argument("--no-resume", dest="resume", action="store_false",
                        help="Overwrite existing output instead of resuming")
    args = parser.parse_args()
    run_extraction(start=args.start, end=args.end, resume=args.resume)
