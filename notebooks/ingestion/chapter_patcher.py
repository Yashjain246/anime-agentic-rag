"""
ingestion/chapter_patcher.py
============================
Source notebook: manga_fix_last2.ipynb

Targeted fix for the 2 manga chapters that were blocked / failed during
the main extraction run in manga_scraper.py. Applies alternate prompts
or fallback sources, then merges the fixed records back into the main
JSONL file.

Run standalone:
    python -m notebooks.ingestion.chapter_patcher

Sections
--------
1. Configuration   (which chapters failed and why)
2. Fix strategies  (alternate prompts / sources)
3. JSONL patcher   (replace-or-append merge)
4. Main fix runner
5. CLI entry-point
"""

from __future__ import annotations
import argparse, json
from pathlib import Path

# =============================================================================
# 1. Configuration
# =============================================================================

MAIN_JSONL = Path("data/manga_chapters (3).jsonl")

# Fill in from the notebook — which 2 chapters failed?
BLOCKED_CHAPTERS: list[int] = []   # e.g. [731, 732]

# TODO: paste any chapter-specific metadata / notes from manga_fix_last2.ipynb


# =============================================================================
# 2. Fix Strategies
# =============================================================================

def fix_chapter(chapter_num: int) -> dict | None:
    """
    Apply the alternate extraction strategy for a known-blocked chapter.

    Returns
    -------
    dict | None
        ``{"chapter": int, "title": str, "summary": str}`` or None on failure.

    TODO: Replace with the actual per-chapter fix logic from the notebook.
          Different chapters may need different strategies.
    """
    raise NotImplementedError


# =============================================================================
# 3. JSONL Patcher
# =============================================================================

def patch_jsonl(fixed_records: list[dict], path: Path = MAIN_JSONL) -> None:
    """
    Merge *fixed_records* into *path*.

    Existing records with a matching ``chapter`` key are **replaced**;
    new records are **appended**. The file is updated in-place.

    TODO: Replace with actual patching code from manga_fix_last2.ipynb.
    """
    raise NotImplementedError


# =============================================================================
# 4. Main Fix Runner
# =============================================================================

def run_fixes(output: Path = MAIN_JSONL) -> None:
    """
    Run fixes for all BLOCKED_CHAPTERS and patch them into *output*.

    TODO: Replace with the actual run loop from manga_fix_last2.ipynb.
    """
    raise NotImplementedError


# =============================================================================
# 5. CLI Entry-Point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fix and patch blocked manga chapters into the main JSONL."
    )
    parser.add_argument(
        "--output", default=str(MAIN_JSONL),
        help="Path to the main manga chapters JSONL file",
    )
    args = parser.parse_args()
    run_fixes(output=Path(args.output))
