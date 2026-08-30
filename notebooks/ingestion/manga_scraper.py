"""
ingestion/manga_scraper.py
==========================
Source notebook: manga_chapters_v2.ipynb

Scrapes manga chapter text from each anime's Fandom wiki, strips wikitext
markup, and uses Gemini to produce a structured {summary, key_events}
record per chapter. Results are appended incrementally to
data/manga_chapters (3).jsonl — the same file src/rag and
scripts/rebuild_lore_db.py already consume, so a freshly scraped chapter
is usable immediately by rebuild_lore_db.py with no format changes.

This is what makes the pipeline scalable: adding a new anime, or filling
in chapters missed on a previous run, no longer requires reconstructing
this step from scratch.

Run standalone:
    python -m notebooks.ingestion.manga_scraper --anime "Demon Slayer"
    python -m notebooks.ingestion.manga_scraper --anime "Chainsaw Man" --chapters 7 11 12
    python -m notebooks.ingestion.manga_scraper --all

Sections
--------
1. Configuration      (per-anime wiki domain + known chapter count)
2. Wikitext cleaning   (strip markup that trips Gemini's safety filters)
3. Gemini extraction   (primary prompt -> fallback prompt -> placeholder)
4. JSONL writer        (resume-aware append, one line per chapter)
5. Audit               (report saved / missing / placeholder chapters)
6. CLI entry-point
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from config.settings import settings

# =============================================================================
# 1. Configuration
# =============================================================================

# Same domain-naming convention Fandom uses for the other 3 (a hyphenated
# short form of the English title) — verify this resolves before a large
# scraping run.
WIKI_CONFIGS: dict[str, dict] = {
    "Demon Slayer":    {"wiki_domain": "kimetsu-no-yaiba.fandom.com", "total_chapters": 205},
    "Jujutsu Kaisen":  {"wiki_domain": "jujutsu-kaisen.fandom.com",   "total_chapters": 271},
    "Attack on Titan": {"wiki_domain": "attackontitan.fandom.com",    "total_chapters": 139},
    "Chainsaw Man":    {"wiki_domain": "chainsaw-man.fandom.com",     "total_chapters": 232},
}

# gemini-3.1-flash-lite-preview was the working model at recovery time.
# config.settings doesn't have a dedicated ingestion-model setting (LLM_MODEL
# is the production chat model) — kept as its own constant so bumping the
# chat model doesn't silently change what a bulk scrape run costs/behaves like.
GEMINI_MODEL = "gemini-3.1-flash-lite-preview"

OUTPUT_JSONL = settings.MANGA_CHAPTERS_PATH

# Mirrors src/llm/clients.py's _SAFETY_SETTINGS. Duplicated rather than
# imported so this module stays runnable standalone, same as the other
# notebooks/ scripts.
SAFETY_SETTINGS = {
    "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
    "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
    "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
    "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
}

_llm: ChatGoogleGenerativeAI | None = None


def get_llm() -> ChatGoogleGenerativeAI:
    global _llm
    if _llm is None:
        _llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            temperature=0.1,
            safety_settings=SAFETY_SETTINGS,
            google_api_key=settings.GOOGLE_API_KEY,
        )
    return _llm


# =============================================================================
# 2. Wikitext Cleaning
# =============================================================================

def clean_wikitext(raw: str) -> str:
    """
    Strip MediaWiki markup down to plain prose before sending to Gemini.

    Raw wikitext contains things like {{nihongo|...}}, [[File:gore.png]],
    <ref>...</ref>, ==Gallery==, and infobox templates with raw violent
    descriptions — these trip Gemini's safety filters even when the actual
    narrative text is fine. Stripping them first reduces blocked chapters
    dramatically (this was the single biggest fix in the original notebook).
    """
    # Drop whole sections that are noisy/triggering and add no narrative value
    raw = re.sub(
        r"==\s*(Gallery|Trivia|Navigation|References|Notes|External Links?)"
        r"\s*==.*?((?===)|$)",
        "", raw, flags=re.DOTALL | re.IGNORECASE,
    )

    # HTML comments, <ref> blocks, and any remaining HTML tags
    raw = re.sub(r"<!--.*?-->", "", raw, flags=re.DOTALL)
    raw = re.sub(r"<ref[^>]*>.*?</ref>", "", raw, flags=re.DOTALL)
    raw = re.sub(r"<[^>]+>", "", raw)

    # File/Image links can contain violent alt-text descriptions
    raw = re.sub(r"\[\[(?:File|Image):[^\]]*\]\]", "", raw, flags=re.IGNORECASE)

    # Infobox / template blocks {{ ... }} — loop since these can nest
    prev = None
    while prev != raw:
        prev = raw
        raw = re.sub(r"\{\{[^{}]*\}\}", "", raw)

    # Wiki links: [[Link|Text]] -> Text, [[Link]] -> Link
    raw = re.sub(r"\[\[[^|\]]+\|([^\]]+)\]\]", r"\1", raw)
    raw = re.sub(r"\[\[([^\]]+)\]\]", r"\1", raw)

    # External links
    raw = re.sub(r"\[https?://\S+\s*([^\]]*)\]", r"\1", raw)

    # Bold/italic markup
    raw = re.sub(r"'{2,3}", "", raw)

    # Section header markup (keep the heading text itself)
    raw = re.sub(r"=+\s*(.*?)\s*=+", r"\n\1\n", raw)

    # Collapse whitespace
    raw = re.sub(r"\n{3,}", "\n\n", raw)
    raw = re.sub(r"[ \t]+", " ", raw)

    return raw.strip()


def fetch_wikitext(wiki_domain: str, chapter_num: int) -> str | None:
    """Fetch raw wikitext for a chapter via the MediaWiki API. None if missing."""
    import requests

    base_url = f"https://{wiki_domain}/api.php"
    params = {
        "action": "query", "prop": "revisions",
        "rvprop": "content", "rvslots": "main",
        "titles": f"Chapter {chapter_num}",
        "format": "json", "redirects": 1,
    }
    resp = requests.get(base_url, params=params, timeout=15)
    resp.raise_for_status()
    pages = resp.json().get("query", {}).get("pages", {})
    page_id = next(iter(pages), "-1")
    if page_id == "-1":
        return None
    return pages[page_id]["revisions"][0]["slots"]["main"]["*"]


# =============================================================================
# 3. Gemini Extraction
# =============================================================================

PRIMARY_TEMPLATE = """
You are a manga lore archivist. Extract a structured summary from the chapter text below.
Focus on: narrative events, plot progression, character decisions, and lore reveals.
Ignore any remaining markup artefacts.

Return ONLY a valid JSON object, no markdown fences, no extra text:
{{
    "plot_summary": "A paragraph describing what happens in this chapter.",
    "key_events": ["Event 1", "Event 2", "Event 3"]
}}

Chapter Text:
{raw_text}
"""

# Some chapters trip the safety filter on the primary prompt's wording alone
# even after cleaning — a more clinical/academic framing gets through where
# "manga lore archivist" doesn't.
FALLBACK_TEMPLATE = """
You are a literary analyst summarising a story chapter.
Describe the sequence of events, character motivations, and plot changes using academic, neutral language.

Return ONLY a valid JSON object, no markdown fences, no extra text:
{{
    "plot_summary": "Academic summary of the chapter's events.",
    "key_events": ["Event 1", "Event 2", "Event 3"]
}}

Chapter Text:
{raw_text}
"""

_primary_prompt = PromptTemplate(input_variables=["raw_text"], template=PRIMARY_TEMPLATE)
_fallback_prompt = PromptTemplate(input_variables=["raw_text"], template=FALLBACK_TEMPLATE)


def parse_json_response(raw_content) -> dict | None:
    """Safely parse JSON out of a Gemini response, tolerant of fences/prose around it."""
    if not raw_content:
        return None
    if isinstance(raw_content, list):
        if not raw_content:
            return None
        text = raw_content[0].get("text", "") if isinstance(raw_content[0], dict) else str(raw_content[0])
    else:
        text = str(raw_content)

    text = text.strip()
    if not text:
        return None

    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    json_match = re.search(r"\{.*\}", text, re.DOTALL)
    if json_match:
        text = json_match.group(0)

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        print(f"    Warning: JSON parse error: {e}")
        print(f"    Raw snippet: {text[:200]}")
        return None


def summarise_chapter(cleaned_text: str) -> dict | None:
    """Try the primary prompt, then the fallback prompt, on already-cleaned wikitext."""
    capped = cleaned_text[:6000]  # stay well inside the token limit

    for attempt, (label, tmpl) in enumerate(
        [("primary", _primary_prompt), ("fallback", _fallback_prompt)], 1
    ):
        try:
            prompt = tmpl.invoke({"raw_text": capped})
            result = get_llm().invoke(prompt)
            content = result.content

            if not content:
                print(f"    Attempt {attempt} ({label}): blocked.", end=" ")
                if attempt == 1:
                    print("Trying fallback...")
                    time.sleep(2)
                    continue
                print("Both prompts blocked.")
                return None

            parsed = parse_json_response(content)
            if parsed:
                if attempt == 2:
                    print("    Fallback prompt succeeded.")
                return parsed

            print(f"    Attempt {attempt} ({label}): bad JSON.", end=" ")
            if attempt == 1:
                print("Trying fallback...")
                time.sleep(2)

        except Exception as e:
            print(f"    Attempt {attempt} ({label}): exception - {e}")
            if attempt == 1:
                time.sleep(3)

    return None


# =============================================================================
# 4. JSONL Writer (resume-aware)
# =============================================================================

def load_saved_chapters(output_file: Path) -> set[tuple[str, int]]:
    """Return the set of (anime_name, chapter_number) already written to *output_file*."""
    saved = set()
    if not output_file.exists():
        return saved
    with output_file.open(encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line.strip())
                saved.add((d["anime_name"], d["chapter_number"]))
            except Exception:
                pass
    return saved


def extract_manga_chapters(
    anime_name: str,
    wiki_domain: str,
    total_chapters: int,
    output_file: Path = OUTPUT_JSONL,
    specific_chapters: list[int] | None = None,
    sleep_between: float = 4.5,
) -> None:
    """
    Scrape + summarise chapters for one anime, appending each result as a
    JSONL line. Safe to re-run: chapters already in *output_file* are
    skipped, so this doubles as the gap-fill mechanism for chapters that
    failed (network error, or genuinely blocked by the safety filter on
    both prompts) on a previous run.
    """
    print(f"--- Starting: {anime_name} ---")
    saved_chapters = load_saved_chapters(output_file)
    already = sum(1 for (a, _c) in saved_chapters if a == anime_name)
    print(f"    Already saved: {already} chapters")

    chapter_list = specific_chapters if specific_chapters else range(1, total_chapters + 1)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("a", encoding="utf-8") as f:
        for chapter_num in chapter_list:
            if (anime_name, chapter_num) in saved_chapters:
                print(f"  Skipping Ch.{chapter_num} - already saved.")
                continue

            print(f"  Fetching Ch.{chapter_num}...")
            try:
                raw_text = fetch_wikitext(wiki_domain, chapter_num)
                if raw_text is None:
                    print(f"  Ch.{chapter_num} not found.")
                    time.sleep(1)
                    continue

                cleaned_text = clean_wikitext(raw_text)
                extracted = summarise_chapter(cleaned_text)

                if extracted:
                    entry = {
                        "anime_name": anime_name,
                        "chapter_number": chapter_num,
                        "summary_text": extracted.get("plot_summary", ""),
                        "key_events": extracted.get("key_events", []),
                    }
                else:
                    entry = {
                        "anime_name": anime_name,
                        "chapter_number": chapter_num,
                        "summary_text": "[BLOCKED - could not extract]",
                        "key_events": [],
                        "error": True,
                    }

                json.dump(entry, f, ensure_ascii=False)
                f.write("\n")
                f.flush()
                print(f"  {'OK' if extracted else 'FAILED'} Ch.{chapter_num} saved.")
                time.sleep(sleep_between)

            except Exception as e:
                print(f"  Error on Ch.{chapter_num}: {e}")
                time.sleep(5)

    print(f"--- Done: {anime_name} ---")


# =============================================================================
# 5. Audit
# =============================================================================

def audit_jsonl(output_file: Path, anime_configs: dict[str, dict]) -> None:
    """Report saved / placeholder / missing chapter counts per configured anime."""
    saved, errors = {}, {}
    if output_file.exists():
        with output_file.open(encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                    key = (d["anime_name"], d["chapter_number"])
                    saved[key] = True
                    if d.get("error"):
                        errors[key] = True
                except Exception:
                    pass

    print("=" * 55)
    for name, cfg in anime_configs.items():
        total = cfg["total_chapters"]
        have = [c for (a, c) in saved if a == name]
        bad = [c for (a, c) in errors if a == name]
        missing = [c for c in range(1, total + 1) if (name, c) not in saved]
        print(f"{name}: {len(have)}/{total} saved | {len(bad)} placeholders | {len(missing)} missing")
        if missing:
            print(f"  Missing:      {missing}")
        if bad:
            print(f"  Placeholders: {bad}")
    print("=" * 55)


# =============================================================================
# 6. CLI Entry-Point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scrape and summarise manga chapters via Gemini, appending to "
                     "data/manga_chapters (3).jsonl."
    )
    parser.add_argument("--anime", choices=list(WIKI_CONFIGS), help="Scrape one configured anime")
    parser.add_argument("--all", action="store_true", help="Scrape every configured anime")
    parser.add_argument("--chapters", type=int, nargs="+",
                         help="Only these chapter numbers (gap-fill mode)")
    parser.add_argument("--audit", action="store_true",
                         help="Report saved/missing/placeholder counts and exit, no scraping")
    args = parser.parse_args()

    if args.audit:
        audit_jsonl(OUTPUT_JSONL, WIKI_CONFIGS)
        sys.exit(0)

    if not args.anime and not args.all:
        parser.error("pass --anime NAME, --all, or --audit")

    targets = list(WIKI_CONFIGS) if args.all else [args.anime]
    for name in targets:
        cfg = WIKI_CONFIGS[name]
        extract_manga_chapters(
            anime_name=name,
            wiki_domain=cfg["wiki_domain"],
            total_chapters=cfg["total_chapters"],
            specific_chapters=args.chapters,
        )

    audit_jsonl(OUTPUT_JSONL, WIKI_CONFIGS)
