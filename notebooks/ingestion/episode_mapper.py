"""
ingestion/episode_mapper.py
============================
Source notebook: Episode vs chapter mapping.ipynb

For each anime, loops "Episode 1..N" pages on the corresponding Fandom
wiki, asks Gemini for {title, chapters_covered, is_filler}, and appends
to data/episode_mapping (1).jsonl — the exact file src/episode/mapping.py
already reads. A second pass handles named movies, skipping any movie
whose chapters were already covered by TV episodes (avoids double-
counting compilation films).

Run standalone:
    python -m notebooks.ingestion.episode_mapper --anime "Demon Slayer"
    python -m notebooks.ingestion.episode_mapper --all

Sections
--------
1. Configuration      (per-anime wiki domain, episode count, movie titles)
2. Gemini extraction   (episode/movie -> {title, chapters_covered, is_filler})
3. Mapping engine      (TV episode loop, then movie loop with dedup)
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
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from config.settings import settings

# =============================================================================
# 1. Configuration
# =============================================================================

# (wiki_domain, total_episodes, movie_titles) per anime — matches the
# 6 anime src/episode/normalizer.py's EPISODE_MAPPED_ANIME covers.
ANIME_CONFIGS: dict[str, dict] = {
    "Demon Slayer": {
        "wiki_domain": "kimetsu-no-yaiba.fandom.com",
        "total_episodes": 63,
        "movie_titles": [
            "Demon Slayer: Kimetsu no Yaiba - The Movie: Mugen Train",
            "Demon Slayer: Kimetsu no Yaiba - Infinity Castle",
        ],
    },
    "Jujutsu Kaisen": {
        "wiki_domain": "jujutsu-kaisen.fandom.com",
        "total_episodes": 59,
        "movie_titles": [
            "Jujutsu Kaisen 0: The Movie",
            "Jujutsu Kaisen: Execution",
        ],
    },
    "Frieren": {
        "wiki_domain": "frieren.fandom.com",
        "total_episodes": 28,
        "movie_titles": [],
    },
    "Solo Leveling": {
        "wiki_domain": "solo-leveling.fandom.com",
        "total_episodes": 25,
        "movie_titles": [],
    },
    "Attack on Titan": {
        "wiki_domain": "attackontitan.fandom.com",
        "total_episodes": 89,
        "movie_titles": [],
    },
    "Chainsaw Man": {
        "wiki_domain": "chainsaw-man.fandom.com",
        "total_episodes": 12,
        "movie_titles": ["Chainsaw Man - The Movie: Reze Arc"],
    },
}

GEMINI_MODEL = "gemini-3.1-flash-lite-preview"
OUTPUT_JSONL = settings.EPISODE_MAPPING_PATH

# Mirrors src/llm/clients.py's safety settings.
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
            model=GEMINI_MODEL, temperature=0.1,
            safety_settings=SAFETY_SETTINGS, google_api_key=settings.GOOGLE_API_KEY,
        )
    return _llm


# =============================================================================
# 2. Gemini Extraction
# =============================================================================

EXTRACTION_TEMPLATE = """
You are an expert anime data archivist.
Analyze the provided raw wikitext for an anime episode or movie.
Extract the title and find exactly which manga chapters it adapts.
If it is a "Filler" and adapts no chapters, return an empty list for chapters_covered and set is_filler to true.
Ensure chapters_covered is strictly a list of integers.

Return ONLY a valid JSON object in this exact format:
{{
    "title": "The name of the episode or movie",
    "chapters_covered": [1, 2, 3],
    "is_filler": false
}}

Raw Wikitext:
{raw_text}
"""
_prompt = PromptTemplate(input_variables=["raw_text"], template=EXTRACTION_TEMPLATE)


def parse_json_response(raw_content) -> dict | None:
    if isinstance(raw_content, list) and raw_content:
        text = raw_content[0].get("text", "") if isinstance(raw_content[0], dict) else str(raw_content[0])
    else:
        text = str(raw_content)

    text = text.strip()
    if not text:
        return None
    if text.startswith("```json"):
        text = text[7:-3].strip()
    elif text.startswith("```"):
        text = text[3:-3].strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def fetch_wikitext(wiki_domain: str, page_title: str) -> str | None:
    base_url = f"https://{wiki_domain}/api.php"
    params = {
        "action": "query", "prop": "revisions", "rvprop": "content",
        "rvslots": "main", "titles": page_title, "format": "json", "redirects": 1,
    }
    response = requests.get(base_url, params=params, timeout=15)
    pages = response.json().get("query", {}).get("pages", {})
    page_id = next(iter(pages), "-1")
    if page_id == "-1":
        return None
    return pages[page_id]["revisions"][0]["slots"]["main"]["*"]


# =============================================================================
# 3. Mapping Engine
# =============================================================================

def load_saved_episodes(output_file: Path) -> set[tuple[str, str]]:
    """Return the set of (anime_name, episode_number-as-str) already saved."""
    saved = set()
    if not output_file.exists():
        return saved
    with output_file.open(encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line.strip())
                saved.add((d["anime_name"], str(d["episode_number"])))
            except Exception:
                pass
    return saved


def map_anime_episodes_and_movies(
    anime_name: str,
    wiki_domain: str,
    total_episodes: int,
    movie_titles: list[str] | None = None,
    output_file: Path = OUTPUT_JSONL,
    sleep_between: float = 4.5,
) -> None:
    print(f"--- Starting: {anime_name} ---")
    movie_titles = movie_titles or []
    already_saved = load_saved_episodes(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Chapters already covered by TV episodes — used to dedupe compilation
    # movies below. Rebuilt from whatever's already on disk for this anime
    # so a resumed run still dedupes correctly.
    mapped_tv_chapters: set[int] = set()
    if output_file.exists():
        with output_file.open(encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line.strip())
                    if d.get("anime_name") == anime_name and d.get("episode_number") != "Movie":
                        mapped_tv_chapters.update(d.get("chapters_covered", []))
                except Exception:
                    pass

    with output_file.open("a", encoding="utf-8") as f:
        # ---- Phase 1: TV episodes ----
        for ep_num in range(1, total_episodes + 1):
            if (anime_name, str(ep_num)) in already_saved:
                print(f"  Skipping Episode {ep_num} - already saved.")
                continue

            print(f"  Fetching Episode {ep_num}...")
            try:
                raw_text = fetch_wikitext(wiki_domain, f"Episode {ep_num}")
                if raw_text is None:
                    print(f"  Episode {ep_num} not found.")
                    time.sleep(1)
                    continue

                result = get_llm().invoke(_prompt.invoke({"raw_text": raw_text}))
                extracted = parse_json_response(result.content)

                if not extracted:
                    print(f"  Episode {ep_num}: could not extract, skipping.")
                    time.sleep(sleep_between)
                    continue

                chapters = extracted.get("chapters_covered", [])
                mapped_tv_chapters.update(chapters)

                entry = {
                    "anime_name": anime_name,
                    "episode_number": ep_num,
                    "episode_title": extracted.get("title", "Unknown"),
                    "chapters_covered": chapters,
                    "is_filler": extracted.get("is_filler", False),
                }
                json.dump(entry, f, ensure_ascii=False)
                f.write("\n")
                f.flush()
                print(f"  Episode {ep_num} saved. Chapters: {chapters}")
                time.sleep(sleep_between)

            except Exception as e:
                print(f"  Error on Episode {ep_num}: {e}")
                time.sleep(10)

        # ---- Phase 2: movies, deduped against TV chapter coverage ----
        for movie_title in movie_titles:
            if (anime_name, "Movie") in already_saved:
                # NOTE: this only guards a single already-saved movie entry;
                # if an anime has 2+ movies, remove already-saved ones from
                # movie_titles by hand before re-running.
                print(f"  Skipping movie pass - a Movie entry already exists for {anime_name}.")
                break

            print(f"  Fetching Movie: {movie_title}...")
            try:
                raw_text = fetch_wikitext(wiki_domain, movie_title)
                if raw_text is None:
                    print(f"  '{movie_title}' not found on wiki.")
                    continue

                result = get_llm().invoke(_prompt.invoke({"raw_text": raw_text}))
                extracted = parse_json_response(result.content)
                if not extracted:
                    continue

                movie_chapters = set(extracted.get("chapters_covered", []))
                if movie_chapters and movie_chapters.issubset(mapped_tv_chapters):
                    print(f"  Skipped: '{movie_title}' is redundant with existing TV coverage.")
                    time.sleep(2)
                    continue

                entry = {
                    "anime_name": anime_name,
                    "episode_number": "Movie",
                    "episode_title": extracted.get("title", movie_title),
                    "chapters_covered": sorted(movie_chapters),
                    "is_filler": extracted.get("is_filler", False),
                }
                json.dump(entry, f, ensure_ascii=False)
                f.write("\n")
                f.flush()
                print(f"  Movie mapped. Chapters: {sorted(movie_chapters)}")
                time.sleep(sleep_between)

            except Exception as e:
                print(f"  Error on movie {movie_title}: {e}")
                time.sleep(10)

    print(f"--- Done: {anime_name} ---")


# =============================================================================
# 4. CLI Entry-Point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Map anime episodes/movies to manga chapters via Gemini, "
                     "appending to data/episode_mapping (1).jsonl."
    )
    parser.add_argument("--anime", choices=list(ANIME_CONFIGS), help="Map one configured anime")
    parser.add_argument("--all", action="store_true", help="Map every configured anime")
    args = parser.parse_args()

    if not args.anime and not args.all:
        parser.error("pass --anime NAME or --all")

    for name in (list(ANIME_CONFIGS) if args.all else [args.anime]):
        cfg = ANIME_CONFIGS[name]
        map_anime_episodes_and_movies(
            anime_name=name,
            wiki_domain=cfg["wiki_domain"],
            total_episodes=cfg["total_episodes"],
            movie_titles=cfg["movie_titles"],
        )
