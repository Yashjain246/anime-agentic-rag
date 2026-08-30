"""
ingestion/character_extractor.py
=================================
Source notebook: Character Persona JSON creator.ipynb

Pulls the character list from a Fandom wiki's Category:Characters page,
fetches each character's raw wikitext, and asks Gemini to extract a
structured persona (traits, speaking style, dialogue examples, background,
plus a per-field confidence score). Appends each result to
data/all_characters (4).jsonl — the exact file src/persona/character_db.py
already reads.

Run standalone:
    python -m notebooks.ingestion.character_extractor --anime "Chainsaw Man"
    python -m notebooks.ingestion.character_extractor --all

Sections
--------
1. Configuration      (per-anime wiki domain)
2. Wiki fetching       (character list, then each character's wikitext)
3. Gemini extraction   (structured persona prompt + JSON parsing)
4. JSONL writer        (resume-aware append, one line per character)
5. CLI entry-point
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

# Same 4 anime the lore DB covers. Fandom's standard Category:Characters page
# is the source list for each.
WIKI_CONFIGS: dict[str, str] = {
    "Demon Slayer": "kimetsu-no-yaiba.fandom.com",
    "Jujutsu Kaisen": "jujutsu-kaisen.fandom.com",
    "Attack on Titan": "attackontitan.fandom.com",
    "Chainsaw Man": "chainsaw-man.fandom.com",
}

GEMINI_MODEL = "gemini-3.1-flash-lite-preview"
OUTPUT_JSONL = settings.CHARACTER_DB_PATH

_llm: ChatGoogleGenerativeAI | None = None


def get_llm() -> ChatGoogleGenerativeAI:
    global _llm
    if _llm is None:
        _llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, google_api_key=settings.GOOGLE_API_KEY)
    return _llm


# =============================================================================
# 2. Wiki Fetching
# =============================================================================

def get_character_list(wiki_domain: str) -> list[str]:
    """Return every actual character page title (ns=0) under Category:Characters."""
    base_url = f"https://{wiki_domain}/api.php"
    params = {
        "action": "query", "format": "json",
        "list": "categorymembers", "cmtitle": "Category:Characters",
        "cmlimit": "500",
    }
    response = requests.get(base_url, params=params, timeout=15)
    data = response.json()

    titles = []
    for member in data.get("query", {}).get("categorymembers", []):
        if member["ns"] == 0:  # 0 = article page; 14 = subcategory; others = files/templates
            titles.append(member["title"])
    return titles


def get_character_wikitext(wiki_domain: str, character_title: str) -> str | None:
    base_url = f"https://{wiki_domain}/api.php"
    params = {
        "action": "query", "format": "json",
        "prop": "revisions", "rvprop": "content", "rvslots": "main",
        "titles": character_title,
    }
    response = requests.get(base_url, params=params, timeout=15)
    data = response.json()

    pages = data.get("query", {}).get("pages", {})
    for page in pages.values():
        if "revisions" in page:
            return page["revisions"][0]["slots"]["main"]["*"]
    return None


# =============================================================================
# 3. Gemini Extraction
# =============================================================================

EXTRACTION_TEMPLATE = """
You are a highly precise anime character analyst and structured data extraction engine.

Your task is to convert raw MediaWiki wikitext into a clean, structured, and AI-usable character persona.

========================
INPUT
========================
Character Name: {character_name}

Wikitext:
{text}

========================
CORE OBJECTIVE
========================
Extract ONLY high-quality, relevant information that helps simulate this character's personality, behavior, and speech in an AI system.

========================
STRICT RULES
========================
1. DO NOT hallucinate.
   - If information is missing -> return "" or [].
2. IGNORE all wiki markup:
   - [[links]], {{{{templates}}}}, HTML tags, references, file links
3. Prefer:
   - Infobox data for factual attributes (age, affiliation, role)
   - "Personality", "History", "Abilities" sections for semantic traits
4. Keep outputs:
   - Concise
   - Information-dense
   - Clean natural language (no wiki syntax)

========================
EXTRACTION GUIDELINES
========================

PERSONALITY:
- Extract behavioral traits (e.g., calm, arrogant, loyal)
- Extract emotional tendencies (e.g., easily angered, empathetic)
- Extract core values (e.g., justice, friendship)

SPEAKING STYLE:
- Tone (e.g., formal, aggressive, playful)
- Speech patterns (short sentences, dramatic, sarcastic, etc.)
- Catchphrases ONLY if explicitly present

DIALOGUES:
- Extract ONLY canon or clearly stated lines
- Keep them short (1-2 sentences max)
- Do NOT invent or paraphrase

BACKGROUND:
- Focus on origin, role, and major defining events
- Avoid excessive storytelling

CORE IDENTITY:
- Age -> from infobox if available
- Affiliation -> organization, village, group
- Role -> occupation/class (e.g., shinobi, demon slayer)

========================
OUTPUT FORMAT (STRICT JSON ONLY)
========================

{{
  "name": "<character name>",

  "core_identity": {{
    "age": "",
    "affiliation": [],
    "role": ""
  }},

  "personality": {{
    "traits": [],
    "emotional_tendencies": [],
    "values": [],
    "description": ""
  }},

  "speaking_style": {{
    "tone": "",
    "patterns": [],
    "quirks": [],
    "example_dialogues": []
  }},

  "dialogue_examples": {{
    "famous_lines": []
  }},

  "background": {{
    "summary": "",
    "key_events": []
  }},

  "confidence": {{
    "personality": 0.0,
    "speaking_style": 0.0,
    "background": 0.0
  }}
}}

========================
FINAL CONSTRAINTS
========================
- Output ONLY valid JSON
- No explanations
- No extra text
- Ensure it is parsable by Python json.loads()
- If any information isn't available -> return "" or [].
"""

_prompt = PromptTemplate(template=EXTRACTION_TEMPLATE, input_variables=["text", "character_name"])


def parse_json_response(raw_content) -> dict | None:
    if isinstance(raw_content, list) and raw_content:
        text = raw_content[0].get("text", "") if isinstance(raw_content[0], dict) else str(raw_content[0])
    else:
        text = str(raw_content)

    text = text.strip()
    if text.startswith("```json"):
        text = text[7:-3].strip()
    elif text.startswith("```"):
        text = text[3:-3].strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        print(f"    Warning: JSON parse error: {e}")
        return None


def extract_character(name: str, raw_text: str) -> dict | None:
    prompt = _prompt.invoke({"text": raw_text, "character_name": name})
    result = get_llm().invoke(prompt)
    return parse_json_response(result.content)


# =============================================================================
# 4. JSONL Writer (resume-aware)
#
# The recovered notebook always appended blindly with no duplicate check —
# re-running it would re-process (and re-bill) every character from scratch.
# =============================================================================

def load_saved_names(output_file: Path) -> set[str]:
    saved = set()
    if not output_file.exists():
        return saved
    with output_file.open(encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line.strip())
                records = d if isinstance(d, list) else [d]
                for r in records:
                    if isinstance(r, dict) and r.get("name"):
                        saved.add(r["name"].strip().lower())
            except Exception:
                pass
    return saved


def extract_all_characters(
    anime_name: str,
    wiki_domain: str,
    output_file: Path = OUTPUT_JSONL,
    sleep_between: float = 4.0,
) -> None:
    print(f"--- Starting: {anime_name} ---")
    characters = get_character_list(wiki_domain)
    print(f"    Found {len(characters)} characters on wiki")

    saved_names = load_saved_names(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("a", encoding="utf-8") as f:
        for name in characters:
            if name.strip().lower() in saved_names:
                print(f"  Skipping {name} - already saved.")
                continue

            print(f"  Processing: {name}")
            try:
                raw_text = get_character_wikitext(wiki_domain, name)
                if not raw_text:
                    print("    No text found, skipping.")
                    continue

                character_dict = extract_character(name, raw_text)
                if character_dict:
                    json.dump(character_dict, f, ensure_ascii=False)
                    f.write("\n")
                    f.flush()
                    print("    Saved.")
                else:
                    print("    Extraction failed, not saved.")

                time.sleep(sleep_between)

            except Exception as e:
                print(f"    Error processing {name}: {e}")
                time.sleep(30)  # likely a rate limit — back off harder than a normal failure

    print(f"--- Done: {anime_name} ---")


# =============================================================================
# 5. CLI Entry-Point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract character personas via Gemini, appending to "
                     "data/all_characters (4).jsonl."
    )
    parser.add_argument("--anime", choices=list(WIKI_CONFIGS), help="Extract one configured anime")
    parser.add_argument("--all", action="store_true", help="Extract every configured anime")
    args = parser.parse_args()

    if not args.anime and not args.all:
        parser.error("pass --anime NAME or --all")

    for name in (list(WIKI_CONFIGS) if args.all else [args.anime]):
        extract_all_characters(anime_name=name, wiki_domain=WIKI_CONFIGS[name])
