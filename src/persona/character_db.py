"""
src/persona/character_db.py
───────────────────────────
Loads and queries the character database from all_characters (4).jsonl.

WHY a dict keyed by lowercase name:
  O(1) exact-match lookup, and case-insensitive matching is the most
  common case ("gojo" vs "Gojo" vs "GOJO").

NOTE on file format: most lines are a single JSON object
  ({"name": "Akaza", ...}), but a few lines are a JSON ARRAY of
  multiple character objects (minor/grouped characters). Both formats
  are handled so no characters are silently dropped.
"""

from __future__ import annotations

import json

from config.settings import settings

_CHARACTER_DB: dict[str, dict] | None = None


def get_character_db() -> dict[str, dict]:
    """
    Returns the full character database as a dict keyed by lowercase name.
    Loaded once at first access, cached for the lifetime of the process.
    """
    global _CHARACTER_DB
    if _CHARACTER_DB is not None:
        return _CHARACTER_DB

    db: dict[str, dict] = {}
    with open(settings.CHARACTER_DB_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            candidates = (
                [record]
                if isinstance(record, dict)
                else [r for r in record if isinstance(r, dict)]
                if isinstance(record, list)
                else []
            )
            for c in candidates:
                name = c.get("name", "").strip()
                if name:
                    db[name.lower()] = c

    _CHARACTER_DB = db
    print(f"[OK] Loaded {len(db)} character personas")
    return _CHARACTER_DB


def find_character(query: str) -> dict | None:
    """
    Fuzzy-match a user's text to a character in the database.

    Matching strategy (in order):
      1. Exact match on the full query        ("satoru gojo" → exact hit)
      2. Substring / word match               ("gojo" matches "satoru gojo")
      3. All query words appear in name words ("Gojo Satoru" → "satoru gojo")
      4. Prefer the SHORTEST matching name    (most iconic/primary character)

    Returns None if nothing matches well enough.

    Examples:
      find_character("gojo")                 → Satoru Gojo record
      find_character("Talk to me like Levi") → Levi Ackerman record
      find_character("become zenitsu")       → Zenitsu Agatsuma record
    """
    db = get_character_db()
    q = query.lower().strip()
    if not q:
        return None

    # 1. Exact match
    if q in db:
        return db[q]

    # 2-3. Substring / word match
    q_words = set(q.split())
    candidates: list[tuple[str, dict]] = []
    for name, record in db.items():
        name_words = set(name.split())
        if q in name or q in name_words:
            candidates.append((name, record))
        elif q_words and q_words.issubset(name_words):
            candidates.append((name, record))

    if not candidates:
        return None

    # 4. Prefer shortest matching name (most specific/iconic)
    candidates.sort(key=lambda pair: len(pair[0]))
    return candidates[0][1]
