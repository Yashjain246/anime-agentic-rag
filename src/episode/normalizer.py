"""
src/episode/normalizer.py
─────────────────────────
Maps user-typed anime aliases to canonical names used across the pipeline.

WHY: Users type "AOT", "JJK", "Kimetsu no Yaiba" etc. but the Lore DB
and EPISODE_MAPPING use canonical English titles ("Attack on Titan",
"Jujutsu Kaisen", "Demon Slayer"). This maps every known alias to the
canonical name used everywhere else.
"""

from __future__ import annotations

ANIME_NAME_ALIASES: dict[str, str] = {
    # Demon Slayer
    "demon slayer": "Demon Slayer",
    "kimetsu no yaiba": "Demon Slayer",
    "kny": "Demon Slayer",
    # Jujutsu Kaisen
    "jujutsu kaisen": "Jujutsu Kaisen",
    "jjk": "Jujutsu Kaisen",
    # Attack on Titan
    "attack on titan": "Attack on Titan",
    "aot": "Attack on Titan",
    "shingeki no kyojin": "Attack on Titan",
    "snk": "Attack on Titan",
    # Chainsaw Man
    "chainsaw man": "Chainsaw Man",
    "csm": "Chainsaw Man",
    # Frieren
    "frieren": "Frieren",
    "sousou no frieren": "Frieren",
    "frieren beyond journey's end": "Frieren",
    # Solo Leveling
    "solo leveling": "Solo Leveling",
    "나 혼자만 레벨업": "Solo Leveling",
}

# Canonical names supported by the Lore DB (chroma_anime_db)
SUPPORTED_ANIME = {
    "Demon Slayer",
    "Jujutsu Kaisen",
    "Attack on Titan",
    "Chainsaw Man",
}

# Canonical names in episode_mapping (includes Frieren, Solo Leveling)
EPISODE_MAPPED_ANIME = {
    "Demon Slayer",
    "Jujutsu Kaisen",
    "Attack on Titan",
    "Chainsaw Man",
    "Frieren",
    "Solo Leveling",
}


def normalize_anime_name(raw: str | None) -> str | None:
    """
    Map a user-typed anime name/alias to its canonical form.
    Returns None if the alias is not recognized.

    Examples:
      normalize_anime_name("jjk")            → "Jujutsu Kaisen"
      normalize_anime_name("Demon Slayer")   → "Demon Slayer"
      normalize_anime_name("something else") → None
    """
    if not raw:
        return None
    return ANIME_NAME_ALIASES.get(raw.strip().lower())


def get_all_canonical_names() -> list[str]:
    """Returns a sorted list of all canonical anime names for UI dropdowns."""
    return sorted(SUPPORTED_ANIME)
