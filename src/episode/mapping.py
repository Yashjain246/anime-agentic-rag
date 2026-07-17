"""
src/episode/mapping.py
──────────────────────
Episode-to-chapter conversion engine.

WHY this exists:
  The spoiler firewall caps the Lore DB by MANGA CHAPTER NUMBER.
  But most anime viewers know what EPISODE they last watched, not
  which chapter. This module bridges the gap:
    "I'm on episode 45 of Demon Slayer" → current_chapter = 95

Data quirks handled:
  - 2 entries have episode_number == "Movie" instead of an int.
    These are skipped in linear episode numbering.
  - Filler episodes (chapters_covered=[]) don't raise the chapter cap.
  - Coverage: 6 anime in mapping, but Lore DB only has 4.
    episode_to_chapter() works for all 6; lore_node returns
    NO_CONTEXT_FOUND for Frieren/Solo Leveling as expected.
"""

from __future__ import annotations

from collections import defaultdict
import json

from config.settings import settings

_EPISODE_MAPPING: dict[str, list[dict]] | None = None


def get_episode_mapping() -> dict[str, list[dict]]:
    """
    Returns {anime_name: [episode records sorted by episode_number]}.
    Loaded once at first access.
    """
    global _EPISODE_MAPPING
    if _EPISODE_MAPPING is not None:
        return _EPISODE_MAPPING

    mapping: dict[str, list[dict]] = defaultdict(list)

    with open(settings.EPISODE_MAPPING_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            mapping[record["anime_name"]].append(record)

    def _sort_key(r: dict):
        ep = r["episode_number"]
        return (0, ep) if isinstance(ep, int) else (1, 0)

    for anime in mapping:
        mapping[anime].sort(key=_sort_key)

    _EPISODE_MAPPING = dict(mapping)
    print(f"[OK] Loaded episode mappings for {len(_EPISODE_MAPPING)} anime: {list(_EPISODE_MAPPING.keys())}")
    return _EPISODE_MAPPING


def episode_to_chapter(anime_name: str, episode_number: int) -> int | None:
    """
    Convert "watched up to episode N of <anime>" into a manga chapter cap.

    Returns:
      int  → highest chapter adapted by episodes 1..N
      None → anime_name not found in episode mapping at all

    Examples:
      episode_to_chapter("Demon Slayer", 12)  → 24
      episode_to_chapter("Demon Slayer", 27)  → 54  (ep27 is filler)
      episode_to_chapter("Frieren", 10)       → 28
    """
    episodes = get_episode_mapping().get(anime_name)
    if episodes is None:
        return None

    max_chapter = 0
    found_any = False

    for record in episodes:
        ep = record["episode_number"]
        if not isinstance(ep, int):
            continue  # skip "Movie" entries
        if ep > episode_number:
            break
        found_any = True
        if record["chapters_covered"]:
            max_chapter = max(max_chapter, max(record["chapters_covered"]))

    if not found_any:
        return 0  # no chapters unlocked yet

    return max_chapter
