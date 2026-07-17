"""
src/episode/detector.py
───────────────────────
Regex-based episode-progress statement detector.

WHY regex (same reasoning as persona switching):
  These are simple, common phrases — catching them cheaply avoids
  an extra LLM call before the router runs.

Patterns covered:
  "I'm on episode 45 of Demon Slayer"
  "I've watched up to episode 20 of Jujutsu Kaisen"
  "I am currently on episode 12 of Attack on Titan"
  "I'm caught up to AOT episode 30"     (anime-name-first variant)
  "I'm on Demon Slayer episode 26"      (anime-name-first variant)
  "I'm at episode 8"                    (no anime → use state)

Deliberately does NOT match:
  "What happens in episode 5?"          (no progress verb → LORE)
"""

from __future__ import annotations

import re

from src.episode.normalizer import normalize_anime_name

# ── Pattern: "I'm on episode 45 of Demon Slayer" ─────────────────────────────
_EPISODE_FIRST_PATTERN = (
    r"(?:i'?m|i am|i've|i have|currently)\s*"
    r"(?:on|at|up\s*to|watched(?:\s*up)?\s*to|caught\s*up\s*to|"
    r"currently\s*on|currently\s*at)\s*"
    r"episode\s*(\d+)"
    r"(?:\s*of\s+([a-zA-Z0-9 :\-'\u2019]+?))?"
    r"[.!?,]*\s*$"
)

# ── Pattern: "I'm on Demon Slayer episode 30" ─────────────────────────────────
_ANIME_FIRST_PATTERN = (
    r"(?:i'?m|i am|i've|i have|currently)\s*"
    r"(?:on|at|up\s*to|watched(?:\s*up)?\s*to|caught\s*up\s*to)\s*"
    r"([a-zA-Z0-9 :\-'\u2019]+?)\s*episode\s*(\d+)"
    r"[.!?,]*\s*$"
)


def detect_episode_progress(user_message: str) -> tuple[int, str | None] | None:
    """
    Detect "I'm on episode N [of <anime>]" style statements.

    Returns:
      (episode_number, canonical_anime_name_or_None)
        → anime is None if not mentioned or alias not recognized
      None if no progress statement is detected at all.
    """
    msg = user_message.strip()

    m = re.search(_EPISODE_FIRST_PATTERN, msg, re.IGNORECASE)
    if m:
        ep_num = int(m.group(1))
        anime = normalize_anime_name(m.group(2)) if m.group(2) else None
        return ep_num, anime

    m = re.search(_ANIME_FIRST_PATTERN, msg, re.IGNORECASE)
    if m:
        anime = normalize_anime_name(m.group(1))
        ep_num = int(m.group(2))
        return ep_num, anime

    return None
