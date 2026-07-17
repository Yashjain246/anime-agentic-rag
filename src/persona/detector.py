"""
src/persona/detector.py
───────────────────────
Regex-based persona-switch detection.

WHY regex instead of an LLM call:
  Persona switches are simple, common phrases. Catching them with regex
  is instant and free — no API call needed. The router still runs
  afterwards for normal intent classification on other messages.

Patterns covered:
  "talk to me like Gojo"       "talk like Levi"
  "respond as Zenitsu"         "be Denji"
  "become Akaza"               "switch to Tanjiro"
  "act like Nezuko"            "pretend to be Muzan"
  "go back to normal"          "reset persona"
  "stop being <x>"             → Default
"""

from __future__ import annotations

import re

from src.persona.character_db import find_character

# ── Persona-switch patterns ───────────────────────────────────────────────────
PERSONA_SWITCH_PATTERNS = [
    r"(?:talk|speak|respond|reply)\s*(?:to me)?\s*(?:like|as)\s+(.+)",
    r"(?:be|become|act\s+like|pretend\s+(?:to\s+be|you(?:'re|\s+are))|switch\s+to|switch\s+persona\s+to)\s+(.+)",
]

PERSONA_RESET_PATTERNS = [
    r"(?:go\s+back\s+to\s+normal|reset\s*(?:persona)?|stop\s+being|exit\s+persona|normal\s+mode|default\s+mode|back\s+to\s+normal)",
]


def detect_persona_switch(user_message: str) -> str | None:
    """
    Check if the user's message is asking to switch (or reset) persona.

    Returns:
      'Default'         → user asked to reset to the neutral assistant
      '<Character Name>' → the matched character's canonical name
      None              → no persona-switch request detected
    """
    msg = user_message.strip().lower()

    # Check for reset requests first
    for pattern in PERSONA_RESET_PATTERNS:
        if re.search(pattern, msg):
            return "Default"

    # Check for "talk like X" / "be X" style requests
    for pattern in PERSONA_SWITCH_PATTERNS:
        match = re.search(pattern, msg, re.IGNORECASE)
        if match:
            name_query = match.group(1).strip()
            # Strip trailing punctuation / filler words
            name_query = re.sub(r"[.!?]+$", "", name_query).strip()
            name_query = re.sub(r"^(?:to\s+)?", "", name_query).strip()

            character = find_character(name_query)
            if character:
                return character["name"]  # canonical name e.g. "Satoru Gojo"

    return None
