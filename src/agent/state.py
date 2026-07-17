"""
src/agent/state.py
──────────────────
LangGraph AgentState — the shared backpack passed between every node.

Fields:
  messages:          Full conversation history (auto-appended via operator.add)
  intent:            Set by router_node: LORE/RECOMMEND/TOOL/GENERAL/
                     PERSONA_SWITCH/EPISODE_UPDATE
  anime_name:        Canonical anime name for Lore DB filtering
  current_chapter:   Spoiler cap — chapters above this are blocked
  spoiler_mode:      True = no chapter cap (full DB access)
  persona:           Active character persona name ('Default' or canonical name)
  image_path:        Path to uploaded screenshot (for trace.moe)
  retrieved_context: Text from Lore DB / Recs DB / tool results
  tool_iteration:    Current tool loop iteration counter
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict


class AgentState(TypedDict):
    messages:          Annotated[list, operator.add]  # conversation history
    intent:            str                             # LORE/RECOMMEND/TOOL/GENERAL
    anime_name:        str                             # e.g. "Jujutsu Kaisen"
    current_chapter:   int                             # spoiler cap
    spoiler_mode:      bool                            # True = no cap
    persona:           str                             # bot personality name
    image_path:        str | None                      # uploaded screenshot path
    retrieved_context: str                             # text from DB or tools
    tool_iteration:    int                             # tracks tool loop depth
