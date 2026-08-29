"""
src/agent/state.py
──────────────────
LangGraph AgentState — the shared backpack passed between every node.

Fields:
  messages:          Full conversation history (auto-appended via operator.add)
  intent:            Primary intent, set by router_node — intents[0], kept
                     for backward compat with anything expecting one label:
                     LORE/RECOMMEND/TOOL/GENERAL/PERSONA_SWITCH/EPISODE_UPDATE
  intents:           Every intent router_node found in the message, in the
                     order asked (e.g. ["LORE", "TOOL"] for a compound
                     question). Drives fan-out to multiple retrieval/tool
                     nodes in the same turn — see _route_after_router.
  anime_name:        Canonical anime name for Lore DB filtering
  current_chapter:   Spoiler cap — chapters above this are blocked
  spoiler_mode:      True = no chapter cap (full DB access)
  persona:           Active character persona name ('Default' or canonical name)
  image_path:        Path to uploaded screenshot (for trace.moe)
  retrieved_context: One {"source": "LORE"/"RECS"/"TOOL", "text": ...} dict
                     per node that actually ran this turn. A list with an
                     operator.add reducer (not a plain str) specifically so
                     that when router_node fans out to more than one of
                     lore_node/recs_node/tools_node in parallel, each node's
                     contribution appends instead of racing to overwrite the
                     same field — LangGraph raises on two parallel branches
                     writing a plain field in the same superstep otherwise.
  tool_iteration:    Current tool loop iteration counter
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict


class AgentState(TypedDict):
    messages:          Annotated[list, operator.add]  # conversation history
    intent:            str                             # primary intent (intents[0])
    intents:           list[str]                       # every intent this turn, in order
    anime_name:        str                             # e.g. "Jujutsu Kaisen"
    current_chapter:   int                             # spoiler cap
    spoiler_mode:      bool                            # True = no cap
    persona:           str                             # bot personality name
    image_path:        str | None                      # uploaded screenshot path
    retrieved_context: Annotated[list[dict], operator.add]  # [{"source", "text"}, ...]
    tool_iteration:    int                             # tracks tool loop depth
