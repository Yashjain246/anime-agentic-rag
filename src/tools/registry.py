"""
src/tools/registry.py
─────────────────────
Centralized tool registry. Import TOOLS from here — never build
the list ad-hoc in graph nodes.
"""

from __future__ import annotations

from config.settings import settings
from src.tools.trace_moe import trace_moe_vision
from src.tools.omdb import omdb_graph_generator
from src.tools.jikan import anilist_schedule
from src.tools.calendar import google_calendar_add

from langchain_tavily import TavilySearch

# ── Tavily news search ────────────────────────────────────────────────────────
anime_news_search = TavilySearch(
    max_results=4,
    name="anime_news_search",
    description=(
        "Search for recent anime news, announcements, trailers, and updates. "
        "Use this for questions about upcoming new seasons, cast announcements, "
        "or general current events in the anime world. "
        "CRITICAL: Do NOT use this to check the release date or broadcast time of weekly episodes for currently airing anime! Use 'anilist_schedule' for that."
    ),
)

# ── Build TOOLS list ──────────────────────────────────────────────────────────
def get_tools() -> list:
    """
    Returns the list of active tools.
    Google Calendar is conditionally included based on ENABLE_CALENDAR_TOOL.
    """
    tools = [
        trace_moe_vision,
        omdb_graph_generator,
        anilist_schedule,
        anime_news_search,
    ]
    if settings.ENABLE_CALENDAR_TOOL:
        tools.append(google_calendar_add)
    return tools


TOOLS = get_tools()
