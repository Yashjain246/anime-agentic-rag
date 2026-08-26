"""
integrations/external_tools.py
===============================
Source notebook: phase3_final.ipynb

All 5 external API integrations used by the agent, plus their
LangChain @tool wrappers. Import ``TOOLS`` to bind them to the agent.

Tools
-----
1. trace_moe     — identify anime + episode + timestamp from a screenshot
2. omdb          — fetch movie/anime metadata (ratings, plot, year)
3. jikan         — MyAnimeList data (anime search, characters, top charts)
4. tavily        — real-time web search for anime news / release dates
5. google_calendar — add anime to the user's watch-list calendar

Run standalone (smoke-test any tool):
    python -m notebooks.integrations.external_tools --tool omdb --input "One Piece"

Sections
--------
1.  Configuration & shared utilities
2.  Tool: trace.moe
3.  Tool: OMDB
4.  Tool: Jikan (MyAnimeList)
5.  Tool: Tavily web search
6.  Tool: Google Calendar
7.  LangChain @tool wrappers  →  TOOLS list
8.  CLI smoke-test
"""

from __future__ import annotations
import argparse, json, os
from pathlib import Path

# TODO: uncomment once dependencies confirmed
# import requests
# from langchain_core.tools import tool
# from tavily import TavilyClient
# from googleapiclient.discovery import build
# from google.oauth2.credentials import Credentials

# =============================================================================
# 1. Configuration & Shared Utilities
# =============================================================================

OMDB_API_KEY      = os.getenv("OMDB_API_KEY", "")
TAVILY_API_KEY    = os.getenv("TAVILY_API_KEY", "")
GOOGLE_CREDS_FILE = Path("token.json")    # OAuth token — created by calendar_auth.py

OMDB_BASE_URL      = "http://www.omdbapi.com/"
TRACE_MOE_API_URL  = "https://api.trace.moe/search"
JIKAN_BASE_URL     = "https://api.jikan.moe/v4"
CALENDAR_SCOPES    = ["https://www.googleapis.com/auth/calendar"]

# TODO: paste any shared request helpers (rate-limiter, retry decorator) from the notebook


# =============================================================================
# 2. Tool: trace.moe
# =============================================================================

def trace_scene_by_url(image_url: str) -> dict:
    """
    Submit a public screenshot URL to trace.moe and return the raw API response.

    Returns
    -------
    dict
        Top-level trace.moe response with ``result`` list.

    TODO: Replace with actual trace.moe call from phase3_final.ipynb.
    """
    raise NotImplementedError


def trace_scene_by_file(image_path: str) -> dict:
    """
    Upload a local screenshot to trace.moe and return the raw API response.

    TODO: Replace with actual file-upload code from phase3_final.ipynb.
    """
    raise NotImplementedError


def format_trace_result(response: dict) -> str:
    """
    Format the top trace.moe hit as a human-readable string.

    TODO: Replace with actual formatting code from phase3_final.ipynb.
    """
    raise NotImplementedError


# =============================================================================
# 3. Tool: OMDB
# =============================================================================

def omdb_search_by_title(title: str, media_type: str = "series") -> dict:
    """
    Look up an anime / show on OMDB by title.

    Parameters
    ----------
    title : str
        Anime title to search.
    media_type : str
        ``"series"``, ``"movie"``, or ``"episode"``.

    TODO: Replace with actual OMDB call from phase3_final.ipynb.
    """
    raise NotImplementedError


def omdb_get_by_imdb_id(imdb_id: str) -> dict:
    """
    Fetch an OMDB record by IMDb ID (e.g. ``"tt1551369"``).

    TODO: Replace with actual OMDB call from phase3_final.ipynb.
    """
    raise NotImplementedError


# =============================================================================
# 4. Tool: Jikan (MyAnimeList)
# =============================================================================

def jikan_search_anime(query: str, limit: int = 5) -> list[dict]:
    """
    Search MyAnimeList for anime matching *query*.

    TODO: Replace with actual Jikan v4 call from phase3_final.ipynb.
    """
    raise NotImplementedError


def jikan_get_anime(mal_id: int) -> dict:
    """Fetch full anime details by MAL ID."""
    raise NotImplementedError


def jikan_search_characters(query: str, limit: int = 5) -> list[dict]:
    """Search MAL characters matching *query*."""
    raise NotImplementedError


def jikan_top_anime(filter_: str = "airing", limit: int = 10) -> list[dict]:
    """Return the current top / airing anime list from MAL."""
    raise NotImplementedError


# =============================================================================
# 5. Tool: Tavily Web Search
# =============================================================================

def tavily_search(query: str, max_results: int = 5) -> list[dict]:
    """
    Run a Tavily web search and return a list of result dicts.

    Returns
    -------
    list[dict]
        Each item: ``{"title": str, "url": str, "content": str}``.

    TODO: Replace with actual Tavily call from phase3_final.ipynb.
    """
    raise NotImplementedError


# =============================================================================
# 6. Tool: Google Calendar
# =============================================================================

def get_calendar_service():
    """
    Authenticate with Google OAuth and return a Calendar API service object.

    Reads credentials from GOOGLE_CREDS_FILE (created by scripts/calendar_auth.py).

    TODO: Replace with actual auth code from phase3_final.ipynb.
    """
    raise NotImplementedError


def calendar_create_watch_event(
    title: str,
    iso_datetime: str,
    description: str = "",
    duration_minutes: int = 30,
    calendar_id: str = "primary",
) -> dict:
    """
    Create a Google Calendar event to remind the user to watch *title*.

    Parameters
    ----------
    title : str
        Anime / episode title (used as the event summary).
    iso_datetime : str
        Start time as ISO 8601 string, e.g. ``"2025-01-15T20:00:00"``.
    description : str
        Optional event body (synopsis, notes, …).
    duration_minutes : int
        Event length in minutes.
    calendar_id : str
        Target calendar (``"primary"`` for the user's default).

    Returns
    -------
    dict
        The created Google Calendar event resource.

    TODO: Replace with actual event-creation code from phase3_final.ipynb.
    """
    raise NotImplementedError


# =============================================================================
# 7. LangChain @tool Wrappers  →  TOOLS list
# =============================================================================
# Uncomment each @tool once the underlying function is implemented.

# from langchain_core.tools import tool

# @tool
# def tool_trace_scene(image_url: str) -> str:
#     """Identify the anime, episode, and timestamp from a screenshot URL."""
#     return format_trace_result(trace_scene_by_url(image_url))

# @tool
# def tool_lookup_anime_metadata(title: str) -> str:
#     """Fetch OMDB metadata (IMDb rating, year, plot) for an anime title."""
#     result = omdb_search_by_title(title)
#     return json.dumps(result, indent=2)

# @tool
# def tool_search_myanimelist(query: str) -> str:
#     """Search MyAnimeList (via Jikan) for anime information."""
#     results = jikan_search_anime(query)
#     return json.dumps(results[:3], indent=2)

# @tool
# def tool_web_search(query: str) -> str:
#     """Search the web for real-time anime news and information."""
#     results = tavily_search(query)
#     return "\n".join(f"• {r['title']}: {r['content']}" for r in results)

# @tool
# def tool_add_to_watch_calendar(title: str, iso_datetime: str, description: str = "") -> str:
#     """Add an anime to the user's Google Calendar watch-list."""
#     event = calendar_create_watch_event(title, iso_datetime, description)
#     return f"Event created: {event.get('htmlLink', 'n/a')}"


# Export list — populate once @tools are uncommented above
TOOLS: list = []  # TODO: [tool_trace_scene, tool_lookup_anime_metadata, ...]


# =============================================================================
# 8. CLI Smoke-Test
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smoke-test an external integration tool.")
    parser.add_argument("--tool",  required=True,
                        choices=["trace_moe", "omdb", "jikan", "tavily", "calendar"])
    parser.add_argument("--input", required=True, help="Query string or image URL")
    parser.add_argument("--date",  default=None,  help="ISO datetime (calendar tool only)")
    args = parser.parse_args()

    if args.tool == "trace_moe":
        print(format_trace_result(trace_scene_by_url(args.input)))
    elif args.tool == "omdb":
        print(json.dumps(omdb_search_by_title(args.input), indent=2))
    elif args.tool == "jikan":
        print(json.dumps(jikan_search_anime(args.input), indent=2))
    elif args.tool == "tavily":
        for r in tavily_search(args.input):
            print(f"• {r['title']}\n  {r['url']}\n")
    elif args.tool == "calendar":
        if not args.date:
            parser.error("--date is required for the calendar tool")
        print(json.dumps(calendar_create_watch_event(args.input, args.date), indent=2))
