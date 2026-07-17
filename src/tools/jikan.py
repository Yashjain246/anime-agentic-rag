"""
src/tools/jikan.py
──────────────────
Tool 3: Get next airing episode schedule using Jikan (MyAnimeList API).
All times are converted to IST (Indian Standard Time, UTC+5:30).
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

import requests
from langchain_core.tools import tool

_IST = timezone(timedelta(hours=5, minutes=30))
_JST = timezone(timedelta(hours=9))

_DAY_MAP = {
    "mondays": 0, "tuesdays": 1, "wednesdays": 2, "thursdays": 3,
    "fridays": 4, "saturdays": 5, "sundays": 6,
}


@tool
def anilist_schedule(anime_title: str) -> str:
    """Get the next airing episode schedule for an anime using Jikan (MyAnimeList).
    Automatically finds the currently airing season. All times in IST.
    Always call this before google_calendar_add to get broadcast_day and broadcast_time.
    Args:
        anime_title: Name of the anime (English or Japanese).
    """
    try:
        r = requests.get(
            "https://api.jikan.moe/v4/anime",
            params={"q": anime_title, "status": "airing", "limit": 5},
            timeout=10,
        )
        r.raise_for_status()
        results = r.json().get("data", [])

        if not results:
            time.sleep(1)
            r2 = requests.get(
                "https://api.jikan.moe/v4/anime",
                params={"q": anime_title, "limit": 5},
                timeout=10,
            )
            results = r2.json().get("data", [])

        if not results:
            return f'Anime not found: "{anime_title}".'

        media = next(
            (x for x in results if x.get("status") == "Currently Airing"),
            results[0],
        )
        title = media.get("title_english") or media.get("title")
        airing = media.get("airing", False)
        mal_id = media.get("mal_id")

        if not airing:
            return f'"{title}" is not currently airing.'

        bc = media.get("broadcast", {})
        day = bc.get("day", "Unknown")
        tstr = bc.get("time", "Unknown")

        wd = _DAY_MAP.get(day.lower(), -1)
        if wd != -1 and tstr != "Unknown":
            h, m = map(int, tstr.split(":"))
            now = datetime.now(_JST)
            ahead = (wd - now.weekday()) % 7
            if ahead == 0 and now.hour >= h:
                ahead = 7
            nxt = (now + timedelta(days=ahead)).replace(
                hour=h, minute=m, second=0, microsecond=0
            )
            air = nxt.astimezone(_IST).strftime("%A, %d %B %Y at %H:%M IST")
            secs = int((nxt - now).total_seconds())
            cd = f"{secs // 86400}d {(secs % 86400) // 3600}h"
        else:
            air = f"{day} at {tstr} JST"
            cd = "Unknown"

        time.sleep(1.5)
        return (
            f"Anime: {title}\n"
            f"Status: Currently Airing\n"
            f"Next episode airs: {air}\n"
            f"Time until airing: {cd}\n"
            f"Broadcast: {day} at {tstr} JST\n"
            f"MAL page: https://myanimelist.net/anime/{mal_id}"
        )

    except Exception as e:
        return f"Schedule error: {e}"