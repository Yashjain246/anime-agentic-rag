"""
src/tools/jikan.py
──────────────────
Tool 3: Get next airing episode schedule using MyAnimeList Official API.
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
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}

MAL_CLIENT_ID = "bf0992117fbb08b2b7677d46a8b05444"


@tool
def anilist_schedule(anime_title: str) -> str:
    """Get the next airing episode schedule for an anime using MyAnimeList Official API.
    Use this tool EVERY TIME the user asks when the next episode of a show will air, broadcast times, or countdowns.
    Automatically finds the currently airing season. All times in IST.
    Always call this before google_calendar_add to get broadcast_day and broadcast_time.
    Args:
        anime_title: The name AS THE USER WROTE IT — this tool runs its own
            fuzzy search against the real anime database, so pass their
            literal wording (fixing only obvious spelling slips), never a
            different, more familiar-sounding title you guess it might
            mean. A phrase that sounds like a pun on a famous show's name
            can be the literal, correct title of a completely different,
            real, unrelated anime — do not "correct" it to the show it
            reminds you of.
            Worked example: user asks "when does chainsmoker cat next
            episode air?" -> pass anime_title="chainsmoker cat" (this
            tool's own search resolves that to the real show "Yani Neko").
            Do NOT pass "Chainsaw Man" just because "chainsmoker" sounds
            similar to "chainsaw" - that is a coincidence, not evidence of
            what the user meant, and searching the wrong title returns a
            wrong schedule for a real, different, unrelated anime.
    """
    try:
        headers = {"X-MAL-CLIENT-ID": MAL_CLIENT_ID}
        params = {
            "q": anime_title, 
            "limit": 10, 
            "fields": "status,broadcast,alternative_titles"
        }
        r = requests.get(
            "https://api.myanimelist.net/v2/anime",
            headers=headers,
            params=params,
            timeout=10,
        )
        r.raise_for_status()
        results = r.json().get("data", [])

        if not results:
            return f'Anime not found: "{anime_title}".'

        # Find the first one currently airing
        media_node = None
        for item in results:
            if item.get("node", {}).get("status") == "currently_airing":
                media_node = item["node"]
                break
        
        # Fallback to the first result if none are airing
        if not media_node:
            media_node = results[0]["node"]

        title = media_node.get("title")
        status = media_node.get("status")

        if status != "currently_airing":
            return f'"{title}" is not currently airing.'

        bc = media_node.get("broadcast", {})
        day = bc.get("day_of_the_week", "Unknown")
        tstr = bc.get("start_time", "Unknown")

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

        return (
            f"Anime: {title}\n"
            f"Status: Currently Airing\n"
            f"Next episode airs: {air}\n"
            f"Time until airing: {cd}\n"
            f"Broadcast: {day} at {tstr} JST"
        )

    except Exception as e:
        return f"Schedule error: {e}"