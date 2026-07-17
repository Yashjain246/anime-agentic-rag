"""
src/tools/calendar.py
─────────────────────
Tool 5: Add anime episode airing events to Google Calendar.

DEPLOYMENT NOTE:
  This tool uses OAuth with a refresh token stored in token.json.
  For single-account deployment (your account only):
    1. Run `python scripts/calendar_auth.py` once locally
    2. This creates token.json with a persistent refresh token
    3. Upload token.json as a secret/env var to Render
    4. The deployed app will silently refresh the token — fully autonomous

  The ENABLE_CALENDAR_TOOL setting controls whether this tool is
  included in TOOLS. Set it to False in .env for users who haven't
  completed the one-time OAuth setup.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from langchain_core.tools import tool

from config.settings import settings

_IST = timezone(timedelta(hours=5, minutes=30))
_JST = timezone(timedelta(hours=9))

_DAY_MAP = {
    "mondays": 0, "tuesdays": 1, "wednesdays": 2, "thursdays": 3,
    "fridays": 4, "saturdays": 5, "sundays": 6,
}


def _get_calendar_service():
    """Build Google Calendar API service using stored token.json."""
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build as google_build
    import base64
    import json

    SCOPES = ["https://www.googleapis.com/auth/calendar.events"]
    creds = None

    if settings.CALENDAR_TOKEN_B64:
        try:
            token_json = base64.b64decode(settings.CALENDAR_TOKEN_B64).decode("utf-8")
            token_info = json.loads(token_json)
            creds = Credentials.from_authorized_user_info(token_info, SCOPES)
        except Exception as e:
            print(f"[Calendar] Failed to parse CALENDAR_TOKEN_B64: {e}")

    if not creds and settings.CALENDAR_TOKEN_PATH.exists():
        creds = Credentials.from_authorized_user_file(
            str(settings.CALENDAR_TOKEN_PATH), SCOPES
        )
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())  # Silent auto-refresh — autonomous in production
            with open(settings.CALENDAR_TOKEN_PATH, "w") as f:
                f.write(creds.to_json())
        else:
            raise RuntimeError(
                "No valid Google Calendar credentials found. "
                "Run `python scripts/calendar_auth.py` first."
            )
    return google_build("calendar", "v3", credentials=creds)


@tool
def google_calendar_add(
    anime_title: str,
    broadcast_day: str,
    broadcast_time: str,
) -> str:
    """Add an anime episode airing to Google Calendar.
    Always call anilist_schedule first to get broadcast_day and broadcast_time.
    Args:
        anime_title: Anime name.
        broadcast_day: e.g. 'Thursdays'.
        broadcast_time: HH:MM JST format.
    """
    if not settings.ENABLE_CALENDAR_TOOL:
        return (
            "Google Calendar integration is not configured. "
            "Contact the bot admin to enable it."
        )
    try:
        service = _get_calendar_service()
        wd = _DAY_MAP.get(broadcast_day.lower())
        if wd is None:
            return f'Could not parse day: "{broadcast_day}"'

        h, m = map(int, broadcast_time.split(":"))
        now = datetime.now(_JST)
        ahead = (wd - now.weekday()) % 7
        if ahead == 0 and now.hour >= h:
            ahead = 7
        nxt = (now + timedelta(days=ahead)).replace(
            hour=h, minute=m, second=0, microsecond=0
        )
        utc_s = nxt.astimezone(timezone.utc)
        utc_e = utc_s + timedelta(minutes=25)
        air = nxt.astimezone(_IST).strftime("%A, %d %B %Y at %H:%M IST")

        event = {
            "summary": f"🎌 {anime_title} — New Episode",
            "description": "Added by Anime Bot.",
            "start": {"dateTime": utc_s.isoformat(), "timeZone": "UTC"},
            "end": {"dateTime": utc_e.isoformat(), "timeZone": "UTC"},
            "reminders": {
                "useDefault": False,
                "overrides": [{"method": "popup", "minutes": 15}],
            },
        }
        created = service.events().insert(calendarId="c0683492932a088b94531c3e63a1523e81cb02ad7ed9c35ac5cc2711b70d99dd@group.calendar.google.com", body=event).execute()
        return (
            f"✅ Event created on the Public Anime Calendar!\n"
            f"Title: {anime_title}\n"
            f"Time: {air}\n"
            f"Link: {created.get('htmlLink')}"
        )
    except Exception as e:
        return f"Calendar error: {e}"
