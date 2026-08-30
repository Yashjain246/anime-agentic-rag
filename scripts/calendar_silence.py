"""
scripts/calendar_silence.py
───────────────────────────
One-off maintenance: clear the owner-side reminder from events already on
the shared "Anime RAG Bot Events" calendar.

WHY THIS EXISTS
    google_calendar_add used to write every event with an explicit
    15-minute popup override. In the Calendar API an event's `reminders`
    are per-authenticated-user, and this app always authenticates as the
    single account that owns _PUBLIC_CALENDAR_ID — so that override never
    reached the user who asked for the event. It fired at the calendar's
    owner, once per insert, for every request from every user of the app.
    src/tools/calendar.py now writes no override, but that only affects
    NEW events; the ones already on the calendar keep their own copy of
    the reminder, and because a per-event override takes precedence over
    calendar-level defaults, turning notifications off in the Google
    Calendar settings UI does not silence them. They have to be edited.

WHAT IT DOES NOT DO
    Nothing is deleted, moved, retitled or re-timed. Only the `reminders`
    field is patched. Event IDs and htmlLinks are unchanged, so every link
    the bot has already handed out keeps working, and anyone who has
    copied an event into their own calendar — or subscribed to this one —
    is unaffected, since their notifications come from their own settings,
    never from this override.

USAGE
    python -m scripts.calendar_silence            # dry run, changes nothing
    python -m scripts.calendar_silence --apply    # actually patch
    python -m scripts.calendar_silence --apply --all   # include past events

    Past events can't fire a notification, so by default only events from
    now onward are touched.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone

from src.tools.calendar import _PUBLIC_CALENDAR_ID, _get_calendar_service

SILENT = {"useDefault": False, "overrides": []}


def _is_already_silent(event: dict) -> bool:
    reminders = event.get("reminders") or {}
    return reminders.get("useDefault") is False and not reminders.get("overrides")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true",
                        help="Actually patch events. Without this, only reports what would change.")
    parser.add_argument("--all", action="store_true",
                        help="Include events in the past (they cannot notify, so normally skipped).")
    args = parser.parse_args()

    service = _get_calendar_service()

    list_kwargs = {
        "calendarId": _PUBLIC_CALENDAR_ID,
        "maxResults": 250,
        # Patch the stored event resources themselves, not expanded recurring
        # instances — an instance can't hold its own reminders independently.
        "singleEvents": False,
    }
    if not args.all:
        list_kwargs["timeMin"] = datetime.now(timezone.utc).isoformat()

    scanned = would_change = changed = failed = 0
    page_token = None

    while True:
        if page_token:
            list_kwargs["pageToken"] = page_token
        resp = service.events().list(**list_kwargs).execute()

        for event in resp.get("items", []):
            scanned += 1
            if _is_already_silent(event):
                continue
            would_change += 1
            summary = event.get("summary", "(no title)")
            start = (event.get("start") or {}).get("dateTime", "?")

            if not args.apply:
                print(f"  would silence: {start}  {summary}")
                continue

            try:
                service.events().patch(
                    calendarId=_PUBLIC_CALENDAR_ID,
                    eventId=event["id"],
                    body={"reminders": SILENT},
                ).execute()
                changed += 1
            except Exception as e:  # keep going; one bad event shouldn't abort the sweep
                failed += 1
                print(f"  FAILED {summary}: {e}", file=sys.stderr)

        page_token = resp.get("nextPageToken")
        if not page_token:
            break

    print()
    print(f"scanned:        {scanned}")
    print(f"needed change:  {would_change}")
    if args.apply:
        print(f"silenced:       {changed}")
        if failed:
            print(f"failed:         {failed}")
    else:
        print("\nDry run — nothing was modified. Re-run with --apply to patch.")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
