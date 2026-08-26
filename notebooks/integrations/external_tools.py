"""
integrations/external_tools.py
================================
Source notebook: phase3_final.ipynb

STATUS: intentionally not reimplemented here. All 5 tools from this
notebook are already live in src/tools/, and in some cases the current
version is a genuine improvement over the original — e.g. an earlier
google_calendar_add used a per-run OOB OAuth flow requiring a pasted auth
code, where src/tools/calendar.py uses a persistent stored token with a
shared calendar, and jikan.py fixed a Jikan API v4 field name that the
earlier version got wrong.

THE ACTUAL WORKING VERSION:

    src/tools/trace_moe.py    — screenshot -> anime/episode identification
    src/tools/omdb.py         — episode ratings chart
    src/tools/jikan.py        — airing schedule (anilist_schedule)
    src/tools/registry.py     — anime_news_search (Tavily) + the TOOLS list
    src/tools/calendar.py     — google_calendar_add

Porting the recovered versions over would replace working, already-fixed
code with an older draft. If a specific piece of recovered logic is
useful reference (e.g. the OMDB chart's matplotlib styling), pull just
that piece into the relevant src/tools/*.py file directly rather than
standing up a parallel copy here.
"""
