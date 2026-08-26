"""
ingestion/chapter_patcher.py
============================
Source notebook: not identified.

This was meant to be a small notebook that patched a couple of chapters
which failed to scrape during the main run. The source was never pinned
down, so this file intentionally stays unimplemented rather than
fabricate logic for a notebook nobody has actually read.

It's likely unnecessary regardless: manga_scraper.py's
extract_manga_chapters() already skips any (anime_name, chapter_number)
already present in the output JSONL, and accepts an explicit --chapters
list — that resume + targeted-retry behaviour is the patch mechanism
this file would otherwise provide.

    python -m notebooks.ingestion.manga_scraper --audit

    python -m notebooks.ingestion.manga_scraper --anime "Demon Slayer" --chapters 71 96
    python -m notebooks.ingestion.manga_scraper --anime "Chainsaw Man" --chapters 136 232
"""
