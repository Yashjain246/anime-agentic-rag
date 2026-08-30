# notebooks/

Portable Python ports of the Colab notebooks used to build the anime-RAG pipeline.
Organised by **domain responsibility** so any engineer knows exactly where to look.

> **Status**: `ingestion/` contains 4 real, runnable modules. Everything else
> either points to an already-working equivalent elsewhere in the repo, or is
> honestly marked as unresolved rather than faked — see the per-file status
> below.

---

## Folder Map

| Folder | Responsibility | Status |
|---|---|---|
| [`ingestion/`](ingestion/) | Raw data collection — scraping, extraction, mapping | 4 real modules; `chapter_patcher.py` unresolved (see file) |
| [`indexing/`](indexing/) | Building vector databases from raw data | Points to `scripts/rebuild_*_db.py` — the actual working version |
| [`retrieval/`](retrieval/) | Hybrid search over the vector databases | Points to `src/rag/` — the actual working version |
| [`integrations/`](integrations/) | External API clients (all 5 tools) | Points to `src/tools/` — the actual working version |
| [`agent/`](agent/) | LangGraph agent orchestration | Points to `src/agent/` — the actual working version |

---

## File Index

```
notebooks/
├── ingestion/
│   ├── manga_scraper.py           REAL. Scrapes wikitext, summarises via Gemini
│   │                              → data/manga_chapters (3).jsonl
│   ├── character_extractor.py     REAL. Extracts character personas via Gemini
│   │                              → data/all_characters (4).jsonl
│   ├── episode_mapper.py          REAL. Maps episodes/movies to manga chapters
│   │                              via Gemini → data/episode_mapping (1).jsonl
│   ├── anime_synopsis_scraper.py  REAL. Pulls anime synopses from the Jikan API
│   │                              (no LLM) → data/anime_desc (1).jsonl
│   └── chapter_patcher.py         UNRESOLVED — source notebook never identified.
│                                  Read the file: likely unnecessary, see below.
│
├── indexing/
│   ├── lore_db_builder.py     Points to scripts/rebuild_lore_db.py (the current
│   │                          fastembed/CPU stack — see file for why the GPU/bge-large
│   │                          version isn't used)
│   └── recs_db_builder.py     Points to scripts/rebuild_recs_db.py, same reason
│
├── retrieval/
│   └── hybrid_search.py       Points to src/rag/ (retriever.py, bm25_index.py, embeddings.py)
│
├── integrations/
│   └── external_tools.py      Points to src/tools/ — already ahead of this notebook's version
│
└── agent/
    └── langgraph_agent.py     Points to src/agent/ — already a complete implementation
```

Every `ingestion/` module is resume-safe: re-running it skips whatever's
already in the target JSONL instead of re-processing (and re-billing) from
scratch. Run any of them with `-h` for its exact CLI.

---

## Relationship to `src/`

`src/` is the **production runtime** consumed by the Streamlit app, and for
indexing/retrieval/integrations/agent it is also now the **only** implementation —
the `notebooks/` files for those stages are deliberately just pointers, not a
second copy, so there's one place to change instead of two that can drift apart.

`ingestion/` is the exception: generating the raw data has no `src/` equivalent
(the production app only ever *reads* the JSONL files, it never generates them),
so these 4 modules are the actual, standalone tools for extending the dataset —
new anime, or filling gaps in existing ones.

| `notebooks/` file | Status | Real implementation lives at |
|---|---|---|
| `ingestion/manga_scraper.py` | Real, runnable | *(this file — no src/ equivalent)* |
| `ingestion/character_extractor.py` | Real, runnable | *(this file — no src/ equivalent)* |
| `ingestion/episode_mapper.py` | Real, runnable | *(this file — no src/ equivalent)* |
| `ingestion/anime_synopsis_scraper.py` | Real, runnable | *(this file — no src/ equivalent)* |
| `ingestion/chapter_patcher.py` | Unresolved | Probably unnecessary — see file |
| `indexing/lore_db_builder.py` | Pointer | `scripts/rebuild_lore_db.py` |
| `indexing/recs_db_builder.py` | Pointer | `scripts/rebuild_recs_db.py` |
| `retrieval/hybrid_search.py` | Pointer | `src/rag/retriever.py`, `bm25_index.py`, `embeddings.py` |
| `integrations/external_tools.py` | Pointer | `src/tools/*.py` |
| `agent/langgraph_agent.py` | Pointer | `src/agent/*.py` |

## Still missing

Nothing — every data-generation stage now has a real, working module. The one
open item is `ingestion/chapter_patcher.py`, whose source was never identified
and whose function is likely already covered by `manga_scraper.py`'s
`--chapters` flag.
