---
title: Anime Agentic RAG
emoji: 🎌
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.35.0
app_file: app/streamlit_app.py
pinned: false
---

# Anime Agentic RAG

An **agentic RAG (Retrieval-Augmented Generation)** system for anime & manga, built with LangGraph, Gemini and ChromaDB — with a spoiler-aware retrieval layer, multi-intent routing, and a CI-gated evaluation suite.

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776ab?logo=python)](https://python.org)
[![LangGraph 1.2](https://img.shields.io/badge/LangGraph-1.2-green)](https://github.com/langchain-ai/langgraph)
[![Streamlit 1.62](https://img.shields.io/badge/Streamlit-1.62-red)](https://streamlit.io)
[![Eval gate](https://img.shields.io/badge/eval-59%20cases%20%C2%B7%208%20evaluators-blueviolet)](#evaluation--llmops)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## Features

| Feature | Details |
|---------|---------|
| **Multi-intent agentic routing** | LangGraph state machine. A single message can carry several intents (LORE / RECOMMEND / TOOL / GENERAL) — they fan out to their nodes **in parallel** and merge into one reply |
| **Per-intent query decomposition** | The router returns not just *which* intents fired but *which part of the message* belongs to each, so one half of a compound question can't contaminate the other's retrieval |
| **Spoiler firewall** | ChromaDB metadata filter applied at the DB level, not post-retrieval — spoiler chapters are never scored in the first place |
| **Hybrid RAG** | MultiQuery → 60% dense + 40% BM25 ensemble → FlashRank reranker |
| **719 character personas** | Dynamic persona engine built from extracted personality data |
| **Episode-to-chapter mapping** | "I'm on episode 45 of Demon Slayer" → manga chapter cap set automatically |
| **5 tools** | trace.moe screenshot ID, OMDB ratings charts, AniList/MAL airing schedule, Tavily news, Google Calendar |
| **Persistent chat history** | Per-user anonymous browser identity; SQLite locally, PostgreSQL (Supabase) in production |
| **LangSmith tracing** | Full observability — set `LANGSMITH_TRACING=true` |
| **CI-gated evaluation** | 59-case golden dataset, 8 evaluators, thresholds enforced on every PR ([details](#evaluation--llmops)) |
| **Lightweight footprint** | fastembed + flashrank (ONNX, no torch/GPU) — ~850MB peak RSS |

## Anime Coverage

| Anime | Lore DB | Episode Mapping |
|-------|---------|-----------------|
| Demon Slayer | Yes | Yes |
| Jujutsu Kaisen | Yes | Yes |
| Attack on Titan | Yes | Yes |
| Chainsaw Man | Yes | Yes |
| Frieren | No | Yes |
| Solo Leveling | No | Yes |

Anything outside the Lore DB is explicitly refused rather than answered from the model's own knowledge — see `unsupported_anime_refused` in the eval suite.

---

## Architecture

```
User Message
     │
     ▼
[persona_node] ──── PERSONA_SWITCH? ──► END      (short-circuits; no retrieval)
     │ NO
     ▼
[episode_node] ──── EPISODE_UPDATE? ──► END      (short-circuits; sets chapter cap)
     │ NO
     ▼
[router_node] ── RouterOutput(items=[{intent, query}, ...]) ── multi-label
     │
     │   fans out IN PARALLEL to every intent present
     ├──────────────► [lore_node]   (scoped query, spoiler-filtered retrieval)
     ├──────────────► [recs_node]   (scoped query)
     ├──────────────► [tools_node]  (scoped query, ≤ MAX_TOOL_ITERATIONS)
     │
     └──────────────► [respond_node] ──► END
                       merges every branch's context into one persona-voiced reply
```

Parallel branches write into `retrieved_context`, which is an `Annotated[list, operator.add]` reducer — without that, concurrent writes from two nodes in the same superstep collide and LangGraph raises. `respond_node` then assembles one section per intent, so a compound message produces a single coherent answer rather than several stitched-together ones.

---

## Evaluation & LLMOps

Retrieval quality and routing correctness are not things you can eyeball reliably, so the repo carries a versioned golden dataset and an automated gate.

### The golden dataset

`data/eval/golden_dataset.jsonl` — **59 cases**, generated from the real manga-chapter corpus (not hand-invented), so expected chapters and facts match what's actually in the vector store.

| Category | Cases | What it checks |
|----------|-------|----------------|
| `LORE` | 20 | Grounded chapter lore retrieval |
| `COMPOUND` | 9 | Multi-intent messages routed and answered in full |
| `RECOMMEND` | 6 | Recommendations drawn from the recs DB |
| `GENERAL` | 6 | Casual chat, no retrieval expected |
| `LORE_SPOILER_BLOCK` | 5 | Spoiler content correctly withheld |
| `TOOL` | 5 | Live-data lookups routed to the right tool |
| `EPISODE_UPDATE` | 4 | Episode → chapter cap conversion |
| `PERSONA_SWITCH` | 4 | Persona detection and switching |

Uploaded to LangSmith idempotently (`scripts/eval/upload_dataset.py`) using `uuid5`-derived stable IDs, so re-running never creates duplicates.

### The evaluators

`src/eval/evaluators.py` — **7 deterministic + 1 LLM-as-judge**:

| Evaluator | Type | Checks |
|-----------|------|--------|
| `intent_match` | deterministic | Router produced exactly the expected intent set (compared as sets) |
| `every_intent_has_source` | deterministic | Every retrieving intent produced a matching context block |
| `retrieval_hit` | deterministic | Expected anime + chapter actually present in retrieved lore |
| `spoiler_not_leaked` | deterministic | Blocked chapters' reference facts absent from the reply |
| `unsupported_anime_refused` | deterministic | `ANIME_NOT_SUPPORTED` sentinel present for out-of-scope series |
| `persona_switch_correct` | deterministic | Resolved persona matches expected |
| `episode_cap_correct` | deterministic | Chapter cap and anime both correct |
| `faithfulness_and_relevance` | LLM judge | Reply grounded in retrieved context and addresses every part asked |

The judge is a Gemini call with Pydantic structured output (`grounded`, `on_topic`, `reasoning`). Its prompt encodes two conventions it otherwise mis-grades: the in-character "that hasn't been revealed yet" phrasing used when context has a gap, and the deliberate episode→chapter conversion in `EPISODE_UPDATE` replies.

### The CI gate

`.github/workflows/eval.yml` runs the suite and **fails the build** if any metric drops below its floor:

```python
MIN_SCORES = {
    "spoiler_not_leaked":         1.00,   # non-negotiable
    "unsupported_anime_refused":  1.00,   # non-negotiable
    "every_intent_has_source":    0.95,
    "faithfulness_and_relevance": 0.80,
    "intent_match":               0.75,
    "episode_cap_correct":        0.75,
    "persona_switch_correct":     0.75,
    "retrieval_hit":              0.50,
}
```

Thresholds sit deliberately *below* the measured baseline rather than on it — a gate level with the baseline just fails flakily against normal LLM-judge variance.

**Latest run** (59 cases, 0 execution errors):

| Metric | Score |
|--------|-------|
| `intent_match` | 1.00 |
| `every_intent_has_source` | 1.00 |
| `faithfulness_and_relevance` | 1.00 |
| `persona_switch_correct` | 1.00 |
| `episode_cap_correct` | 1.00 |
| `spoiler_not_leaked` | 1.00 |
| `unsupported_anime_refused` | 1.00 |
| `retrieval_hit` | 0.60 |

### Cost control

The gate is **pull-request only**, never on push — one full run is ~216 real Gemini calls against a free-tier quota. Two further measures keep it inside the limits:

- **20s pacing between cases** (`_PACING_SECONDS`), bringing throughput to ~8.5 calls/min against the free tier's 15/min ceiling. Before pacing, runs peaked at 27/min and the resulting 429s were being scored as genuine failures.
- **A dedicated `EVAL_GOOGLE_API_KEY` secret**, separate from the key the live app uses, so evaluation never competes with real users for the same daily budget.

### Known limitations

- **`retrieval_hit` sits at 0.60.** All misses are bare *"what happens in Chapter N"* queries, which carry no topical content for either embeddings or BM25 to match on — the retriever finds the right series but not the right chapter. Fixing it needs a retrieval redesign (e.g. metadata-aware chapter lookup), not prompt tuning, so it is tracked rather than papered over with a lower threshold.
- **CI currently evaluates `gemini-3.5-flash-lite` while production defaults to `gemini-3.1-flash-lite`** — a quota workaround. A passing gate therefore does not strictly prove the deployed model configuration. The judge is also a Gemini call, so its strictness varies with the model behind it; treat judge-scored metrics as directional across model changes.
- **Recommendation replies sometimes embellish** beyond the retrieved synopsis with the model's own knowledge of a series. Real, and fixing it is prompt work that needs its own eval cycle.

### Running it locally

```bash
python -m scripts.eval.upload_dataset     # idempotent; safe to re-run
python -m scripts.eval.run_eval --prefix local
```

Requires `LANGSMITH_API_KEY`. Expect ~35 minutes and ~216 Gemini calls.

---

## Quick Start

### 1. Prerequisites

Python 3.11+ (CI and `requirements-lock.txt` target 3.13). No GPU required — embeddings and reranking run on CPU via ONNX.

### 2. Clone & set up

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

pip install -r requirements.txt        # or requirements-lock.txt for pinned versions
```

### 3. Configure

```bash
copy .env.example .env          # then fill in your keys
```

| Variable | Required | Where to get it |
|----------|----------|-----------------|
| `GOOGLE_API_KEY` | Yes | [aistudio.google.com](https://aistudio.google.com) |
| `TAVILY_API_KEY` | Yes | [tavily.com](https://tavily.com) (free tier) |
| `OMDB_API_KEY` | Yes | [omdbapi.com](http://www.omdbapi.com/apikey.aspx) (free) |
| `MAL_CLIENT_ID` | Optional | [myanimelist.net/apiconfig](https://myanimelist.net/apiconfig) — without it, airing-schedule lookups are disabled |
| `LANGSMITH_API_KEY` | Optional | [smith.langchain.com](https://smith.langchain.com) — free tier, 5,000 traces/month |
| `ADMIN_PASSWORD` | Optional | Your choice; empty disables the admin panel entirely |

> Credential values are whitespace-stripped on load, and `os.environ` is normalized at import, because secrets pasted into GitHub Actions / Streamlit secrets pick up trailing newlines very easily. See [Key design decisions](#key-design-decisions).

### 4. Copy data files

```bash
python scripts/copy_data.py
```

### 5. Run

```bash
streamlit run app/streamlit_app.py
```

---

## Testing

```bash
python -m pytest tests/ -q      # 41 tests, no API calls, runs in seconds
```

`tests/test_smoke.py` covers module imports plus the pure logic that has historically broken — router intent merging, scoped-query fallback, persona detection edge cases, evaluator exemptions, credential normalization, and the calendar event body. Every bug found in production or in an eval run has a regression test here.

---

## Project Structure

```
anime-agentic-rag/
├── .github/workflows/
│   └── eval.yml                 # PR-gated evaluation run
├── app/
│   └── streamlit_app.py         # Streamlit frontend (UI + agent integration)
├── config/
│   └── settings.py              # Central config (pydantic-settings)
├── data/
│   ├── manga_chapters (3).jsonl
│   ├── anime_desc (1).jsonl
│   ├── all_characters (4).jsonl
│   ├── episode_mapping (1).jsonl
│   ├── chroma_anime_db.zip      # Lore vector DB
│   ├── chroma_recs_db.zip       # Recs vector DB
│   └── eval/
│       └── golden_dataset.jsonl # 59 versioned evaluation cases
├── notebooks/                   # Portable Python ports of the Colab notebooks
│   ├── ingestion/               # Raw data scraping & patching
│   ├── indexing/                # Vector database build scripts
│   ├── retrieval/               # Hybrid search pipeline
│   ├── integrations/            # External API tools
│   ├── agent/                   # LangGraph agent (standalone)
│   └── README.md                # Notebook index + src/ mapping
├── scripts/
│   ├── copy_data.py             # Copy data from old project
│   ├── rebuild_lore_db.py       # Rebuild chroma_anime_db from manga_chapters JSONL
│   ├── rebuild_recs_db.py       # Rebuild chroma_recs_db from anime_desc JSONL
│   ├── calendar_auth.py         # One-time Google Calendar OAuth
│   ├── calendar_silence.py      # Maintenance: clear owner-side event reminders
│   └── eval/
│       ├── build_golden_dataset.py   # Generate dataset from the real corpus
│       ├── upload_dataset.py         # Idempotent LangSmith upload
│       └── run_eval.py               # Run experiment + enforce thresholds
├── src/
│   ├── agent/
│   │   ├── state.py             # AgentState TypedDict (reducer-backed context)
│   │   ├── nodes.py             # All 7 LangGraph nodes
│   │   ├── graph.py             # Graph wiring + parallel intent fan-out
│   │   └── runner.py            # Public run_agent_with_state() API
│   ├── rag/
│   │   ├── embeddings.py        # Lazy fastembed (ONNX) embedding singleton
│   │   ├── vectorstores.py      # Lazy ChromaDB loaders (auto-unzip)
│   │   ├── bm25_index.py        # BM25 in-memory index
│   │   └── retriever.py         # Hybrid RAG pipeline (FlashRank reranker)
│   ├── tools/
│   │   ├── trace_moe.py         # Screenshot anime identifier
│   │   ├── omdb.py              # Episode ratings chart generator
│   │   ├── jikan.py             # Airing schedule (IST)
│   │   ├── calendar.py          # Google Calendar watch-list
│   │   └── registry.py          # LangChain TOOLS list
│   ├── eval/
│   │   └── evaluators.py        # 7 deterministic + 1 LLM-judge evaluator
│   ├── persona/                 # 719-character persona engine
│   ├── episode/                 # Episode→chapter mapping engine
│   ├── llm/                     # Lazy Gemini LLM singletons
│   ├── prompts/                 # Pydantic router + prompt builders
│   └── db/                      # Persistent chat history (SQLite/PostgreSQL)
├── tests/
│   ├── conftest.py              # sys.path setup for pytest
│   └── test_smoke.py            # Import + logic tests (no API calls)
├── .env.example
├── requirements.txt
├── requirements-lock.txt        # Pinned, resolved against Python 3.13
├── DEPLOYMENT.md                # Streamlit Community Cloud + Supabase guide
├── PRIVACY.md
└── README.md
```

---

## LangSmith Tracing

```bash
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_key_here
LANGSMITH_PROJECT=anime-rag-prod
```

Free tier at [smith.langchain.com](https://smith.langchain.com): 5,000 traces/month, 14-day retention. Evaluation experiments land in their own project (`anime-rag-eval-ci` in CI), keeping production traces clean.

## Google Calendar Setup (Optional)

```bash
python scripts/calendar_auth.py     # one-time OAuth, opens browser

# then in .env:
ENABLE_CALENDAR_TOOL=true
```

> Set the OAuth app's publishing status to **In production** before running this — in "Testing" mode Google expires the refresh token after 7 days. See [DEPLOYMENT.md](DEPLOYMENT.md#step-3--google-calendar-setup-optional).

Events are written to one shared calendar and the user is handed the event link to add to their own calendar. Source events carry no reminder override: Calendar API reminders are per-authenticated-user, so an override would notify only the calendar's owner — once per request, from every user — while the people who actually wanted the reminder get theirs from their own calendar's defaults.

## Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for the full Streamlit Community Cloud + Supabase guide.

---

## Key Design Decisions

- **Lazy loading** — the embedding model loads only on the first LORE/RECOMMEND query, so startup is instant.
- **Torch-free retrieval** — fastembed and flashrank both run on ONNX Runtime instead of torch/sentence-transformers, cutting peak memory from ~2.5–3GB to ~850MB. That is the difference between being OOM-killed on Streamlit Cloud's 1GB free tier and fitting comfortably.
- **Multi-label Pydantic router** — `RouterOutput(items: list[RouterIntentItem])` returns an intent *and its scoped sub-query*. An earlier single-label router collapsed "what happened in chapter 5, and when does it air?" into one intent and answered half the question.
- **Reducer-backed state** — `retrieved_context` is `Annotated[list, operator.add]` so parallel intent branches can write concurrently without collision.
- **DB-level spoiler filter** — ChromaDB metadata filtering means spoiler chapters are never even scored, rather than being retrieved and then discarded.
- **Scoped TOOL intent** — TOOL covers four specific lookups (schedules, ratings, news, screenshot ID) rather than acting as a general "needs live data" catch-all, which used to route questions like "what's the weather?" to a tool that cannot answer them.
- **Global credential normalization** — `config/settings.py` strips whitespace from every credential *and* normalizes `os.environ`, because some SDKs read the environment directly and never see the parsed settings object. A single trailing newline in a secret previously produced a 401 on every OMDB call and an `InvalidHeader` on every Tavily call.
- **Single embedding singleton** — one `FastEmbedEmbeddings` instance shared by both vector stores.

## License

MIT — see [LICENSE](LICENSE).
