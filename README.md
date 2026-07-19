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

# 🎌 Anime Agentic RAG
A production-ready **agentic RAG (Retrieval-Augmented Generation)** system for anime & manga, built with LangGraph, Gemini, and ChromaDB.

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776ab?logo=python)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-green)](https://github.com/langchain-ai/langgraph)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

## Features

| Feature | Details |
|---------|---------|
| 🗺️ **Agentic routing** | LangGraph state machine — LORE / RECOMMEND / TOOL / GENERAL |
| 📖 **Spoiler firewall** | ChromaDB metadata filter at DB level (not post-retrieval) |
| 🔍 **Hybrid RAG** | MultiQuery → 60% dense + 40% BM25 → FlashRank reranker |
| 🎭 **738 character personas** | Dynamic persona engine from extracted personality data |
| 📺 **Episode→chapter mapping** | "I'm on episode 45 of Demon Slayer" → chapter cap auto-set |
| 🔧 **5 tools** | trace.moe, OMDB ratings, Jikan schedule, Tavily news, Google Calendar |
| 💾 **Persistent chat history** | SQLite locally, PostgreSQL (Supabase) in production |
| 📊 **LangSmith tracing** | Full observability — set `LANGSMITH_TRACING=true` |
| ⚡ **Lightweight footprint** | fastembed + flashrank (ONNX, no torch/GPU) — ~850MB peak RSS |

## Anime Coverage

| Anime | Lore DB | Episode Mapping |
|-------|---------|-----------------|
| Demon Slayer | ✅ | ✅ |
| Jujutsu Kaisen | ✅ | ✅ |
| Attack on Titan | ✅ | ✅ |
| Chainsaw Man | ✅ | ✅ |
| Frieren | ❌ | ✅ |
| Solo Leveling | ❌ | ✅ |

## Quick Start

### 1. Prerequisites

- Python 3.11 installed (no GPU required — embeddings and reranking run on CPU via ONNX)

### 2. Clone & Setup

```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure

```bash
# Copy and fill in your API keys
copy .env.example .env
# Edit .env with your keys
```

**Required API keys:**
- `GOOGLE_API_KEY` → [aistudio.google.com](https://aistudio.google.com)
- `TAVILY_API_KEY` → [tavily.com](https://tavily.com) (free tier)
- `OMDB_API_KEY` → [omdbapi.com](http://www.omdbapi.com/apikey.aspx) (free)

### 4. Copy Data Files

```bash
python scripts/copy_data.py
```

### 5. Run

```bash
streamlit run app/streamlit_app.py
```

## Project Structure

```
anime-rag/
├── app/
│   └── streamlit_app.py         # Streamlit frontend (UI + agent integration)
├── config/
│   └── settings.py              # Central config (pydantic-settings)
├── data/                        # JSONL files + ChromaDB ZIPs (gitignored)
│   ├── manga_chapters (3).jsonl
│   ├── anime_desc (1).jsonl
│   ├── all_characters (4).jsonl
│   ├── episode_mapping (1).jsonl
│   ├── chroma_anime_db.zip      # Lore vector DB
│   └── chroma_recs_db.zip       # Recs vector DB
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
│   └── calendar_auth.py         # One-time Google Calendar OAuth
├── src/
│   ├── agent/
│   │   ├── state.py             # AgentState TypedDict
│   │   ├── nodes.py             # All 7 LangGraph nodes
│   │   ├── graph.py             # Graph wiring + compilation
│   │   └── runner.py            # Public run_agent_with_state() API
│   ├── rag/
│   │   ├── embeddings.py        # Lazy fastembed (ONNX) embedding singleton
│   │   ├── vectorstores.py      # Lazy ChromaDB loaders (auto-unzip)
│   │   ├── bm25_index.py        # BM25 in-memory index
│   │   └── retriever.py         # 5-layer RAG pipeline (FlashRank reranker)
│   ├── tools/
│   │   ├── trace_moe.py         # Screenshot anime identifier
│   │   ├── omdb.py              # Episode ratings chart generator
│   │   ├── jikan.py             # MAL airing schedule (IST)
│   │   ├── calendar.py          # Google Calendar watch-list
│   │   └── registry.py          # LangChain TOOLS list
│   ├── persona/                 # 697-character persona engine
│   ├── episode/                 # Episode→chapter mapping engine
│   ├── llm/                     # Lazy Gemini LLM singletons
│   ├── prompts/                 # Pydantic router + prompt builders
│   └── db/                      # Persistent chat history (SQLite/PostgreSQL)
├── tests/
│   ├── conftest.py              # sys.path setup for pytest
│   └── test_smoke.py            # Import + logic smoke tests (no API calls)
├── .env.example                 # API key template
├── requirements.txt
├── DEPLOYMENT.md                # Streamlit Community Cloud + Supabase deployment guide
└── README.md
```

## LangSmith Tracing

Add to `.env`:
```
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_key_here
LANGSMITH_PROJECT=anime-rag-prod
```

Sign up free at [smith.langchain.com](https://smith.langchain.com) — 5,000 traces/month.

## Google Calendar Setup (Optional)

```bash
# One-time OAuth (opens browser)
python scripts/calendar_auth.py

# Then enable in .env:
ENABLE_CALENDAR_TOOL=true
```

## Deployment (Streamlit Community Cloud + Supabase)

See [DEPLOYMENT.md](DEPLOYMENT.md) for full deployment guide.

## Architecture

```
User Message
     │
     ▼
[persona_node] ──── PERSONA_SWITCH? ──► END
     │ NO
     ▼
[episode_node] ──── EPISODE_UPDATE? ──► END
     │ NO
     ▼
[router_node] ─── Pydantic RouterOutput ───┐
     │                                      │
     ├── LORE ──► [lore_node] ──────────►  │
     ├── RECOMMEND ──► [recs_node] ──────► [respond_node] ──► END
     ├── TOOL ──► [tools_node] ──────────►  │
     └── GENERAL ──────────────────────────►│
```

## Key Design Decisions

- **Lazy loading**: The embedding model loads only on the first LORE/RECOMMEND query — startup is instant
- **Torch-free retrieval**: Embeddings (fastembed) and reranking (flashrank) both run on ONNX Runtime instead of torch/sentence-transformers, cutting peak memory from ~2.5-3GB to ~850MB — the difference between getting killed on Streamlit Cloud's 1GB free tier and fitting comfortably
- **Pydantic router**: `RouterOutput(intent: Literal['LORE','RECOMMEND','TOOL','GENERAL'])` — no silent misclassification
- **Configurable tool loop**: `MAX_TOOL_ITERATIONS=5` prevents infinite loops while supporting multi-step chains
- **DB-level spoiler filter**: ChromaDB metadata filter means spoiler chapters are never even scored
- **Single embedding singleton**: One `FastEmbedEmbeddings` instance shared by both vector stores
