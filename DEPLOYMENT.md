# Deployment Guide — Anime Agentic RAG

This guide covers deploying the app to **Streamlit Community Cloud** (free, recommended) with **Supabase** (PostgreSQL for chat history). Render is documented as a paid fallback.

---

## Why Streamlit Community Cloud

The app used to get killed on Streamlit Cloud's 1GB free-tier RAM cap after a few chats. That was caused by the app's own footprint, not the platform:

- `RECS_EMBEDDING_MODEL` defaulted to `bge-large-en-v1.5` (~1.3GB), loaded the moment anyone asked for a recommendation, on top of a separate lore embedder.
- Embeddings and reranking ran on `sentence-transformers`/`torch`, which cost roughly 1GB of RAM just from being imported — before any model weights even loaded.

Both are fixed now:
- Lore and recs share one `bge-small-en-v1.5` embedding model (`config/settings.py`), loaded once.
- Embeddings run via **fastembed** (ONNX runtime) and reranking via **flashrank** — both are torch-free, dropping the same capability to a few hundred MB. See `src/rag/embeddings.py` and `src/rag/retriever.py`.
- Measured peak RSS for a full LORE + RECOMMEND session is now ~850MB (down from ~2.5-3GB+), comfortably under the 1GB cap.

(Hugging Face Spaces was considered, but as of ~2026-07 HF blocks new free accounts from CPU Basic hardware — an active, undocumented policy change, not something to build a deployment on right now.)

We also considered Google Cloud Run, but it requires a billing account (card on file) to enable Cloud Run at all, even within its always-free quota — ruled out given the "no card" requirement. Streamlit Community Cloud needs no card and no account changes: it's the platform already in use.

---

## Prerequisites

| What | Where |
|---|---|
| Streamlit Community Cloud account | [share.streamlit.io](https://share.streamlit.io) (free, GitHub login) |
| Supabase project | [supabase.com](https://supabase.com) (free tier works) |
| All API keys | GOOGLE_API_KEY, TAVILY_API_KEY, OMDB_API_KEY |
| ChromaDB ZIPs | `data/chroma_anime_db.zip`, `data/chroma_recs_db.zip` (already in the repo) |

---

## Step 1 — Supabase (PostgreSQL)

Local `chat_history.db` (SQLite) does not persist across Streamlit Cloud restarts/redeploys — use Supabase Postgres for anything beyond local dev.

1. Create a new Supabase project
2. Go to **Settings → Database → Connection string → URI**
3. Copy the connection string — it looks like:
   ```
   postgresql://postgres:[password]@db.[project-ref].supabase.co:5432/postgres
   ```
4. You'll use this as `DATABASE_URL` in Streamlit Cloud's secrets

---

## Step 2 — Prepare the Repository

Ensure your repo contains:
- `requirements.txt` (all dependencies — no torch, no CUDA index)
- `data/chroma_anime_db.zip` and `data/chroma_recs_db.zip`
- All `data/*.jsonl` files
- `app/streamlit_app.py`

> **Note**: The ChromaDB ZIPs are auto-extracted on first run by `src/rag/vectorstores.py` — no manual step needed. If you ever need to rebuild them from source data (e.g. after changing the embedding model), use `python scripts/rebuild_lore_db.py` and `python scripts/rebuild_recs_db.py`.

---

## Step 3 — Google Calendar Setup (Optional)

If you want the Calendar tool in production:

```bash
# Run once locally — opens browser for OAuth consent
python scripts/calendar_auth.py
```

This creates `token.json`. Base64-encode it and set it as a secret:
```bash
python -c "import base64; print(base64.b64encode(open('token.json','rb').read()).decode())"
```
Set the output as `CALENDAR_TOKEN_B64` in your secrets, plus `ENABLE_CALENDAR_TOOL=true`.

---

## Step 4 — Deploy to Streamlit Community Cloud

1. Push your repo to GitHub (public or private — Streamlit Cloud supports both with a GitHub login)
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select your repo, branch, and set **Main file path** to `app/streamlit_app.py`
4. Open **Advanced settings → Secrets** and paste (TOML format):

```toml
GOOGLE_API_KEY = "your_gemini_api_key"
TAVILY_API_KEY = "your_tavily_key"
OMDB_API_KEY = "your_omdb_key"
DATABASE_URL = "postgresql://postgres:...@supabase..."
LLM_MODEL = "gemini-3.1-flash-lite"
LANGSMITH_TRACING = "false"
ENABLE_CALENDAR_TOOL = "false"
```

5. Deploy. First boot installs dependencies (~1-2 min, no PyTorch to compile/download) and extracts the ChromaDB zips on first LORE/RECOMMEND request (~5-10 sec).

---

## Step 5 — Render (paid fallback)

If you ever want to avoid Streamlit Cloud's sleep-on-inactivity behavior, Render Starter ($7/mo) works with the same codebase — no footprint constraints to worry about at that tier.

| Setting | Value |
|---|---|
| **Runtime** | Python 3 |
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `streamlit run app/streamlit_app.py --server.port $PORT --server.address 0.0.0.0` |
| **Instance Type** | Starter ($7/mo) |

Environment variables are the same as the Streamlit Cloud secrets above (plain `KEY=value` env vars instead of TOML).

---

## Architecture

```
User Browser
     │
     ▼
Streamlit Community Cloud (Streamlit app)
     │
     ├── ChromaDB (local files, from ZIP, ~850MB peak RSS incl. app)
     ├── BM25 Index (in-memory, rebuilt ~5s)
     ├── fastembed (ONNX) — bge-small-en-v1.5, shared by lore + recs
     ├── flashrank (ONNX) — ms-marco-MiniLM-L-12-v2 reranker
     ├── Gemini API (gemini-3.1-flash-lite)
     ├── Tavily API
     ├── OMDB API
     ├── Jikan API (no key needed)
     └── Supabase PostgreSQL (chat history)
```

---

## Monitoring

- **LangSmith**: Set `LANGSMITH_TRACING=true` + `LANGSMITH_API_KEY` to trace every agent run at [smith.langchain.com](https://smith.langchain.com)
- **Streamlit Cloud logs**: Real-time logs in the app's "Manage app" panel show BM25 index builds, ChromaDB loads, and tool calls

---

## Cost Estimate (Monthly)

| Service | Plan | Cost |
|---|---|---|
| Streamlit Community Cloud | Free | $0 |
| Supabase | Free tier | $0 |
| Gemini API | Pay-per-use | ~$0-5 |
| Tavily API | Free (1k req/mo) | $0 |
| OMDB API | Free (1k req/day) | $0 |
| **Total** | | **~$0-5/mo** |
