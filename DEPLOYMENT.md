# Deployment Guide — Anime Agentic RAG

This guide covers deploying the app to **Render** (web service) with **Supabase** (PostgreSQL for chat history).

---

## Prerequisites

| What | Where |
|---|---|
| Render account | [render.com](https://render.com) (free tier works) |
| Supabase project | [supabase.com](https://supabase.com) (free tier works) |
| All API keys | GOOGLE_API_KEY, TAVILY_API_KEY, OMDB_API_KEY |
| ChromaDB ZIPs | `chroma_anime_db.zip`, `chroma_recs_db.zip` (from Colab Phase 2) |

---

## Step 1 — Supabase (PostgreSQL)

1. Create a new Supabase project
2. Go to **Settings → Database → Connection string → URI**
3. Copy the connection string — it looks like:
   ```
   postgresql://postgres:[password]@db.[project-ref].supabase.co:5432/postgres
   ```
4. You'll use this as `DATABASE_URL` in Render

---

## Step 2 — Prepare the Repository

Ensure your repo contains:
- `requirements.txt` (all dependencies)
- `data/chroma_anime_db.zip` and `data/chroma_recs_db.zip`
- All `data/*.jsonl` files
- `app/streamlit_app.py`

> **Note**: The ChromaDB ZIPs are auto-extracted on first run by `vectorstores.py` — no manual step needed.

---

## Step 3 — Google Calendar Setup (Optional)

If you want the Calendar tool in production:

```bash
# Run once locally — opens browser for OAuth consent
python scripts/calendar_auth.py
```

This creates `token.json`. Upload it to Render as a **Secret File** at path `/etc/secrets/token.json`, then set:
```
CALENDAR_TOKEN_PATH=/etc/secrets/token.json
ENABLE_CALENDAR_TOOL=true
```

---

## Step 4 — Deploy to Render

### 4a. Create a Web Service

1. Go to Render → **New → Web Service**
2. Connect your GitHub repo
3. Configure:

| Setting | Value |
|---|---|
| **Runtime** | Python 3 |
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `streamlit run app/streamlit_app.py --server.port $PORT --server.address 0.0.0.0` |
| **Instance Type** | Starter ($7/mo) minimum — Standard recommended for GPU |

### 4b. Environment Variables

Add these in Render → **Environment**:

```
GOOGLE_API_KEY=your_gemini_api_key
TAVILY_API_KEY=your_tavily_key
OMDB_API_KEY=your_omdb_key
DATABASE_URL=postgresql://postgres:...@supabase...
LLM_MODEL=gemini-3.1-flash-lite
EMBEDDING_DEVICE=cpu          # Render CPU instances only
LANGSMITH_TRACING=false
ENABLE_CALENDAR_TOOL=false    # set true if you uploaded token.json
```

> **GPU note**: `EMBEDDING_DEVICE` auto-detects on startup (GPU → CPU fallback). On Render CPU instances, explicitly set `EMBEDDING_DEVICE=cpu` to skip the detection delay.

---

## Step 5 — First Deploy

On the first deploy, the app will:
1. Install dependencies (~3-4 min for PyTorch + sentence-transformers)
2. Extract ChromaDB ZIPs on first request (~10 sec)
3. Load the BGE embedding model on first LORE/RECOMMEND query (~30-60 sec on CPU)

Subsequent requests are fast — everything is cached in memory for the session.

---

## Architecture on Render

```
User Browser
     │
     ▼
Render Web Service (Streamlit)
     │
     ├── ChromaDB (local files, from ZIP)
     ├── BM25 Index (in-memory, rebuilt ~5s)
     ├── BGE Embeddings (loaded on first LORE query)
     ├── Gemini API (gemini-3.1-flash-lite)
     ├── Tavily API
     ├── OMDB API
     ├── Jikan API (no key needed)
     └── Supabase PostgreSQL (chat history)
```

---

## Monitoring

- **LangSmith**: Set `LANGSMITH_TRACING=true` + `LANGSMITH_API_KEY` to trace every agent run at [smith.langchain.com](https://smith.langchain.com)
- **Render Logs**: Real-time logs in the Render dashboard show BM25 index builds, ChromaDB loads, and tool calls

---

## Cost Estimate (Monthly)

| Service | Plan | Cost |
|---|---|---|
| Render Web Service | Starter | $7 |
| Supabase | Free tier | $0 |
| Gemini API | Pay-per-use | ~$0-5 |
| Tavily API | Free (1k req/mo) | $0 |
| OMDB API | Free (1k req/day) | $0 |
| **Total** | | **~$7-12/mo** |
