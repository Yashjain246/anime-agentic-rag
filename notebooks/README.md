# notebooks/

Portable Python ports of the Colab notebooks used to build the anime-RAG pipeline.
Organised by **domain responsibility** so any engineer knows exactly where to look.

> **Status**: Scaffolded — paste the actual cell code from each notebook into the
> `# TODO` sections to make the modules runnable.

---

## Folder Map

| Folder | Responsibility | Source Notebook(s) |
|---|---|---|
| [`ingestion/`](ingestion/) | Raw data collection — scraping & patching | `manga_chapters_v2.ipynb`, `manga_fix_last2.ipynb` |
| [`indexing/`](indexing/) | Building vector databases from raw data | `anime_rag_v3.ipynb` (build section), `phase2_recs_db.ipynb` |
| [`retrieval/`](retrieval/) | Hybrid search over the vector databases | `anime_rag_v3.ipynb` (search section) |
| [`integrations/`](integrations/) | External API clients (all 5 tools) | `phase3_final.ipynb` |
| [`agent/`](agent/) | LangGraph agent orchestration | `phase4_langgraph_updated (1).ipynb` |

---

## File Index

```
notebooks/
├── ingestion/
│   ├── manga_scraper.py        scrapes wikitext, summarises via Gemini → JSONL
│   └── chapter_patcher.py     fixes the 2 blocked chapters, patches JSONL
│
├── indexing/
│   ├── lore_db_builder.py     embeds 843 chapters  → chroma_anime_db
│   └── recs_db_builder.py     embeds 500 synopses  → chroma_recs_db
│
├── retrieval/
│   └── hybrid_search.py       BGE dense + BM25 + cross-encoder + spoiler firewall
│
├── integrations/
│   └── external_tools.py      trace.moe · OMDB · Jikan · Tavily · Google Calendar
│                              + LangChain @tool wrappers (TOOLS list)
│
└── agent/
    └── langgraph_agent.py     AgentState · graph nodes · routing · session runner
```

---

## Relationship to `src/`

`src/` is the **production runtime** consumed by the Streamlit app.
`notebooks/` is the **original research pipeline** kept for data regeneration
and experimentation. When you improve something here, port it to `src/`.

| `notebooks/` file | Equivalent `src/` module(s) |
|---|---|
| `ingestion/manga_scraper.py` | *(data generation — no direct src equivalent)* |
| `ingestion/chapter_patcher.py` | *(data generation — no direct src equivalent)* |
| `indexing/lore_db_builder.py` | `src/rag/vectorstores.py` |
| `indexing/recs_db_builder.py` | `src/rag/vectorstores.py` |
| `retrieval/hybrid_search.py` | `src/rag/retriever.py`, `src/rag/bm25_index.py`, `src/rag/embeddings.py` |
| `integrations/external_tools.py` | `src/tools/trace_moe.py`, `src/tools/omdb.py`, `src/tools/jikan.py`, `src/tools/calendar.py` |
| `agent/langgraph_agent.py` | `src/agent/graph.py`, `src/agent/nodes.py`, `src/agent/state.py`, `src/agent/runner.py` |
