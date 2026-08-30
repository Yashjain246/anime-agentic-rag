"""
indexing/recs_db_builder.py
============================
Source notebook: Anime Rag phase 2 Anime desc.ipynb

STATUS: intentionally not reimplemented here — same reason as
indexing/lore_db_builder.py. An earlier version used BAAI/bge-large-en-v1.5
on GPU; the current app deliberately runs bge-small-en-v1.5 via fastembed
(CPU, torch-free) to fit Streamlit Cloud's memory limit.

THE ACTUAL WORKING VERSION:

    python scripts/rebuild_recs_db.py

Reads data/anime_desc (1).jsonl, composes the same "Title / Genres /
Score / Synopsis" document format the recovered notebook used, embeds
via src/rag/embeddings.get_recs_embeddings() (fastembed, bge-small,
shared with the lore DB's embedding model), and re-zips to
data/chroma_recs_db.zip.

Do not add a second implementation here — see lore_db_builder.py's
docstring for why.
"""
