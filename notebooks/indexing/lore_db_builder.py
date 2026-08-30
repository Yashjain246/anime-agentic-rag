"""
indexing/lore_db_builder.py
============================
Source notebook: anime RAG manga lore Db creation (final).ipynb

STATUS: intentionally not reimplemented here. An earlier version of this
pipeline built the DB with BAAI/bge-large-en-v1.5 on GPU via
sentence-transformers — exactly the setup DEPLOYMENT.md documents as the
cause of the Streamlit Cloud OOM kills that got fixed by switching to
fastembed + bge-small-en-v1.5, torch-free, CPU-only
(src/rag/embeddings.py). Reintroducing the GPU/bge-large version here
would undo that fix.

THE ACTUAL WORKING VERSION — already adapted to the current lightweight
stack, and what production/deployment actually uses:

    python scripts/rebuild_lore_db.py

It reads the same data/manga_chapters (3).jsonl, composes documents with
the same "Anime: X | Chapter Y\\nSummary: ...\\nKey Events:\\n..." format
the recovered notebook used, embeds via src/rag/embeddings.get_lore_embeddings()
(fastembed, bge-small-en-v1.5, CPU), and re-zips to data/chroma_anime_db.zip
in the flat layout src/rag/vectorstores.py expects.

Run that after adding new chapters via notebooks/ingestion/manga_scraper.py
to bring them into the live vector store.

Do not add a second implementation here — if the rebuild logic needs to
change, change scripts/rebuild_lore_db.py; a duplicate copy here would
just be a second place for the two to drift apart.
"""
