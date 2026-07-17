"""
src/rag/bm25_index.py
─────────────────────
BM25 in-memory index builder.

WHY BM25 alongside ChromaDB:
  Character names like 'Rengoku', 'Akaza', 'Shibuya' are proper nouns.
  Dense vector search (ChromaDB) finds semantic meaning.
  BM25 finds exact keyword matches.
  Together they cover both semantic AND exact-name queries.

The index rebuilds from the JSONL file in ~5s each session.
"""

from __future__ import annotations

import json

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from config.settings import settings

_bm25_retriever: BM25Retriever | None = None
_bm25_docs: list[Document] | None = None


def get_bm25_retriever() -> BM25Retriever:
    """
    Returns the BM25 retriever built from manga_chapters (3).jsonl.
    Rebuilt in-memory on first call (~5 seconds).
    """
    global _bm25_retriever, _bm25_docs

    if _bm25_retriever is not None:
        return _bm25_retriever

    print("Building BM25 index from JSONL...")
    all_docs: list[Document] = []

    with open(settings.MANGA_CHAPTERS_PATH, encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            if data.get("error"):
                continue
            events_text = "\n".join(f"- {e}" for e in data.get("key_events", []))
            page_content = (
                f"Anime: {data['anime_name']} | Chapter {data['chapter_number']}\n"
                f"Summary: {data.get('summary_text', '')}\n\n"
                f"Key Events:\n{events_text}"
            )
            all_docs.append(
                Document(
                    page_content=page_content,
                    metadata={
                        "anime_name": data["anime_name"],
                        "chapter_number": data["chapter_number"],
                    },
                )
            )

    _bm25_docs = all_docs
    _bm25_retriever = BM25Retriever.from_documents(all_docs)
    _bm25_retriever.k = settings.BM25_TOP_K
    print(f"[OK] BM25 index built: {len(all_docs)} documents")
    return _bm25_retriever


def get_bm25_docs() -> list[Document]:
    """Returns the raw document list used to build the BM25 index."""
    get_bm25_retriever()  # ensures _bm25_docs is populated
    return _bm25_docs or []
