"""
src/rag/vectorstores.py
───────────────────────
Lazy loaders for ChromaDB vector stores.

Both stores are loaded on first access — not at import time.
Also handles unzipping the database archives if the extracted
directories don't exist yet (replaces the Colab !unzip shell commands).
"""

from __future__ import annotations

import zipfile
from pathlib import Path

from langchain_chroma import Chroma

from config.settings import settings
from src.rag.embeddings import get_lore_embeddings, get_recs_embeddings

_lore_vectorstore: Chroma | None = None
_recs_vectorstore: Chroma | None = None


def _ensure_unzipped(zip_path: Path, extract_to: Path) -> None:
    """
    Unzip a ChromaDB archive if the target directory doesn't exist yet.
    Safe to call multiple times — no-op if already extracted.

    Both chroma_*.zip archives store their contents flat (chroma.sqlite3
    and the collection UUID folder at the ZIP root, no wrapping directory),
    so we extract directly into extract_to.
    """
    if extract_to.exists() and any(extract_to.iterdir()):
        return  # already unzipped
    if not zip_path.exists():
        raise FileNotFoundError(
            f"ChromaDB archive not found: {zip_path}\n"
            f"Place it at: {zip_path}"
        )
    extract_to.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {zip_path.name} -> {extract_to} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)
    print(f"[Extracted] {zip_path.name}")


def get_lore_vectorstore() -> Chroma:
    """
    Returns the Lore DB ChromaDB instance (manga chapter summaries).
    Unzips chroma_anime_db.zip if needed. Loads embeddings on first call.
    """
    global _lore_vectorstore
    if _lore_vectorstore is None:
        _ensure_unzipped(settings.CHROMA_LORE_ZIP, settings.CHROMA_LORE_DIR)
        _lore_vectorstore = Chroma(
            persist_directory=str(settings.CHROMA_LORE_DIR),
            embedding_function=get_lore_embeddings(),
        )
        print(f"[OK] Lore DB loaded: {_lore_vectorstore._collection.count()} chapters")
    return _lore_vectorstore


def get_recs_vectorstore() -> Chroma:
    """
    Returns the Recs DB ChromaDB instance (anime synopses for recommendations).
    Unzips chroma_recs_db.zip if needed. Shares the same embedding model.
    """
    global _recs_vectorstore
    if _recs_vectorstore is None:
        _ensure_unzipped(settings.CHROMA_RECS_ZIP, settings.CHROMA_RECS_DIR)
        _recs_vectorstore = Chroma(
            persist_directory=str(settings.CHROMA_RECS_DIR),
            embedding_function=get_recs_embeddings(),
        )
        print(f"[OK] Recs DB loaded: {_recs_vectorstore._collection.count()} anime")
    return _recs_vectorstore
