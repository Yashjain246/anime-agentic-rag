"""
indexing/lore_db_builder.py
============================
Source notebook: anime_rag_v3.ipynb  (DB-build section)

Takes the 843 manga chapter summaries from data/manga_chapters (3).jsonl,
generates BGE embeddings, and upserts them into a persistent ChromaDB
collection (chroma_anime_db). This is the "lore" vector store used by
the RAG retrieval pipeline for plot / lore questions.

Run standalone:
    python -m notebooks.indexing.lore_db_builder [--rebuild]

Sections
--------
1. Configuration
2. Data loading      (read + validate manga_chapters (3).jsonl)
3. Embedding model   (BAAI/bge-small-en-v1.5 via sentence-transformers)
4. ChromaDB setup    (client, collection, embedding function)
5. Ingestion loop    (chunk → embed → upsert in batches)
6. CLI entry-point
"""

from __future__ import annotations
import argparse, json
from pathlib import Path

# TODO: uncomment once dependencies confirmed
# from sentence_transformers import SentenceTransformer
# import chromadb
# from chromadb.utils import embedding_functions

# =============================================================================
# 1. Configuration
# =============================================================================

MANGA_JSONL       = Path("data/manga_chapters (3).jsonl")
CHROMA_LORE_PATH  = Path("data/chroma_anime_db")
COLLECTION_NAME   = "manga_lore"          # TODO: verify against notebook
BGE_MODEL_NAME    = "BAAI/bge-small-en-v1.5"
BATCH_SIZE        = 100


# =============================================================================
# 2. Data Loading
# =============================================================================

def load_chapters(path: Path = MANGA_JSONL) -> list[dict]:
    """
    Load and validate chapter records from *path*.

    Returns
    -------
    list[dict]
        Each record: ``{"chapter": int, "title": str, "summary": str}``.

    TODO: Replace with actual loading code from anime_rag_v3.ipynb.
    """
    raise NotImplementedError


# =============================================================================
# 3. Embedding Model
# =============================================================================

def get_embedding_model():
    """
    Load and return the BGE sentence-transformer model.

    TODO: Replace with actual model-loading code from anime_rag_v3.ipynb.
    """
    raise NotImplementedError


# =============================================================================
# 4. ChromaDB Setup
# =============================================================================

def get_chroma_collection(db_path: Path = CHROMA_LORE_PATH, rebuild: bool = False):
    """
    Create (or open) the persistent ChromaDB collection at *db_path*.

    Parameters
    ----------
    rebuild : bool
        If True, delete and recreate the collection from scratch.

    TODO: Replace with actual ChromaDB setup code from anime_rag_v3.ipynb.
    """
    raise NotImplementedError


# =============================================================================
# 5. Ingestion Loop
# =============================================================================

def build_lore_db(
    chapters: list[dict],
    db_path: Path = CHROMA_LORE_PATH,
    rebuild: bool = False,
) -> None:
    """
    Embed *chapters* and upsert them into the ChromaDB lore collection.

    Parameters
    ----------
    chapters : list[dict]
        Chapter records from :func:`load_chapters`.
    db_path : Path
        Directory for the persistent ChromaDB.
    rebuild : bool
        Wipe and recreate the collection if True.

    TODO: Replace with the actual ingestion loop from anime_rag_v3.ipynb.
    """
    raise NotImplementedError


# =============================================================================
# 6. CLI Entry-Point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build the lore ChromaDB from manga_chapters (3).jsonl."
    )
    parser.add_argument("--rebuild", action="store_true",
                        help="Wipe and recreate the collection")
    parser.add_argument("--db-path", default=str(CHROMA_LORE_PATH))
    args = parser.parse_args()

    print(f"Loading chapters from {MANGA_JSONL} …")
    chapters = load_chapters()
    print(f"Loaded {len(chapters)} chapters. Building DB …")
    build_lore_db(chapters, db_path=Path(args.db_path), rebuild=args.rebuild)
    print("Done.")
