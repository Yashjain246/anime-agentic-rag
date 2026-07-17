"""
indexing/recs_db_builder.py
============================
Source notebook: phase2_recs_db.ipynb

Loads 500 anime synopses from data/anime_desc (1).jsonl, generates BGE
embeddings, and upserts them into a persistent ChromaDB collection
(chroma_recs_db). This is the "recommendations" vector store used when
a user asks for anime suggestions.

Run standalone:
    python -m notebooks.indexing.recs_db_builder [--rebuild]

Sections
--------
1. Configuration
2. Data loading      (read + validate anime_desc (1).jsonl)
3. Embedding model   (shared BGE model — same as lore_db_builder)
4. ChromaDB setup    (client, collection)
5. Ingestion loop    (embed → upsert in batches)
6. CLI entry-point
"""

from __future__ import annotations
import argparse, json
from pathlib import Path

# TODO: uncomment once dependencies confirmed
# from sentence_transformers import SentenceTransformer
# import chromadb

# =============================================================================
# 1. Configuration
# =============================================================================

ANIME_DESC_JSONL  = Path("data/anime_desc (1).jsonl")
CHROMA_RECS_PATH  = Path("data/chroma_recs_db")
COLLECTION_NAME   = "anime_recs"          # TODO: verify against notebook
BGE_MODEL_NAME    = "BAAI/bge-small-en-v1.5"
BATCH_SIZE        = 100


# =============================================================================
# 2. Data Loading
# =============================================================================

def load_anime_synopses(path: Path = ANIME_DESC_JSONL) -> list[dict]:
    """
    Load and validate anime synopsis records from *path*.

    Returns
    -------
    list[dict]
        Each record: ``{"title": str, "synopsis": str, ...}``.

    TODO: Replace with actual loading + validation code from phase2_recs_db.ipynb.
    """
    raise NotImplementedError


# =============================================================================
# 3. Embedding Model
# =============================================================================

def get_embedding_model():
    """
    Load and return the BGE sentence-transformer model.

    Reuses the same model as lore_db_builder — import from there if
    already loaded to avoid loading twice.

    TODO: Replace with actual model-loading code from phase2_recs_db.ipynb.
    """
    raise NotImplementedError


# =============================================================================
# 4. ChromaDB Setup
# =============================================================================

def get_chroma_collection(db_path: Path = CHROMA_RECS_PATH, rebuild: bool = False):
    """
    Create (or open) the persistent ChromaDB recs collection at *db_path*.

    TODO: Replace with actual ChromaDB setup code from phase2_recs_db.ipynb.
    """
    raise NotImplementedError


# =============================================================================
# 5. Ingestion Loop
# =============================================================================

def build_recs_db(
    records: list[dict],
    db_path: Path = CHROMA_RECS_PATH,
    rebuild: bool = False,
) -> None:
    """
    Embed *records* and upsert them into the ChromaDB recs collection.

    TODO: Replace with the actual ingestion loop from phase2_recs_db.ipynb.
    """
    raise NotImplementedError


# =============================================================================
# 6. CLI Entry-Point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build the recs ChromaDB from anime_desc (1).jsonl."
    )
    parser.add_argument("--rebuild", action="store_true",
                        help="Wipe and recreate the collection")
    parser.add_argument("--db-path", default=str(CHROMA_RECS_PATH))
    args = parser.parse_args()

    print(f"Loading synopses from {ANIME_DESC_JSONL} …")
    records = load_anime_synopses()
    print(f"Loaded {len(records)} records. Building DB …")
    build_recs_db(records, db_path=Path(args.db_path), rebuild=args.rebuild)
    print("Done.")
