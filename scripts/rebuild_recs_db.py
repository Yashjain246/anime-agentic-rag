"""
scripts/rebuild_recs_db.py
───────────────────────────
Rebuilds data/chroma_recs_db from data/anime_desc (1).jsonl using the
current RECS_EMBEDDING_MODEL (now BAAI/bge-small-en-v1.5, shared with the
lore DB — see config/settings.py). The DB was previously built with
bge-large-en-v1.5 (1024-dim); switching to bge-small (384-dim) requires a
full re-embed since ChromaDB collections are fixed to one vector dimension.

Also re-zips the result to data/chroma_recs_db.zip using the same flat
layout (no wrapping folder) that src/rag/vectorstores.py expects.

Run once after changing RECS_EMBEDDING_MODEL:
    python scripts/rebuild_recs_db.py
"""

from __future__ import annotations

import json
import shutil
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from langchain_chroma import Chroma
from langchain_core.documents import Document

from config.settings import settings
from src.rag.embeddings import get_recs_embeddings


def load_anime_records(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_document(record: dict) -> Document:
    title = record.get("title", "Unknown")
    genres = record.get("genres") or []
    genres_str = ", ".join(genres) if isinstance(genres, list) else str(genres)
    score = record.get("score", "N/A")
    synopsis = record.get("synopsis", "")

    page_content = (
        f"Title: {title}\n"
        f"Genres: {genres_str}\n"
        f"Score: {score}\n"
        f"Synopsis: {synopsis}"
    )
    metadata = {"title": title, "genres": genres_str, "score": score}
    return Document(page_content=page_content, metadata=metadata)


def rezip_flat(source_dir: Path, zip_path: Path) -> None:
    """Zip the *contents* of source_dir at the archive root (no wrapping folder)."""
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in source_dir.rglob("*"):
            if file_path.is_file():
                zf.write(file_path, file_path.relative_to(source_dir))


def main() -> None:
    anime_desc_path = settings.DATA_DIR / "anime_desc (1).jsonl"
    recs_dir = settings.CHROMA_RECS_DIR
    recs_zip = settings.CHROMA_RECS_ZIP

    print(f"Loading anime synopses from {anime_desc_path} ...")
    records = load_anime_records(anime_desc_path)
    print(f"Loaded {len(records)} records.")

    documents = [build_document(r) for r in records]

    if recs_dir.exists():
        print(f"Removing old {recs_dir} ...")
        shutil.rmtree(recs_dir)
    recs_dir.mkdir(parents=True, exist_ok=True)

    print(f"Embedding + building Chroma collection at {recs_dir} "
          f"using {settings.RECS_EMBEDDING_MODEL} ...")
    Chroma.from_documents(
        documents=documents,
        embedding=get_recs_embeddings(),
        persist_directory=str(recs_dir),
    )
    print("[OK] Recs DB rebuilt.")

    print(f"Re-zipping -> {recs_zip} ...")
    rezip_flat(recs_dir, recs_zip)
    print(f"[OK] {recs_zip} ({recs_zip.stat().st_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
