"""
scripts/rebuild_lore_db.py
────────────────────────────
Rebuilds data/chroma_anime_db from data/manga_chapters (3).jsonl.

The previously committed chroma_anime_db.zip was found to contain the
WRONG data — all 526 records had the recs/anime-synopsis schema
({genres, score, title}) instead of manga chapter data ({anime_name,
chapter_number, summary_text}). This rebuilds it correctly from the
real chapter JSONL, using the same page_content format as
src/rag/bm25_index.py so both halves of the hybrid retriever
(dense + BM25) are built from consistent text.

Also re-zips the result to data/chroma_anime_db.zip using the same flat
layout (no wrapping folder) that src/rag/vectorstores.py expects.

Run:
    python scripts/rebuild_lore_db.py
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
from src.rag.embeddings import get_lore_embeddings


def load_chapters(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            if data.get("error"):
                continue
            records.append(data)
    return records


def build_document(data: dict) -> Document:
    events_text = "\n".join(f"- {e}" for e in data.get("key_events", []))
    page_content = (
        f"Anime: {data['anime_name']} | Chapter {data['chapter_number']}\n"
        f"Summary: {data.get('summary_text', '')}\n\n"
        f"Key Events:\n{events_text}"
    )
    metadata = {
        "anime_name": data["anime_name"],
        "chapter_number": data["chapter_number"],
    }
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
    manga_path = settings.MANGA_CHAPTERS_PATH
    lore_dir = settings.CHROMA_LORE_DIR
    lore_zip = settings.CHROMA_LORE_ZIP

    print(f"Loading manga chapters from {manga_path} ...")
    records = load_chapters(manga_path)
    print(f"Loaded {len(records)} chapters.")

    counts: dict[str, int] = {}
    for r in records:
        counts[r["anime_name"]] = counts.get(r["anime_name"], 0) + 1
    for anime, n in sorted(counts.items()):
        print(f"  {anime}: {n} chapters")

    documents = [build_document(r) for r in records]

    if lore_dir.exists():
        print(f"Removing old {lore_dir} ...")
        shutil.rmtree(lore_dir)
    lore_dir.mkdir(parents=True, exist_ok=True)

    print(f"Embedding + building Chroma collection at {lore_dir} "
          f"using {settings.LORE_EMBEDDING_MODEL} ...")
    Chroma.from_documents(
        documents=documents,
        embedding=get_lore_embeddings(),
        persist_directory=str(lore_dir),
    )
    print("[OK] Lore DB rebuilt.")

    print(f"Re-zipping -> {lore_zip} ...")
    rezip_flat(lore_dir, lore_zip)
    print(f"[OK] {lore_zip} ({lore_zip.stat().st_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
