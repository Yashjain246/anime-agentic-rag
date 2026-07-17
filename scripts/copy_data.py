"""
scripts/copy_data.py
────────────────────
Copies all data files from the original Colab project folder into
the new anime-rag/data/ directory.

Run once after cloning the project:
  python scripts/copy_data.py
"""

import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

SOURCE_DIR = Path(r"C:\Users\jainy\.gemini\antigravity-ide\scratch\Agentic Ai Anime bot")

FILES_TO_COPY = [
    "all_characters (4).jsonl",
    "manga_chapters (3).jsonl",
    "episode_mapping (1).jsonl",
    "anime_desc (1).jsonl",
    "chroma_anime_db.zip",
    "chroma_recs_db.zip",
]

print(f"Copying data files to: {DATA_DIR}\n")
for filename in FILES_TO_COPY:
    src = SOURCE_DIR / filename
    dst = DATA_DIR / filename
    if dst.exists():
        print(f"  ✅ Already exists: {filename}")
        continue
    if src.exists():
        shutil.copy2(src, dst)
        size_mb = dst.stat().st_size / 1024 / 1024
        print(f"  📂 Copied: {filename} ({size_mb:.1f} MB)")
    else:
        print(f"  ⚠️  Not found: {src}")

print("\n✅ Data copy complete!")
print(f"Data directory: {DATA_DIR}")
