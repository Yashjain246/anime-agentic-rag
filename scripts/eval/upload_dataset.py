"""
scripts/eval/upload_dataset.py
────────────────────────────────
Uploads data/eval/golden_dataset.jsonl to a LangSmith Dataset.

This is the step that actually touches your LangSmith account — nothing
before this (build_golden_dataset.py) does. Requires LANGSMITH_TRACING
and LANGSMITH_API_KEY set (Phase 0 already did this).

Idempotent: each case's "id" field is turned into a stable UUID
(uuid5, same string -> same UUID every time), so re-running this after
editing golden_dataset.jsonl updates the existing examples in place
instead of creating duplicates.

Run:
    python -m scripts.eval.upload_dataset
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from langsmith import Client

from config.settings import settings

DATASET_NAME = "anime-rag-golden-v1"
DATASET_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "eval" / "golden_dataset.jsonl"

# Fixed namespace so uuid5(NAMESPACE, case_id) is stable across runs/machines.
_NAMESPACE = uuid.UUID("a1f3c9e0-6b2d-4e8a-9c1f-7d4b2a8e5f10")


def main() -> None:
    if not (settings.LANGSMITH_TRACING and settings.LANGSMITH_API_KEY):
        raise SystemExit(
            "LANGSMITH_TRACING/LANGSMITH_API_KEY not set — nothing to upload to. "
            "Set them in .env (or Streamlit secrets) first."
        )

    cases = []
    with open(DATASET_PATH, encoding="utf-8") as f:
        for line in f:
            cases.append(json.loads(line))

    client = Client()

    if not client.has_dataset(dataset_name=DATASET_NAME):
        client.create_dataset(
            DATASET_NAME,
            description="Phase 1 golden set for the anime-agentic-rag agent — "
                         "LORE cases built from real manga_chapters.jsonl rows, "
                         "compound cases encode regression coverage for the "
                         "multi-intent routing bugs fixed in dev.",
        )
        print(f"Created dataset '{DATASET_NAME}'")
    else:
        print(f"Dataset '{DATASET_NAME}' already exists — upserting examples")

    examples = [
        {
            "id": str(uuid.uuid5(_NAMESPACE, case["id"])),
            "inputs": {"message": case["input"], "kwargs": case.get("kwargs", {})},
            "outputs": {"expected": case["expected"]},
            "metadata": {"id": case["id"], "category": case["category"]},
        }
        for case in cases
    ]

    client.create_examples(dataset_name=DATASET_NAME, examples=examples)
    print(f"Upserted {len(examples)} examples into '{DATASET_NAME}'")


if __name__ == "__main__":
    main()
