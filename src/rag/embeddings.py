"""
src/rag/embeddings.py
─────────────────────
Lazy singleton for HuggingFace BGE embeddings.

WHY lazy loading fixes the 3-4 minute startup delay:
  The original notebook loaded HuggingFaceEmbeddings at Cell 3 — before
  anything else ran. That forced a 3-4 minute wait on every session start,
  even for GENERAL or TOOL queries that never touch the vector stores.

  Here the model is loaded only on the FIRST call to get_embeddings().
  For GENERAL / TOOL / PERSONA_SWITCH / EPISODE_UPDATE paths, the embedding
  model is never loaded at all. For LORE / RECOMMEND paths it loads once,
  then all subsequent queries reuse the cached instance.
"""

from __future__ import annotations

from langchain_huggingface import HuggingFaceEmbeddings

from config.settings import settings

_lore_embeddings: HuggingFaceEmbeddings | None = None
_recs_embeddings: HuggingFaceEmbeddings | None = None

def get_lore_embeddings() -> HuggingFaceEmbeddings:
    global _lore_embeddings
    if _lore_embeddings is None:
        _lore_embeddings = HuggingFaceEmbeddings(
            model_name=settings.LORE_EMBEDDING_MODEL,
            encode_kwargs={"normalize_embeddings": True},
            model_kwargs={"device": settings.EMBEDDING_DEVICE},
        )
    return _lore_embeddings

def get_recs_embeddings() -> HuggingFaceEmbeddings:
    global _recs_embeddings
    if _recs_embeddings is None:
        _recs_embeddings = HuggingFaceEmbeddings(
            model_name=settings.RECS_EMBEDDING_MODEL,
            encode_kwargs={"normalize_embeddings": True},
            model_kwargs={"device": settings.EMBEDDING_DEVICE},
        )
    return _recs_embeddings
