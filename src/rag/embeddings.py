"""
src/rag/embeddings.py
─────────────────────
Lazy singleton for BGE embeddings via fastembed (ONNX runtime).

WHY fastembed instead of sentence-transformers/HuggingFaceEmbeddings:
  Importing torch + sentence-transformers costs ~1GB of RAM before a
  single model weight is even loaded — that alone blows past Streamlit
  Community Cloud's 1GB free-tier cap. fastembed runs the same BGE
  models through ONNX Runtime with no torch dependency, cutting the
  same embedding capability down to ~200MB total.

WHY lazy loading fixes the 3-4 minute startup delay:
  The model is loaded only on the FIRST call to get_lore_embeddings()/
  get_recs_embeddings(). For GENERAL / TOOL / PERSONA_SWITCH /
  EPISODE_UPDATE paths, it's never loaded at all. For LORE / RECOMMEND
  paths it loads once, then all subsequent queries reuse the cached
  instance.
"""

from __future__ import annotations

from langchain_community.embeddings import FastEmbedEmbeddings

from config.settings import settings

_lore_embeddings: FastEmbedEmbeddings | None = None
_recs_embeddings: FastEmbedEmbeddings | None = None


def get_lore_embeddings() -> FastEmbedEmbeddings:
    global _lore_embeddings
    if _lore_embeddings is None:
        _lore_embeddings = FastEmbedEmbeddings(model_name=settings.LORE_EMBEDDING_MODEL)
    return _lore_embeddings


def get_recs_embeddings() -> FastEmbedEmbeddings:
    """
    Lore and recs use the same model by default (BAAI/bge-small-en-v1.5) —
    reuse the lore singleton instead of loading a second copy of the same
    weights into memory. Only instantiates a separate model if the two
    settings are ever configured to diverge.
    """
    global _recs_embeddings
    if settings.RECS_EMBEDDING_MODEL == settings.LORE_EMBEDDING_MODEL:
        return get_lore_embeddings()
    if _recs_embeddings is None:
        _recs_embeddings = FastEmbedEmbeddings(model_name=settings.RECS_EMBEDDING_MODEL)
    return _recs_embeddings
