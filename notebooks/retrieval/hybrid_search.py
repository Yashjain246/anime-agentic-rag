"""
retrieval/hybrid_search.py
===========================
Source notebook: anime RAG manga lore Db creation (final).ipynb
(retrieval section)

STATUS: intentionally not reimplemented here. The 5-layer pipeline
(MultiQueryRetriever -> EnsembleRetriever(dense+BM25) -> ChromaDB
spoiler-filter -> BM25 post-filter -> CrossEncoderReranker) is
structurally identical to what's already running in production — an
earlier version used HuggingFaceCrossEncoder + bge-large/GPU where the
current app uses FlashrankRerank + bge-small/CPU, for the same
memory-footprint reasons noted in indexing/lore_db_builder.py.

THE ACTUAL WORKING VERSION:

    src/rag/retriever.py    — build_retriever(), the full pipeline + spoiler firewall
    src/rag/bm25_index.py   — the BM25 half of the ensemble
    src/rag/embeddings.py   — the shared fastembed singleton

This is the current source of truth; the graph in src/agent/nodes.py calls
build_retriever() directly. Duplicating the same pipeline here would just
be a second copy to keep in sync with every future change to the real one.
"""
