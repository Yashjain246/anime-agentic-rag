"""
src/rag/retriever.py
────────────────────
The full 5-layer RAG retrieval pipeline with optional spoiler firewall.

Pipeline layers:
  1. MultiQueryRetriever  → rewrites user query into 2 targeted sub-queries
  2. EnsembleRetriever    → 60% ChromaDB (dense) + 40% BM25 (sparse)
  3. ChromaDB filter      → hard chapter cap at DB level (spoiler firewall)
  4. BM25 post-filter     → same cap applied after BM25 retrieval
  5. FlashrankRerank      → reranks all candidates by true relevance

WHY filter at the DB level (not post-retrieval):
  If the filter were applied AFTER retrieval, the LLM would still SEE
  the spoiler chunks — it just wouldn't use them. By filtering at the
  ChromaDB metadata level, those vectors are never even scored.

WHY FlashrankRerank instead of a HuggingFace CrossEncoderReranker:
  Same reason as src/rag/embeddings.py — a torch-based cross-encoder
  costs hundreds of MB just to import, on top of the ~1GB torch +
  sentence-transformers already avoided for embeddings. FlashRank runs
  a small ONNX cross-encoder with no torch dependency at all.
"""

from __future__ import annotations

from langchain_classic.retrievers import EnsembleRetriever, MultiQueryRetriever
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors import FlashrankRerank
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever

from config.settings import settings
from src.llm.clients import get_query_gen_llm
from src.rag.bm25_index import get_bm25_retriever
from src.rag.vectorstores import get_lore_vectorstore

# ── Multi-query prompt ────────────────────────────────────────────────────────
MULTI_QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""\
You are an expert at manga and anime lore retrieval.
Rewrite the user's question into exactly 2 targeted search queries
that will find relevant manga chapter summaries in a database.
Rules:
- Each query must approach the question from a DIFFERENT angle
- Use specific character names, locations, or arc names when possible
- Keep each query under 15 words
- Output ONLY the 2 queries, one per line, no numbering

Original question: {question}

2 search queries:""",
)

# ── Reranker singleton ────────────────────────────────────────────────────────
_reranker: FlashrankRerank | None = None


def _get_reranker(top_n: int) -> FlashrankRerank:
    global _reranker
    if _reranker is None:
        _reranker = FlashrankRerank(model=settings.RERANKER_MODEL, top_n=top_n)
    _reranker.top_n = top_n
    return _reranker


# ── Filtered BM25 wrapper ─────────────────────────────────────────────────────
class _FilteredBM25(BaseRetriever):
    """
    Wraps BM25Retriever with metadata filtering so EnsembleRetriever
    can call it with the same interface as ChromaDB.
    BM25 doesn't support metadata filters natively — we apply them
    as a Python post-filter after retrieval.
    """

    anime_name: str | None = None
    max_chapter: int | None = None

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        results = get_bm25_retriever().invoke(query)
        if self.anime_name:
            results = [
                d for d in results if d.metadata["anime_name"] == self.anime_name
            ]
        if self.max_chapter is not None:
            results = [
                d for d in results
                if d.metadata["chapter_number"] <= self.max_chapter
            ]
        return results[: settings.BM25_TOP_K]


# ── Main factory ──────────────────────────────────────────────────────────────
def build_retriever(
    anime_name: str | None = None,
    max_chapter: int | None = None,
    top_n: int | None = None,
) -> ContextualCompressionRetriever:
    """
    Build the full RAG pipeline with an optional spoiler firewall.

    Args:
        anime_name:  Canonical anime name to filter to (None = all anime).
        max_chapter: Hard chapter cap — chapters above this are invisible.
                     None = no spoiler cap (spoiler mode ON).
        top_n:       Number of final results after reranking.

    Example:
        # User is at chapter 90 of JJK — block everything above it
        retriever = build_retriever("Jujutsu Kaisen", max_chapter=90)
        results   = retriever.invoke("What happens to Gojo in Shibuya?")
        # → Returns only chapters 1-90. Ch.91 (Prison Realm) never appears.
    """
    if top_n is None:
        top_n = settings.RERANKER_TOP_N

    # ── ChromaDB metadata filter ──────────────────────────────────────────────
    # ChromaDB uses MongoDB-style operators: $eq, $lte, $and
    chroma_filter = None
    conditions: list[dict] = []
    if anime_name:
        conditions.append({"anime_name": {"$eq": anime_name}})
    if max_chapter is not None:
        conditions.append({"chapter_number": {"$lte": max_chapter}})
    if len(conditions) == 1:
        chroma_filter = conditions[0]
    elif len(conditions) > 1:
        chroma_filter = {"$and": conditions}

    # ── Dense retriever (ChromaDB) ────────────────────────────────────────────
    filtered_dense = get_lore_vectorstore().as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": settings.DENSE_TOP_K,
            **({"filter": chroma_filter} if chroma_filter else {}),
        },
    )

    # ── Sparse retriever (BM25 with post-filter) ──────────────────────────────
    filtered_bm25 = _FilteredBM25(anime_name=anime_name, max_chapter=max_chapter)

    # ── Ensemble: 60% dense + 40% sparse ─────────────────────────────────────
    ensemble = EnsembleRetriever(
        retrievers=[filtered_dense, filtered_bm25],
        weights=[0.6, 0.4],
    )

    # ── MultiQuery: rewrites question into 2 sub-queries ─────────────────────
    multi_query = MultiQueryRetriever.from_llm(
        retriever=ensemble,
        llm=get_query_gen_llm(),
        prompt=MULTI_QUERY_PROMPT,
    )

    # ── FlashRank reranker: final sorting by true relevance ──────────────────
    return ContextualCompressionRetriever(
        base_compressor=_get_reranker(top_n),
        base_retriever=multi_query,
    )
