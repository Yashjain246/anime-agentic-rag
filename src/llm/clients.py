"""
src/llm/clients.py
──────────────────
Lazy singletons for both LLM instances.
Models are instantiated on first access — not at import time.
This keeps startup instant even if the import chain is heavy.
"""

from __future__ import annotations

from langchain_google_genai import ChatGoogleGenerativeAI

from config.settings import settings

# ── Safety settings — same as original notebook ───────────────────────────────
_SAFETY_SETTINGS = {
    "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
    "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
    "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
    "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
}

# ── Private singletons ────────────────────────────────────────────────────────
_query_gen_llm: ChatGoogleGenerativeAI | None = None
_agent_llm: ChatGoogleGenerativeAI | None = None


def get_query_gen_llm() -> ChatGoogleGenerativeAI:
    """
    Returns the deterministic LLM used for:
      - Router classification
      - MultiQueryRetriever sub-query generation
    Temperature=0 → same query always produces same sub-queries.
    """
    global _query_gen_llm
    if _query_gen_llm is None:
        _query_gen_llm = ChatGoogleGenerativeAI(
            model=settings.LLM_MODEL,
            temperature=settings.QUERY_GEN_TEMPERATURE,
            safety_settings=_SAFETY_SETTINGS,
        )
    return _query_gen_llm


def get_agent_llm() -> ChatGoogleGenerativeAI:
    """
    Returns the creative LLM used for:
      - Final answer generation (respond_node)
      - Persona confirmation messages
      - Episode progress confirmations
      - Tool orchestration (tools_node)
    Temperature=0.7 → natural, slightly creative responses.
    """
    global _agent_llm
    if _agent_llm is None:
        _agent_llm = ChatGoogleGenerativeAI(
            model=settings.LLM_MODEL,
            temperature=settings.AGENT_TEMPERATURE,
            safety_settings=_SAFETY_SETTINGS,
        )
    return _agent_llm
