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
from src.db.chat_history import get_db

# ── Safety settings — same as original notebook ───────────────────────────────
_SAFETY_SETTINGS = {
    "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
    "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
    "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
    "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
}

# ── Per-(model, temperature) client cache ─────────────────────────────────────
# Both the primary and fallback model may be in use over the course of a day,
# so this caches a client per combination actually used rather than one fixed
# singleton per role.
_clients: dict[tuple[str, float], ChatGoogleGenerativeAI] = {}


def _select_model() -> str:
    """Picks whichever model still has daily quota left, primary first.
    Each model has its own separate free-tier daily cap on Google's side, so
    once the primary (settings.LLM_MODEL) is used up for the day, requests
    should move to the fallback model instead of starting to fail. If both
    are already exhausted, keep using the fallback rather than blocking the
    user outright — a real quota error from the API is still handled
    upstream, and local counting is only an approximation of Google's own
    reset time."""
    db = get_db()
    primary, fallback = settings.LLM_MODEL, settings.LLM_MODEL_FALLBACK
    limit = settings.LLM_DAILY_QUOTA_PER_MODEL
    if db.get_model_usage_today(primary) < limit:
        return primary
    if db.get_model_usage_today(fallback) < limit:
        return fallback
    return fallback


def _client_for(model_name: str, temperature: float) -> ChatGoogleGenerativeAI:
    key = (model_name, temperature)
    if key not in _clients:
        _clients[key] = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            safety_settings=_SAFETY_SETTINGS,
        )
    return _clients[key]


def get_query_gen_llm() -> ChatGoogleGenerativeAI:
    """
    Returns the deterministic LLM used for:
      - Router classification
      - MultiQueryRetriever sub-query generation
    Temperature=0 → same query always produces same sub-queries.
    """
    model_name = _select_model()
    get_db().increment_model_usage(model_name)
    return _client_for(model_name, settings.QUERY_GEN_TEMPERATURE)


def get_agent_llm() -> ChatGoogleGenerativeAI:
    """
    Returns the creative LLM used for:
      - Final answer generation (respond_node)
      - Persona confirmation messages
      - Episode progress confirmations
      - Tool orchestration (tools_node)
    Temperature=0.7 → natural, slightly creative responses.
    """
    model_name = _select_model()
    get_db().increment_model_usage(model_name)
    return _client_for(model_name, settings.AGENT_TEMPERATURE)
