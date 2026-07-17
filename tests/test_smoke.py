"""
tests/test_smoke.py
────────────────────
Basic smoke tests — verify all core modules import and initialise without
crashing. These tests do NOT call any external APIs or load the 1.3 GB
embedding model; they only check that the Python modules are importable
and that configuration loads correctly.

Run:
    pytest tests/ -v
"""

from __future__ import annotations

import pytest


# ── Configuration ─────────────────────────────────────────────────────────────

def test_settings_loads():
    """Settings singleton must load without raising."""
    from config.settings import settings
    assert settings.LLM_MODEL == "gemini-3.1-flash-lite"
    assert settings.LORE_EMBEDDING_MODEL.startswith("BAAI/")


def test_settings_paths_exist(tmp_path):
    """DATA_DIR and CHARTS_DIR must be valid Path objects."""
    from config.settings import settings
    assert settings.DATA_DIR is not None
    assert settings.CHARTS_DIR is not None


# ── Agent modules ─────────────────────────────────────────────────────────────

def test_agent_state_importable():
    from src.agent.state import AgentState
    assert AgentState is not None


def test_agent_graph_importable():
    from src.agent.graph import build_graph, get_agent
    assert callable(build_graph)
    assert callable(get_agent)


def test_agent_runner_importable():
    from src.agent.runner import run_agent, run_agent_with_state
    assert callable(run_agent)
    assert callable(run_agent_with_state)


# ── RAG modules ───────────────────────────────────────────────────────────────

def test_retriever_importable():
    from src.rag.retriever import build_retriever
    assert callable(build_retriever)


def test_vectorstores_importable():
    from src.rag.vectorstores import get_lore_vectorstore, get_recs_vectorstore
    assert callable(get_lore_vectorstore)
    assert callable(get_recs_vectorstore)


def test_bm25_importable():
    from src.rag.bm25_index import get_bm25_retriever
    assert callable(get_bm25_retriever)


# ── Tools ─────────────────────────────────────────────────────────────────────

def test_tools_registry_importable():
    from src.tools.registry import TOOLS, get_tools
    assert isinstance(TOOLS, list)


def test_trace_moe_importable():
    from src.tools.trace_moe import trace_moe_vision
    assert trace_moe_vision is not None


def test_omdb_importable():
    from src.tools.omdb import omdb_graph_generator
    assert omdb_graph_generator is not None


def test_jikan_importable():
    from src.tools.jikan import anilist_schedule
    assert anilist_schedule is not None


def test_calendar_importable():
    from src.tools.calendar import google_calendar_add
    assert google_calendar_add is not None


# ── Persona ───────────────────────────────────────────────────────────────────

def test_persona_detector_importable():
    from src.persona.detector import detect_persona_switch
    assert detect_persona_switch("talk like Gojo") is not None


def test_persona_reset_detected():
    from src.persona.detector import detect_persona_switch
    assert detect_persona_switch("go back to normal") == "Default"


def test_persona_no_match():
    from src.persona.detector import detect_persona_switch
    assert detect_persona_switch("What happens in episode 5?") is None


# ── Episode ───────────────────────────────────────────────────────────────────

def test_episode_detector_importable():
    from src.episode.detector import detect_episode_progress
    result = detect_episode_progress("I'm on episode 45 of Demon Slayer")
    assert result is not None
    ep, anime = result
    assert ep == 45
    assert anime == "Demon Slayer"


def test_normalizer():
    from src.episode.normalizer import normalize_anime_name, get_all_canonical_names
    assert normalize_anime_name("jjk") == "Jujutsu Kaisen"
    assert normalize_anime_name("aot") == "Attack on Titan"
    assert normalize_anime_name("unknown") is None
    assert len(get_all_canonical_names()) > 0


# ── DB ────────────────────────────────────────────────────────────────────────

def test_chat_history_importable():
    from src.db.chat_history import ChatHistoryDB, get_db
    assert callable(get_db)


def test_chat_history_create_session(tmp_path):
    """Creates a session and loads its (empty) history without error."""
    from src.db.chat_history import ChatHistoryDB
    db = ChatHistoryDB(db_path=tmp_path / "test.db")
    sid = db.create_session()
    assert len(sid) > 0
    history = db.load_history(sid)
    assert history == []


# ── Prompts ───────────────────────────────────────────────────────────────────

def test_router_prompt():
    from src.prompts.router import build_classification_prompt, RouterOutput
    prompt = build_classification_prompt("Who is Gojo?")
    assert "Gojo" in prompt


def test_respond_prompt_lore():
    from src.prompts.respond import build_system_prompt
    system, _ = build_system_prompt("LORE", "You are an assistant.", "Chapter 1: ...")
    assert "CONTEXT" in system


def test_respond_prompt_spoiler_block():
    from src.prompts.respond import build_system_prompt
    system, _ = build_system_prompt("LORE", "You are an assistant.", "NO_CONTEXT_FOUND")
    assert "spoiler" in system.lower() or "beyond" in system.lower()
