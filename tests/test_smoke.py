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


def test_settings_strips_whitespace_from_credentials(monkeypatch):
    """Regression, found in a real CI eval run: OMDB_API_KEY arrived from
    GitHub Actions secrets with a trailing newline, which got URL-encoded
    into the request as 'apikey=e7c5279c%0A' and returned 401
    Unauthorized — so every ratings lookup failed while looking like a
    bad-credential problem rather than a whitespace one. Secrets pasted
    into GitHub/Streamlit/.env pick up trailing newlines very easily, so
    every credential field is stripped on load."""
    from config.settings import Settings

    monkeypatch.setenv("OMDB_API_KEY", "abc123\n")
    monkeypatch.setenv("TAVILY_API_KEY", "  tvly-xyz  ")
    monkeypatch.setenv("MAL_CLIENT_ID", "mal456\r\n")
    s = Settings()

    assert s.OMDB_API_KEY == "abc123"
    assert s.TAVILY_API_KEY == "tvly-xyz"
    assert s.MAL_CLIENT_ID == "mal456"


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


def test_persona_switch_with_trailing_modifier_phrase():
    """Regression, found via a real CI eval run: 'Talk like Gojo from
    now on.' never switched persona at all - the regex captured 'gojo
    from now on' as the character name and find_character() found
    nothing, so the whole request silently fell through to normal
    routing instead of switching. Same for other common trailing
    phrases users add after the actual name."""
    from src.persona.detector import detect_persona_switch
    assert detect_persona_switch("Talk like Gojo from now on.") == "Satoru Gojo"
    assert detect_persona_switch("Pretend to be Tanjiro please") is not None


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
    db = ChatHistoryDB(db_url=tmp_path / "test.db")
    assert db._connected, "DB failed to connect — earlier assertions would pass even in no-op mode"
    sid = db.create_session()
    assert len(sid) > 0
    history = db.load_history(sid)
    assert history == []


def test_chat_history_round_trip(tmp_path):
    """Writes a turn and reads it back — catches silent no-op-mode failures
    and connection-release bugs that 'no exception raised' alone would miss."""
    from src.db.chat_history import ChatHistoryDB
    from langchain_core.messages import HumanMessage, AIMessage
    db = ChatHistoryDB(db_url=tmp_path / "test.db")
    assert db._connected

    sid = db.create_session(user_id="test-user")
    db.save_turn(sid, "hello", "hi there")
    db.save_turn(sid, "second question", "second answer")

    history = db.load_history(sid)
    assert len(history) == 4
    assert isinstance(history[0], HumanMessage) and history[0].content == "hello"
    assert isinstance(history[1], AIMessage) and history[1].content == "hi there"
    assert isinstance(history[2], HumanMessage) and history[2].content == "second question"

    sessions = db.list_sessions(user_id="test-user")
    assert len(sessions) == 1
    assert db.get_session_preview(sid) == "hello"

    db.delete_session(sid)
    assert db.load_history(sid) == []


def test_chat_history_many_connections_reused(tmp_path):
    """Repeatedly opens/releases connections (what the sidebar does per
    saved chat) — would have caught the _release_conn recursion bug that
    only manifests once a connection is actually closed/released."""
    from src.db.chat_history import ChatHistoryDB
    db = ChatHistoryDB(db_url=tmp_path / "test.db")
    sid = db.create_session()
    for i in range(20):
        db.save_turn(sid, f"q{i}", f"a{i}")
        db.get_session_preview(sid)
        db.list_sessions()
    assert len(db.load_history(sid)) == 40


def test_site_feedback_one_row_per_user(tmp_path):
    """A user can only ever have one site_feedback row — a repeat
    submission updates it in place, and rating/comment update
    independently without clobbering each other."""
    from src.db.chat_history import ChatHistoryDB
    db = ChatHistoryDB(db_url=tmp_path / "test.db")

    db.add_site_feedback("user1", rating="up")
    assert len(db.get_recent_feedback()) == 1

    db.add_site_feedback("user1", comment="nice project")
    fb = db.get_recent_feedback()
    assert len(fb) == 1
    assert fb[0]["rating"] == "up"  # preserved, not cleared by the comment-only update
    assert fb[0]["comment"] == "nice project"

    db.add_site_feedback("user1", rating="down")
    fb = db.get_recent_feedback()
    assert len(fb) == 1
    assert fb[0]["rating"] == "down"  # overwritten
    assert fb[0]["comment"] == "nice project"  # still preserved

    db.add_site_feedback("user2", rating="up")
    assert len(db.get_recent_feedback()) == 2  # a different user gets their own row


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


def test_respond_prompt_combined_multi_intent():
    """A compound message gets one section per intent, each grounded only
    in its own context — not one blended prompt."""
    from src.prompts.respond import build_combined_system_prompt
    system, _ = build_combined_system_prompt(
        ["LORE", "TOOL"],
        "You are an assistant.",
        {"LORE": "Chapter 1: ...", "TOOL": "Broadcast: Sunday at 21:00 JST"},
    )
    assert "Chapter 1" in system
    assert "Broadcast: Sunday" in system


def test_respond_prompt_combined_acknowledges_general():
    """A GENERAL part riding along a real intent (e.g. "Hi! Who is Muzan?")
    must not get silently dropped just because it has no section of its
    own — see build_combined_system_prompt's had_general handling."""
    from src.prompts.respond import build_combined_system_prompt
    system, _ = build_combined_system_prompt(
        ["GENERAL", "LORE"], "You are an assistant.", {"LORE": "Chapter 1: ..."},
    )
    assert "casual" in system.lower() or "conversational" in system.lower()


def test_lore_guard_uses_scoped_query_not_raw_message():
    """Regression: lore_node used to run its unsupported-anime guard on
    the FULL raw message. In a compound message like "What happens in
    One Piece and when's the next JJK episode?", "JJK" belongs to the
    TOOL half but still made _mentions_supported_anime return True on the
    raw text — which skipped the ANIME_NOT_SUPPORTED refusal and let it
    silently retrieve and answer with a completely unrelated anime's
    lore. router_node's per-intent query decomposition (intent_queries)
    is what fixes this: lore_node must reason about its own scoped query,
    not the raw message that also contains the other intent's content."""
    from src.agent.nodes import _mentions_supported_anime, _mentions_other_known_anime

    raw_message = "What happens in One Piece and when's the next JJK episode?"
    scoped_lore_query = "What happens in One Piece?"

    # The bug: JJK (from the TOOL half) fools the guard on the raw message.
    assert _mentions_supported_anime(raw_message) is True

    # The fix: scoped to just the LORE part, the guard correctly sees no
    # supported anime named, and the unsupported-anime refusal fires.
    assert _mentions_supported_anime(scoped_lore_query) is False
    assert _mentions_other_known_anime(scoped_lore_query) is True


def test_intent_query_falls_back_to_full_message():
    """_intent_query returns the router's scoped sub-query when present,
    and falls back to the full message (never errors) when router_node
    didn't supply one for that intent — e.g. the ValidationError fallback
    path, or a state built without intent_queries at all."""
    from langchain_core.messages import HumanMessage
    from src.agent.nodes import _intent_query

    state = {
        "messages": [HumanMessage(content="How was Muzan killed and when's JJK airing?")],
        "intent_queries": {"LORE": "How was Muzan killed?", "TOOL": "When's JJK airing?"},
    }
    assert _intent_query(state, "LORE") == "How was Muzan killed?"
    assert _intent_query(state, "TOOL") == "When's JJK airing?"

    state_no_queries = {"messages": [HumanMessage(content="Who is Gojo?")], "intent_queries": {}}
    assert _intent_query(state_no_queries, "LORE") == "Who is Gojo?"


def test_mentions_supported_anime_excludes_episode_mapped_only_shows():
    """Regression: _mentions_supported_anime used to check EVERY alias in
    ANIME_NAME_ALIASES, not just the 4 Lore-DB anime's aliases. Frieren and
    Solo Leveling are in that alias map (they're episode-mapped) but are
    NOT lore-indexed - so a message naming Frieren wrongly reported "yes,
    a supported anime is named", which skipped the ANIME_NOT_SUPPORTED
    refusal and let lore_node silently retrieve irrelevant chunks from
    the 4 supported anime instead of honestly declining, the same failure
    shape as the One Piece/AoT bug. Found via 30-case live testing:
    "Recommend anime like Frieren and who is Frieren as a character?"."""
    from src.agent.nodes import _mentions_supported_anime, _mentions_other_known_anime

    assert _mentions_supported_anime("Who is Frieren as a character?") is False
    assert _mentions_other_known_anime("Who is Frieren as a character?") is True

    assert _mentions_supported_anime("What happened to Frieren's party?") is False
    assert _mentions_supported_anime("Recommend anime like Solo Leveling") is False

    # The 4 actually-supported anime must still correctly match.
    assert _mentions_supported_anime("What happens to Gojo in JJK?") is True
    assert _mentions_supported_anime("Tell me about Demon Slayer") is True


def test_router_node_merges_duplicate_intent_queries_not_overwrites():
    """Regression: router_node's intent_queries used to be built with
    {item.intent: item.query for item in result.items} - a plain dict
    comprehension, which means when the router legitimately emits the
    same intent twice (e.g. a schedule ask AND a separate ratings ask are
    both TOOL), the second query silently overwrote the first instead of
    both surviving. Confirmed live: "what's the AOT rating graph, when's
    the next episode airing?" classified as ['TOOL', 'TOOL'] with two
    genuinely distinct, correctly-scoped queries from the LLM, but only
    one reached tools_node - the ratings half vanished entirely and
    omdb_graph_generator was never called for it. This test locks in the
    merge behavior directly (no live LLM call) by driving router_node's
    same code path with a fake structured-output result."""
    from unittest.mock import patch
    from langchain_core.messages import HumanMessage
    from src.prompts.router import RouterOutput, RouterIntentItem
    from src.agent.nodes import router_node

    fake_result = RouterOutput(items=[
        RouterIntentItem(intent="TOOL", query="Show me the AOT rating graph."),
        RouterIntentItem(intent="TOOL", query="When is the next AOT episode airing?"),
    ])
    state = {"messages": [HumanMessage(content="what's the AOT rating graph, when's the next episode airing?")]}

    # Force the slow (LLM) path regardless of what the fast keyword path
    # would do with this message — this test is about router_node's
    # dedup/merge logic downstream of the LLM call, not the fast path.
    with patch("src.agent.nodes._fast_classify", return_value=None), \
         patch("src.agent.nodes.get_query_gen_llm") as mock_llm:
        mock_llm.return_value.with_structured_output.return_value.invoke.return_value = fake_result
        result = router_node(state)

    assert result["intents"] == ["TOOL"]  # deduped, not ['TOOL', 'TOOL']
    combined = result["intent_queries"]["TOOL"]
    assert "rating graph" in combined.lower()
    assert "episode airing" in combined.lower()  # both halves survived, merged


# ── Eval (Phase 1) ───────────────────────────────────────────────────────────────

def test_eval_evaluators_importable():
    from src.eval.evaluators import ALL_EVALUATORS, DETERMINISTIC_EVALUATORS
    assert len(ALL_EVALUATORS) == len(DETERMINISTIC_EVALUATORS) + 1  # + the LLM judge
    assert all(callable(fn) for fn in ALL_EVALUATORS)


def test_deterministic_evaluators_score_a_correct_lore_case():
    """Drives every deterministic evaluator against a synthetic (run, example)
    pair, no live API call — just locking in the calling contract (signature,
    key/score shape, "not applicable" -> score=None) so a refactor of
    src/eval/evaluators.py can't silently break scripts/eval/run_eval.py."""
    from types import SimpleNamespace
    from src.eval.evaluators import DETERMINISTIC_EVALUATORS

    example = SimpleNamespace(
        inputs={"message": "What happens in Chapter 11 of Demon Slayer?", "kwargs": {}},
        outputs={"expected": {
            "intents": ["LORE"], "expected_anime": "Demon Slayer",
            "expected_chapter": 11, "reference_facts": ["x"], "should_decline": False,
        }},
        metadata={"id": "lore_demon_slayer_11", "category": "LORE"},
    )
    run = SimpleNamespace(outputs={
        "reply": "Some grounded reply about chapter 11.",
        "intents": ["LORE"],
        "persona": "Default",
        "current_chapter": 9999,
        "anime_name": "Demon Slayer",
        "retrieved_context": [{"source": "LORE", "text": "[Demon Slayer — Chapter 11]\nsome text"}],
    })

    results = {fn.__name__: fn(run, example) for fn in DETERMINISTIC_EVALUATORS}

    assert results["intent_match"]["score"] == 1
    assert results["every_intent_has_source"]["score"] == 1
    assert results["retrieval_hit"]["score"] == 1
    # Not applicable to a plain LORE case — must be explicitly None, not 0
    # (0 would misreport as "failed" rather than "not scored").
    for key in ("spoiler_not_leaked", "unsupported_anime_refused", "persona_switch_correct", "episode_cap_correct"):
        assert results[key]["score"] is None


def test_golden_dataset_builder_produces_valid_rows():
    """Runs the actual dataset generator (fully local — reads real
    manga_chapters.jsonl, no API calls, no upload) and checks the shape
    every downstream evaluator relies on holds for every row."""
    from scripts.eval.build_golden_dataset import (
        build_lore_cases, build_episode_update_cases, build_recommend_cases,
        build_tool_cases, build_persona_cases, build_general_cases, build_compound_cases,
    )

    all_cases = (
        build_lore_cases() + build_episode_update_cases() + build_recommend_cases()
        + build_tool_cases() + build_persona_cases() + build_general_cases() + build_compound_cases()
    )
    assert len(all_cases) >= 50

    ids = [c["id"] for c in all_cases]
    assert len(ids) == len(set(ids))  # every id unique — required for upload_dataset's upsert-by-id

    for case in all_cases:
        assert case["input"] and isinstance(case["input"], str)
        assert isinstance(case["kwargs"], dict)
        assert "intents" in case["expected"] and isinstance(case["expected"]["intents"], list)

    # Spot-check: episode-cap ground truth actually matches the app's own
    # documented example (episode 12 of Demon Slayer -> chapter 24).
    ep_case = next(c for c in all_cases if c["id"] == "episode_update_demon_slayer_12")
    assert ep_case["expected"]["expected_chapter_cap"] == 24


def test_run_agent_with_state_intents_fallback_on_short_circuit():
    """Regression, found via the Phase 1 eval run: on the PERSONA_SWITCH/
    EPISODE_UPDATE short-circuit paths, only "intent" gets set by the
    graph — "intents" stays at its initial_state value of [] (present,
    just empty). result.get("intents", [default]) never actually applied
    the fallback, because dict.get()'s default only kicks in when a key
    is MISSING, not when it's present-but-falsy — so every persona/episode
    turn reported intents=[] even though persona/current_chapter updated
    correctly. Confirmed live: "Be Levi." scored persona="Levi Ackerman"
    (correct) but intents=[] (wrong) in the same run. Fixed with `or`
    instead of a dict.get() default."""
    from unittest.mock import patch
    from langchain_core.messages import AIMessage

    fake_graph_result = {
        "messages": [AIMessage(content="Tch. Get to the point.")],
        "persona": "Levi Ackerman",
        "intent": "PERSONA_SWITCH",
        "intents": [],  # present but empty — exactly the short-circuit shape
        "current_chapter": 9999,
        "anime_name": "",
        "spoiler_mode": False,
    }
    with patch("src.agent.runner.get_agent") as mock_get_agent:
        mock_get_agent.return_value.invoke.return_value = fake_graph_result
        from src.agent.runner import run_agent_with_state
        result = run_agent_with_state(message="Be Levi.")

    assert result["intents"] == ["PERSONA_SWITCH"]  # not []
    assert result["persona"] == "Levi Ackerman"


def test_every_intent_has_source_exempts_persona_and_episode():
    """Regression, found by the first CI-triggered eval run: this
    evaluator only exempted GENERAL from needing a retrieved_context
    source, but PERSONA_SWITCH and EPISODE_UPDATE also never produce one
    by design (persona_node/episode_node short-circuit before
    lore_node/recs_node/tools_node ever run). Every real PERSONA_SWITCH/
    EPISODE_UPDATE case in that run was correct (persona actually
    switched, chapter cap actually updated) but got wrongly failed here."""
    from types import SimpleNamespace
    from src.eval.evaluators import every_intent_has_source

    for intent in ("PERSONA_SWITCH", "EPISODE_UPDATE"):
        example = SimpleNamespace(inputs={}, outputs={}, metadata={"category": intent})
        run = SimpleNamespace(outputs={"intents": [intent], "retrieved_context": []})
        result = every_intent_has_source(run, example)
        assert result["score"] == 1, f"{intent} should not require a source: {result}"

    # A real LORE case with no source SHOULD still fail — the exemption
    # must be narrow, not a blanket "empty sources is fine."
    example = SimpleNamespace(inputs={}, outputs={}, metadata={"category": "LORE"})
    run = SimpleNamespace(outputs={"intents": ["LORE"], "retrieved_context": []})
    assert every_intent_has_source(run, example)["score"] == 0
