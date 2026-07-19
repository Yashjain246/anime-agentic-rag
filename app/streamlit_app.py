"""
app/streamlit_app.py
─────────────────────
Production-quality Streamlit frontend for the Anime Agentic RAG bot.

Features:
  - Dark anime-themed UI with custom CSS
  - Streaming responses (token by token)
  - Persistent chat history via SQLite
  - Status/progress indicator showing current graph node
  - Left sidebar: anime selector, episode progress, spoiler toggle, persona
  - File upload for trace.moe screenshot identification
  - LangSmith trace link when tracing is enabled
"""

from __future__ import annotations

import os
import sys
import tempfile
import uuid
from pathlib import Path

# ── Add project root to sys.path ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Anime RAG Bot",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Deferred imports (after page config) ─────────────────────────────────────
from config.settings import settings
from src.agent.runner import run_agent_with_state, stream_agent_with_state
from src.db.chat_history import get_db
from src.episode.normalizer import get_all_canonical_names
from src.persona.character_db import get_character_db
from PIL import Image
import pillow_heif

pillow_heif.register_heif_opener()  # lets PIL.Image.open() decode HEIC/HEIF uploads

settings.setup_langsmith()
settings.ensure_dirs()

# ────────────────────────────────────────────────────────────────────────────
# Custom CSS — Dark anime aesthetic
# ────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Rajdhani:wght@500;600;700&display=swap');

  /* ── Base ── */
  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0a0a0f;
    color: #e8e8f0;
  }

  /* ── Main container ── */
  .main .block-container {
    padding: 1.5rem 2rem 2rem;
    max-width: 900px;
  }

  /* ── Header ── */
  .anime-header {
    background: linear-gradient(135deg, #1a0533 0%, #0d1f3c 50%, #0a0a0f 100%);
    border: 1px solid rgba(139, 92, 246, 0.3);
    border-radius: 16px;
    padding: 1.5rem 2rem;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 1rem;
    box-shadow: 0 0 40px rgba(139, 92, 246, 0.15);
  }
  .anime-header h1 {
    font-family: 'Rajdhani', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(90deg, #a855f7, #3b82f6, #ec4899);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
  }
  .anime-header p {
    color: #94a3b8;
    margin: 0;
    font-size: 0.9rem;
  }

  /* ── Chat messages ── */
  .stChatMessage {
    background: transparent !important;
    border: none !important;
    gap: 1rem !important;
  }
  /* Streamlit 1.58 renders the avatar as a bare <img alt="user avatar"> /
     <img alt="assistant avatar"> directly inside .stChatMessage — it carries
     no data-testid at all, so the previous "stChatMessageAvatar" selector
     (an exact-match testid that doesn't exist in this version) never
     applied. Target by the alt text instead, which is stable. */
  .stChatMessage img[alt="user avatar"],
  .stChatMessage img[alt="assistant avatar"] {
    width: 3.6rem !important;
    height: 3.6rem !important;
    border-radius: 50% !important;
    object-fit: cover !important;
    border: 2px solid rgba(139, 92, 246, 0.45) !important;
    box-shadow: 0 0 18px rgba(139, 92, 246, 0.25) !important;
    flex-shrink: 0 !important;
    margin-top: 0.2rem;
  }
  .user-bubble {
    background: linear-gradient(135deg, #1e1b4b, #312e81);
    border: 1px solid rgba(139, 92, 246, 0.4);
    border-radius: 16px 16px 4px 16px;
    padding: 0.9rem 1.2rem;
    margin: 0.5rem 0;
    max-width: 80%;
    margin-left: auto;
    box-shadow: 0 4px 15px rgba(139, 92, 246, 0.1);
    line-height: 1.6;
  }
  .bot-bubble {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    border: 1px solid rgba(59, 130, 246, 0.3);
    border-radius: 16px 16px 16px 4px;
    padding: 0.9rem 1.2rem;
    margin: 0.5rem 0;
    max-width: 88%;
    box-shadow: 0 4px 15px rgba(59, 130, 246, 0.08);
    line-height: 1.7;
  }

  /* ── Status indicator ── */
  .status-box {
    background: linear-gradient(90deg, rgba(16,24,40,0.9), rgba(30,41,59,0.9));
    border: 1px solid rgba(99, 102, 241, 0.4);
    border-left: 3px solid #6366f1;
    border-radius: 8px;
    padding: 0.6rem 1rem;
    font-size: 0.82rem;
    color: #94a3b8;
    margin: 0.5rem 0;
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }
  .status-pulse {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #6366f1;
    animation: pulse 1.2s infinite;
    flex-shrink: 0;
  }
  @keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%       { opacity: 0.4; transform: scale(0.8); }
  }

  /* ── Agent step list — spinning circle while a step runs, then a
     green checkmark once it's done, inside the "Agent thinking..."
     status container ── */
  .step-list {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
    padding: 0.15rem 0;
  }
  .step-row {
    display: flex;
    align-items: center;
    gap: 0.65rem;
    font-size: 0.85rem;
    color: #cbd5e1;
  }
  .step-row.step-active { color: #e8e8f0; }
  .step-icon {
    flex-shrink: 0;
    width: 1.05rem;
    height: 1.05rem;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
  }
  .step-spinner {
    border: 2px solid rgba(139,92,246,0.25);
    border-top-color: #a855f7;
    animation: step-spin 0.7s linear infinite;
  }
  @keyframes step-spin {
    to { transform: rotate(360deg); }
  }
  /* Animated success checkmark — a ring draws itself, fills in, then the
     tick draws on top, in sequence (~0.7s total). Inspired by the
     Touch ID confirmation animation: a deliberate draw-in rather than an
     icon just appearing instantly. Pure CSS/SVG, no JS. */
  .step-check-wrap {
    width: 1.05rem;
    height: 1.05rem;
  }
  .step-check-svg { overflow: visible; }
  .step-ring {
    fill: none;
    stroke: #10b981;
    stroke-width: 3;
    stroke-dasharray: 95;
    stroke-dashoffset: 95;
    animation: step-ring-draw 0.35s ease-out forwards;
  }
  .step-fill {
    fill: #10b981;
    opacity: 0;
    transform-origin: center;
    transform: scale(0);
    filter: drop-shadow(0 0 6px rgba(16,185,129,0.6));
    animation: step-fill-pop 0.3s cubic-bezier(0.34, 1.56, 0.64, 1) 0.3s forwards;
  }
  .step-tick {
    fill: none;
    stroke: #052e1d;
    stroke-width: 4;
    stroke-linecap: round;
    stroke-linejoin: round;
    stroke-dasharray: 24;
    stroke-dashoffset: 24;
    animation: step-tick-draw 0.2s ease-out 0.48s forwards;
  }
  @keyframes step-ring-draw {
    to { stroke-dashoffset: 0; }
  }
  @keyframes step-fill-pop {
    to { opacity: 1; transform: scale(1); }
  }
  @keyframes step-tick-draw {
    to { stroke-dashoffset: 0; }
  }

  /* ── Intent badge ── */
  .intent-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    padding: 0.2rem 0.7rem;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    margin-bottom: 0.4rem;
  }
  .badge-lore      { background: rgba(168,85,247,0.2); color: #c084fc; border: 1px solid rgba(168,85,247,0.4); }
  .badge-recommend { background: rgba(16,185,129,0.2); color: #34d399; border: 1px solid rgba(16,185,129,0.4); }
  .badge-tool      { background: rgba(245,158,11,0.2); color: #fbbf24; border: 1px solid rgba(245,158,11,0.4); }
  .badge-general   { background: rgba(59,130,246,0.2); color: #60a5fa; border: 1px solid rgba(59,130,246,0.4); }
  .badge-persona   { background: rgba(236,72,153,0.2); color: #f472b6; border: 1px solid rgba(236,72,153,0.4); }

  /* ── "How it works" — top-left of the main content area ── */
  .st-key-how_it_works_top_btn button {
    border-radius: 999px !important;
    font-size: 0.78rem !important;
    padding: 0.3rem 0.9rem !important;
    font-weight: 500 !important;
    background: rgba(139,92,246,0.1) !important;
    border: 1px solid rgba(139,92,246,0.35) !important;
    color: #c4b5fd !important;
    transition: all 0.2s ease !important;
  }
  .st-key-how_it_works_top_btn button:hover {
    background: rgba(139,92,246,0.25) !important;
    border-color: rgba(168,85,247,0.6) !important;
    color: #e9d5ff !important;
  }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d0d1a 0%, #0a0a0f 100%);
    border-right: 1px solid rgba(139,92,246,0.2);
  }
  [data-testid="stSidebar"] .stSelectbox label,
  [data-testid="stSidebar"] .stSlider label,
  [data-testid="stSidebar"] .stToggle label,
  [data-testid="stSidebar"] .stNumberInput label {
    color: #94a3b8 !important;
    font-size: 0.82rem;
    font-weight: 500;
  }
  .sidebar-section {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 0.9rem;
    margin: 0.7rem 0;
  }
  .sidebar-title {
    font-family: 'Rajdhani', sans-serif;
    font-weight: 600;
    font-size: 0.85rem;
    color: #a855f7;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
  }

  /* ── Input area ── */
  .stChatInputContainer {
    background: rgba(15,23,42,0.9) !important;
    border: 1px solid rgba(99,102,241,0.4) !important;
    border-radius: 12px !important;
  }

  /* ── Sidebar chat-history buttons ── */
  [data-testid="stSidebar"] button {
    border-radius: 8px !important;
  }

  /* ── Destructive admin action — stays red regardless of theme primary ── */
  .st-key-admin_clear_btn button {
    background: rgba(239,68,68,0.15) !important;
    border: 1px solid rgba(239,68,68,0.5) !important;
    color: #fca5a5 !important;
  }
  .st-key-admin_clear_btn button:hover {
    background: rgba(239,68,68,0.28) !important;
    border-color: #ef4444 !important;
    color: #fecaca !important;
  }

  /* ── Scrollbar ── */
  ::-webkit-scrollbar { width: 4px; }
  ::-webkit-scrollbar-track { background: #0a0a0f; }
  ::-webkit-scrollbar-thumb { background: #4338ca; border-radius: 2px; }
</style>
""", unsafe_allow_html=True)

# ── Anonymous per-browser identity ────────────────────────────────────────────
def _get_or_create_anon_user_id() -> str:
    """
    Stable per-browser identity via a URL query param — no login required.
    On first visit, generate a UUID and write it into the URL so it
    survives page reloads (same URL = same identity, same chat history).
    A fresh tab/browser without the ?uid= param gets a new identity —
    that's the expected anonymous-session tradeoff (no cookies/
    localStorage involved, so it doesn't survive a bare URL with the
    param stripped, e.g. a bookmarked/shared link).
    """
    uid = st.query_params.get("uid")
    if not uid:
        uid = str(uuid.uuid4())
        st.query_params["uid"] = uid
    return uid


# ── Session state initialisation ─────────────────────────────────────────────
def _init_session():
    if "anon_user_id" not in st.session_state:
        st.session_state.anon_user_id = _get_or_create_anon_user_id()

    defaults = {
        "messages": [],           # list of {"role": "user"/"assistant", "content": str, "intent": str}
        "lc_messages": [],        # LangChain message objects for the agent
        "session_id": None,       # current DB session ID
        "anime_name": "",
        "current_chapter": 9999,
        "spoiler_mode": False,
        "persona": "Default",
        "last_intent": "",
        "image_path": None,
        "suggestion_picks": None, # random picks stable per session
        "seen_intro": False,      # whether the "How it works" dialog has been dismissed
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

_init_session()

# ── Helper: intent badge HTML ─────────────────────────────────────────────────
_INTENT_BADGE_CLASSES = {
    "LORE": "badge-lore",
    "RECOMMEND": "badge-recommend",
    "TOOL": "badge-tool",
    "GENERAL": "badge-general",
    "PERSONA_SWITCH": "badge-persona",
    "EPISODE_UPDATE": "badge-persona",
}

def _intent_badge(intent: str) -> str:
    cls = _INTENT_BADGE_CLASSES.get(intent, "badge-general")
    label = intent.replace("_", " ").title()
    return f'<span class="intent-badge {cls}">{label}</span>'

# ── Helper: agent step list — spinner while running, animated checkmark once
# done ──────────────────────────────────────────────────────────────────────
_STEP_CHECK_SVG = (
    '<span class="step-icon step-check-wrap">'
    '<svg class="step-check-svg" viewBox="0 0 36 36" width="17" height="17">'
    '<circle class="step-ring" cx="18" cy="18" r="15"/>'
    '<circle class="step-fill" cx="18" cy="18" r="15"/>'
    '<path class="step-tick" d="M10 18.5l5.5 5.5L26 12"/>'
    '</svg></span>'
)


def _step_row_html(label: str, active: bool) -> str:
    icon = '<span class="step-icon step-spinner"></span>' if active else _STEP_CHECK_SVG
    row_cls = "step-row step-active" if active else "step-row step-done"
    return f'<div class="{row_cls}">{icon}<span class="step-label">{label}</span></div>'


def _render_steps_html(steps: list[str], in_progress: bool) -> str:
    """Static list render — used for history replay, where every step is
    already resolved (all done, or the last one active if still streaming)."""
    last_idx = len(steps) - 1
    rows = [
        _step_row_html(label, active=(in_progress and i == last_idx))
        for i, label in enumerate(steps)
    ]
    return f'<div class="step-list">{"".join(rows)}</div>'

# ── Helper: start new session ────────────────────────────────────────────────
def _new_session():
    # Reset UI state entirely to a clean slate (do not create DB entry yet)
    st.session_state.session_id = None
    st.session_state.messages = []
    st.session_state.lc_messages = []
    st.session_state.last_intent = ""
    st.session_state.image_path = None

# ── "How it works" onboarding dialog ──────────────────────────────────────────
@st.dialog("How it works", width="large")
def _how_it_works_dialog():
    st.markdown("""
    <div style="line-height:1.9; font-size:0.95rem;">
      <p><b style="color:#a855f7">1. Just ask</b> — type anything in the chat
      box: plot &amp; lore questions, "suggest anime like X", airing
      schedules, episode ratings charts, or just casual anime chat.</p>
      <p><b style="color:#60a5fa">2. Set your anime &amp; episode</b>
      (sidebar) — this keeps lore answers spoiler-safe for anything beyond
      where you've read or watched.</p>
      <p><b style="color:#f472b6">3. Pick a persona</b> (sidebar) — have the
      bot talk as one of 738 characters instead of its default voice.</p>
      <p><b style="color:#fbbf24">4. Upload a screenshot</b> (sidebar) — find
      out what anime and episode it's from.</p>
      <p><b style="color:#34d399">5. Airing soon?</b> — when the bot gives
      you a broadcast time, it'll offer to add it straight to your Google
      Calendar.</p>
      <p style="color:#94a3b8; font-size:0.85rem; margin-top:1rem;">Your
      chat history saves automatically to this browser — reopen an old
      conversation or start a new one anytime from the sidebar.</p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Got it, let's start", type="primary", use_container_width=True):
        st.session_state.seen_intro = True
        # Persisted in the URL (same mechanism as the anon uid), not just
        # session_state — session_state alone resets on every page reload,
        # and deriving "seen it before" from chat history (has any saved
        # sessions?) had a real bug: clearing all chat history via the
        # sidebar delete buttons made a returning user look first-time
        # again and re-triggered this dialog every time their history hit
        # zero. A dedicated query param means "already saw the intro" and
        # "currently has chat history" are tracked independently.
        st.query_params["intro_seen"] = "1"
        st.rerun()


def _is_returning_user() -> bool:
    """True if this anon browser identity already has saved chat history —
    used so an existing user (from before this dialog existed, so no
    intro_seen param yet) isn't treated as first-time just because the
    query param is missing."""
    try:
        return len(get_db().list_sessions(user_id=st.session_state.anon_user_id)) > 0
    except Exception:
        return True  # DB unreachable — don't nag with the intro

if (
    not st.session_state.seen_intro
    and not st.session_state.messages
    and "intro_seen" not in st.query_params
    and not _is_returning_user()
):
    _how_it_works_dialog()

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    # Header
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0 0.5rem;">
      <h2 style="font-family:'Rajdhani',sans-serif; font-size:1.4rem;
                 background:linear-gradient(90deg,#a855f7,#3b82f6);
                 -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                 margin:0.2rem 0 0;">Anime RAG Bot</h2>
      <p style="color:#475569; font-size:0.75rem; margin:0;">Agentic • RAG • Spoiler-Safe</p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── What can I ask? (always visible, not just on an empty chat) ─────────
    with st.expander("What can I ask?"):
        st.markdown(
            "- **Plot & lore** for Demon Slayer, Jujutsu Kaisen, Attack on "
            "Titan, or Chainsaw Man — set your episode below and answers "
            "stay spoiler-safe for anything beyond it\n"
            "- **Recommendations** — \"suggest anime like X\"\n"
            "- **Airing schedules** — next-episode times, with an offer to "
            "add them to your Google Calendar\n"
            "- **Episode ratings** — a ratings chart for any season\n"
            "- **Screenshot ID** — upload an image, find out what it's from\n"
            "- **Character personas** — pick any of 738 characters below "
            "and the bot talks as them"
        )

    # ── Anime Settings ─────────────────────────────────────────────────────
    st.markdown('<div class="sidebar-title">Anime Settings</div>', unsafe_allow_html=True)

    anime_options = ["(All Anime)"] + get_all_canonical_names()
    selected_anime = st.selectbox(
        "Active Anime",
        options=anime_options,
        index=0,
        key="anime_select",
        help="Filter lore questions to this anime",
    )
    st.session_state.anime_name = "" if selected_anime == "(All Anime)" else selected_anime

    col1, col2 = st.columns([3, 2])
    with col1:
        ep_input = st.number_input(
            "Current Episode",
            min_value=0, max_value=1000, value=0, step=1,
            help="Set your episode progress for spoiler protection",
        )
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Set", use_container_width=True, key="set_ep") and ep_input > 0:
            if st.session_state.anime_name:
                # Inject episode update as a user message
                st.session_state._pending_episode_msg = (
                    f"I'm on episode {ep_input} of {st.session_state.anime_name}"
                )
            else:
                st.warning("Select an anime first")

    spoiler_mode = st.toggle(
        "Spoiler Mode (see everything)",
        value=st.session_state.spoiler_mode,
        key="spoiler_toggle",
    )
    st.session_state.spoiler_mode = spoiler_mode

    st.divider()

    # ── Persona ────────────────────────────────────────────────────────────
    st.markdown('<div class="sidebar-title">Persona</div>', unsafe_allow_html=True)

    db_chars = get_character_db()
    # Full list, not capped: a selectbox with a couple hundred extra plain
    # strings is not a real perf concern, and capping it previously meant
    # ~4 in 5 characters (anything past the cutoff alphabetically) weren't
    # even selectable here at all — see the sync note below for why that
    # combination was actively breaking chat-triggered persona switches.
    char_names = ["Default"] + sorted(
        c.get("name", "") for c in db_chars.values()
        if c.get("name")
    )

    # A selectbox with key= treats st.session_state[key] as its value on
    # every rerun after the first, ignoring index= entirely. persona can
    # also change from elsewhere — a "talk to me like X" chat message
    # (persona_node) or loading an old session from history — and without
    # this sync, the *next* rerun's selectbox would silently overwrite
    # st.session_state.persona back to whatever it last held (confirmed
    # live: a chat-triggered switch to Satoru Gojo answered correctly once,
    # then reverted to "Default" as soon as the page rerendered).
    if st.session_state.get("persona_select") != st.session_state.persona:
        st.session_state.persona_select = (
            st.session_state.persona if st.session_state.persona in char_names else "Default"
        )

    selected_persona = st.selectbox(
        "Bot Persona",
        options=char_names,
        key="persona_select",
        help="Choose which character the bot should speak as",
    )
    st.session_state.persona = selected_persona

    if selected_persona != "Default":
        char = db_chars.get(selected_persona.lower())
        if char:
            traits = char.get("personality", {}).get("traits", [])
            if traits:
                st.caption(f"{', '.join(traits[:3])}")

    st.divider()

    # ── Screenshot Upload ──────────────────────────────────────────────────
    st.markdown('<div class="sidebar-title">Screenshot ID</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload an anime screenshot",
        type=["jpg", "jpeg", "png", "webp", "heic", "heif"],
        key="screenshot_upload",
        label_visibility="collapsed",
        help="iPhone/Mac HEIC screenshots are supported too",
    )
    if uploaded_file:
        # Decode through PIL and normalize to JPEG regardless of the source
        # format. This is what makes HEIC/HEIF (the default format for
        # iPhone/Mac photos, though Apple's own screenshot tool actually
        # saves PNG — a HEIC upload here is more likely a photo of a screen
        # or a converted screenshot) work at all: browsers generally can't
        # preview HEIC in an <img> tag, and trace.moe's API isn't guaranteed
        # to accept it, so decoding once with pillow-heif's registered
        # opener and re-encoding to JPEG sidesteps both problems for every
        # format uniformly.
        try:
            img = Image.open(uploaded_file)
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                img.save(tmp, format="JPEG", quality=92)
                st.session_state.image_path = tmp.name
            st.image(img, caption="Uploaded screenshot", use_container_width=True)
            if st.button("Identify this screenshot", use_container_width=True):
                st.session_state._pending_screenshot_msg = (
                    "What anime is this screenshot from?"
                )
        except Exception:
            st.error("Couldn't read that image — try a different file.")
            st.session_state.image_path = None
    else:
        st.session_state.image_path = None

    st.divider()

    # ── Chat History ───────────────────────────────────────────────────────
    st.markdown('<div class="sidebar-title">Chat History</div>', unsafe_allow_html=True)

    if st.button("New Chat", use_container_width=True, type="primary"):
        _new_session()
        st.rerun()

    try:
        db = get_db()
        sessions = db.list_sessions(user_id=st.session_state.anon_user_id)

        if sessions:
            st.caption(f"{len(sessions)} previous conversation(s)")
            for sess in sessions[:15]:  # show last 15
                preview = db.get_session_preview(sess["session_id"])
                is_current = sess["session_id"] == st.session_state.session_id
                label = f"{'▶ ' if is_current else ''}{preview}"
                
                h_col1, h_col2 = st.columns([8, 2])
                with h_col1:
                    if st.button(label, key=f"hist_{sess['session_id']}", use_container_width=True):
                        # Load this session
                        history = db.load_history(sess["session_id"])
                        st.session_state.session_id = sess["session_id"]
                        st.session_state.lc_messages = history
                        st.session_state.anime_name = sess.get("anime_name", "")
                        st.session_state.persona = sess.get("persona", "Default")
                        # Rebuild display messages from history
                        msgs = []
                        for msg in history:
                            from langchain_core.messages import HumanMessage, AIMessage
                            if isinstance(msg, HumanMessage):
                                msgs.append({"role": "user", "content": msg.content, "intent": ""})
                            elif isinstance(msg, AIMessage):
                                msgs.append({"role": "assistant", "content": msg.content, "intent": ""})
                        st.session_state.messages = msgs
                        st.rerun()
                with h_col2:
                    if st.button("×", key=f"del_{sess['session_id']}", help="Delete chat", use_container_width=False):
                        db.delete_session(sess["session_id"])
                        if is_current:
                            st.session_state.session_id = None
                            st.session_state.messages = []
                            st.session_state.lc_messages = []
                        st.rerun()
    except Exception as _db_err:
        st.caption("Chat history unavailable: DB connection error.")

    # ── Admin panel (hidden unless ADMIN_PASSWORD is configured) ────────────
    if settings.ADMIN_PASSWORD:
        st.divider()
        with st.expander("Admin", expanded=st.session_state.get("is_admin", False)):
            if not st.session_state.get("is_admin"):
                admin_pw = st.text_input("Password", type="password", key="admin_pw_input")
                if st.button("Unlock", key="admin_unlock_btn"):
                    if admin_pw == settings.ADMIN_PASSWORD:
                        st.session_state.is_admin = True
                        st.rerun()
                    else:
                        st.error("Incorrect password")
            else:
                if st.session_state.pop("admin_just_cleared", False):
                    # Shown on the run AFTER the clear (not the same run that
                    # calls st.rerun() below) — a message queued right before
                    # st.rerun() can be interrupted before it ever reaches the
                    # browser, so we flag-and-show-next-run instead.
                    st.success("Database cleared.")

                try:
                    db = get_db()
                    stats = db.get_stats()
                    st.metric("Total sessions (all users)", stats["sessions"])
                    st.metric("Total messages (all users)", stats["turns"])
                    if stats["db_size_mb"] is not None:
                        st.metric("DB file size", f"{stats['db_size_mb']:.2f} MB")

                    st.warning("Clearing deletes ALL users' chat history. This cannot be undone.")
                    confirm_clear = st.checkbox("I understand this is irreversible", key="admin_confirm_clear")
                    if st.button(
                        "Clear entire database",
                        type="primary",
                        disabled=not confirm_clear,
                        key="admin_clear_btn",
                    ):
                        db.clear_all()
                        st.session_state.session_id = None
                        st.session_state.messages = []
                        st.session_state.lc_messages = []
                        st.session_state.admin_just_cleared = True
                        st.rerun()
                except Exception as _admin_db_err:
                    st.caption("Admin stats unavailable: DB connection error.")

                if st.button("Lock admin panel", key="admin_lock_btn"):
                    st.session_state.is_admin = False
                    st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# MAIN AREA — Header
# ─────────────────────────────────────────────────────────────────────────────
# Top-left entry point for onboarding — moved out of the sidebar (easy to
# miss behind the collapse toggle) to the very top of the main content
# area instead, as far up-left as this layout allows, ahead of the
# anime-header banner.
top_left_col, _top_spacer = st.columns([1.3, 8.7])
with top_left_col:
    if st.button("How it works", key="how_it_works_top_btn", use_container_width=True):
        _how_it_works_dialog()

st.markdown("""
<div class="anime-header">
  <div>
    <h1>Anime Agentic RAG</h1>
    <p>Spoiler-safe lore • Smart recommendations • Real-time tools • 738 character personas</p>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Active status bar ─────────────────────────────────────────────────────────
status_parts = []
if st.session_state.anime_name:
    status_parts.append(f"Anime: {st.session_state.anime_name}")
if st.session_state.persona != "Default":
    status_parts.append(f"Persona: {st.session_state.persona}")
if st.session_state.spoiler_mode:
    status_parts.append("Spoiler Mode ON")
elif st.session_state.current_chapter < 9999:
    status_parts.append(f"Chapter cap: {st.session_state.current_chapter}")

if status_parts:
    st.markdown(
        f'<div class="status-box"><span class="status-pulse"></span>'
        f'{"  ·  ".join(status_parts)}</div>',
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# CHAT DISPLAY
# ─────────────────────────────────────────────────────────────────────────────

# Display existing messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="app/assets/user_avatar.png" if msg["role"] == "user" else "app/assets/bot_avatar.png"):
        if msg["role"] == "user":
            st.markdown(
                f'<div class="user-bubble">{msg["content"]}</div>',
                unsafe_allow_html=True,
            )
        else:
            if msg.get("steps"):
                with st.status("Response generated", state="complete"):
                    st.markdown(
                        _render_steps_html(msg["steps"], in_progress=False),
                        unsafe_allow_html=True,
                    )
            if msg.get("intent"):
                st.markdown(_intent_badge(msg["intent"]), unsafe_allow_html=True)
            st.markdown(
                f'<div class="bot-bubble">{msg["content"]}</div>',
                unsafe_allow_html=True,
            )
            # Try to extract an image path from the text just in case it wasn't explicitly saved
            import re
            png_match = re.search(r'([A-Za-z]:\\[^\s]+\.png|/[^\s]+\.png)', msg["content"])
            img_path = msg.get("image") or (png_match.group(1) if png_match else None)
            
            if img_path and Path(img_path).exists():
                st.image(Image.open(img_path), use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# SUGGESTED QUESTIONS (shown only on empty chat)
# ─────────────────────────────────────────────────────────────────────────────
import random as _random

# Pool — one question randomly picked per category each session.
# Curated to only include questions this app actually answers well:
# LORE is restricted to the 4 anime actually indexed in the Lore DB
# (src/episode/normalizer.py SUPPORTED_ANIME) — anything else falls
# outside the retriever's coverage and produces weak/irrelevant
# grounding. RECS entries are spot-checked against the recs vector
# store; overly abstract prompts ("best anime for beginners") were
# dropped after producing unfocused, low-relevance matches.
_LORE_POOL = [
    "What happens to Gojo in Shibuya?",
    "What is Gojo Satoru's Six Eyes and Infinity technique?",
    "What happens to Eren Yeager at the end of Attack on Titan?",
    "How does Tanjiro unlock the Sun Breathing style?",
    "What is Chainsaw Man's contract devil power?",
]
_REC_POOL = [
    "Suggest anime like Attack on Titan",
    "Recommend dark psychological anime like Death Note",
    "What should I watch after Fullmetal Alchemist Brotherhood?",
    "Suggest anime similar to Demon Slayer",
    "Recommend anime with strong female leads",
]
_SCHEDULE_POOL = [
    "When does JJK next episode air?",
    "When does the next episode of One Piece air?",
    "When does Chainsaw Man next air?",
    "What time does My Hero Academia broadcast?",
]
_RATINGS_POOL = [
    "Show me the episode ratings for Death Note",
    "What are Attack on Titan's episode ratings?",
    "Show me Jujutsu Kaisen's ratings chart",
    "What's Demon Slayer's highest rated episode?",
]
# "Talk to me like X" is recognized by src/persona/detector.py's
# PERSONA_SWITCH_PATTERNS and actually switches st.session_state.persona
# (same effect as picking one from the sidebar) — these are confirmed
# canonical names in the character DB, not just plausible-sounding ones.
_PERSONA_POOL = [
    "Talk to me like Satoru Gojo",
    "Talk to me like Levi Ackerman",
    "Talk to me like Tanjiro Kamado",
    "Talk to me like Makima",
    "Talk to me like Denji",
]

# category key → (question pool, badge color hex, rgba triplet for bg/border)
_SUGGESTION_CATEGORIES = {
    "lore":     (_LORE_POOL,     "#c084fc", "168,85,247"),
    "rec":      (_REC_POOL,      "#34d399", "16,185,129"),
    "schedule": (_SCHEDULE_POOL, "#fbbf24", "245,158,11"),
    "ratings":  (_RATINGS_POOL,  "#60a5fa", "59,130,246"),
    "persona":  (_PERSONA_POOL,  "#f472b6", "236,72,153"),
}

# Resolve the chat input (and any pending pill/episode/screenshot message)
# *before* deciding whether to show the welcome screen below. Previously
# st.chat_input() was called after this section, so on the very first
# message of a session the welcome block still rendered "empty chat" (it
# hadn't seen user_input yet) — showing the welcome screen and the new
# exchange stacked on top of each other for that one run. st.chat_input
# stays pinned to the bottom of the page regardless of where it's called,
# so moving it earlier only affects when we read its value, not where it
# renders.
pending = None
if "_pending_msg" in st.session_state:
    pending = st.session_state.pop("_pending_msg")
elif hasattr(st.session_state, "_pending_episode_msg"):
    pending = st.session_state._pending_episode_msg
    del st.session_state._pending_episode_msg
elif hasattr(st.session_state, "_pending_screenshot_msg"):
    pending = st.session_state._pending_screenshot_msg
    del st.session_state._pending_screenshot_msg

user_input = st.chat_input(
    placeholder="Ask about lore, get recommendations, check schedules... or just chat!",
    key="chat_input",
)

# Use pending message if no direct input
if pending and not user_input:
    user_input = pending

# An explicit st.empty() slot, not a bare "if" block, is what actually
# guarantees this content disappears on a later rerun. The keyed pill
# buttons below are exactly the case Streamlit's key= persistence is built
# for (surviving being conditionally skipped) — which meant a bare `if`
# left stale, still-clickable pills on screen even after the guard above
# went False (confirmed live: 4 old pills stayed visible with the wrong,
# un-restyled gray look after the very first real reply). Writing into
# welcome_slot.container() each run, and simply not writing to it when the
# condition is False, reliably clears prior content instead.
welcome_slot = st.empty()
if not st.session_state.messages and not user_input:
    with welcome_slot.container():
        # Show 3 of the 5 categories (lore / recs / schedule / ratings /
        # persona chat), randomly chosen, one question each — stable across
        # widget reruns within a session so the pills don't shuffle out from
        # under the user on every rerun.
        if st.session_state.suggestion_picks is None:
            chosen_categories = _random.sample(list(_SUGGESTION_CATEGORIES), 3)
            st.session_state.suggestion_picks = [
                (cat, _random.choice(_SUGGESTION_CATEGORIES[cat][0]))
                for cat in chosen_categories
            ]

        picks = st.session_state.suggestion_picks

        # Original welcome screen HTML (design unchanged)
        st.markdown(f"""
          <div style="text-align:center; padding: 2rem 1rem 0.5rem 1rem; color:#e8e8f0;">
            <h3 style="font-size:1.6rem; font-weight:700; margin-bottom:0.6rem;
                       background:linear-gradient(90deg,#a855f7,#60a5fa); -webkit-background-clip:text;
                       -webkit-text-fill-color:transparent;">
              Ready to explore the anime world
            </h3>
            <p style="font-size:0.9rem; max-width:560px; margin:0 auto; line-height:1.7; color:#94a3b8;">
              Ask about <b style="color:#a855f7">plot &amp; lore</b>,
              get <b style="color:#34d399">recommendations</b>,
              check <b style="color:#fbbf24">airing schedules</b>,
              see <b style="color:#60a5fa">episode ratings</b>,
              or identify a screenshot.<br><br>
              Set your episode in the sidebar for <b style="color:#f472b6">spoiler-safe</b> answers.
            </p>
          </div>
        """, unsafe_allow_html=True)

        # Clickable pill buttons, colored per category. Scoped with the
        # .st-key-<key> class Streamlit attaches to each widget's wrapper —
        # a plain "nth-child(N) within any stHorizontalBlock" selector (a
        # previous approach here) matches every N-column row on the page,
        # including the sidebar's episode-input/Set-button pair, which
        # silently inherited a pill's color and font/padding overrides.
        # Rules are generated for all 5 categories every time (cheap, and
        # simpler than tracking which 3 are actually showing this run).
        pill_css_rules = "\n".join(
            f"""
          .st-key-pill_{cat} button {{
            border-radius: 8px !important;
            font-size: 0.82rem !important;
            padding: 0.4rem 0.8rem !important;
            font-weight: 400 !important;
            transition: opacity 0.2s, transform 0.2s !important;
            background: rgba({rgba},0.1) !important;
            border: 1px solid rgba({rgba},0.3) !important;
            color: {color} !important;
          }}
          .st-key-pill_{cat} button:hover {{
            opacity: 0.85 !important;
            transform: translateY(-1px) !important;
          }}
            """
            for cat, (_, color, rgba) in _SUGGESTION_CATEGORIES.items()
        )
        st.markdown(f"<style>{pill_css_rules}</style>", unsafe_allow_html=True)

        pill_cols = st.columns(len(picks))
        for col, (cat, question) in zip(pill_cols, picks):
            with col:
                if st.button(question, key=f"pill_{cat}", use_container_width=True):
                    st.session_state._pending_msg = question
                    st.rerun()

if user_input:
    # ── Ensure DB session exists before saving ────────────────────────────
    if not st.session_state.session_id:
        db = get_db()
        new_sid = db.create_session(
            user_id=st.session_state.anon_user_id,
            anime_name=st.session_state.anime_name,
            persona=st.session_state.persona,
        )
        st.session_state.session_id = new_sid

    # ── Display user message ──────────────────────────────────────────────
    st.session_state.messages.append({
        "role": "user",
        "content": user_input,
        "intent": "",
    })
    with st.chat_message("user", avatar="app/assets/user_avatar.png"):
        st.markdown(
            f'<div class="user-bubble">{user_input}</div>',
            unsafe_allow_html=True,
        )

    # ── Status indicator while agent runs ────────────────────────────────
    with st.chat_message("assistant", avatar="app/assets/bot_avatar.png"):
        response_placeholder = st.empty()
        
        # ── Run agent stream ──────────────────────────────────────────────
        agent_steps = []
        # (placeholder, label) pairs, one per step, oldest first. Each step
        # gets its OWN st.empty() rather than all steps sharing one placeholder
        # re-rendered as a whole list. Streamlit's markdown target does a full
        # HTML replace on every call — with one shared placeholder, appending
        # step N+1 meant re-inserting the entire list's HTML, which replayed
        # the draw-in checkmark animation on every already-completed step, not
        # just the new one. A separate placeholder per step means finishing
        # step N only ever touches step N's own DOM node.
        step_slots: list[tuple] = []

        def _finish_previous_step():
            if step_slots:
                prev_ph, prev_label = step_slots[-1]
                prev_ph.markdown(_step_row_html(prev_label, active=False), unsafe_allow_html=True)

        def _start_step(label: str):
            _finish_previous_step()
            ph = st.empty()
            ph.markdown(_step_row_html(label, active=True), unsafe_allow_html=True)
            step_slots.append((ph, label))
            agent_steps.append(label)

        try:
            result = None
            with st.status("Agent thinking...", expanded=True) as status_container:
                for event in stream_agent_with_state(
                    message=user_input,
                    anime_name=st.session_state.anime_name,
                    current_chapter=st.session_state.current_chapter,
                    spoiler_mode=st.session_state.spoiler_mode,
                    persona=st.session_state.persona,
                    image_path=st.session_state.image_path,
                    history=st.session_state.lc_messages,
                ):
                    if event["type"] == "node":
                        # Display what the agent is currently doing with
                        # descriptive text. Note: no status_container.update
                        # (label=...) here — an update() call that doesn't
                        # also repeat expanded=True was silently collapsing
                        # the box on every single step (confirmed via the
                        # rendered <details> losing its "open" attribute
                        # mid-run) — leaving the header static at "Agent
                        # thinking..." avoids needing to fight that, and the
                        # step list below is the actual detail anyway.
                        node = event["name"]
                        node_desc = {
                            "router_node": "Analyzing intent & routing...",
                            "lore_node": "Lore Retrieval (fetching manga chapters & vector search)...",
                            "recs_node": "Recommendations (searching anime synopses)...",
                            "tools_node": "External Tools (calling APIs...)",
                            "respond_node": "Generating Response...",
                            "persona_node": "Personalizing response style...",
                            "episode_node": "Checking Episode Progress...",
                        }.get(node, f"Executing {node}...")

                        _start_step(node_desc)

                        if node == "tools_node":
                            context = event["update"].get("retrieved_context", "")
                            tools_called = [line.strip("[]:") for line in context.split("\n") if line.startswith("[") and line.endswith("]:")]
                            if tools_called:
                                _start_step(f"Tools used: {', '.join(tools_called)}")

                    elif event["type"] == "final":
                        result = event["result"]

                # Stream finished — the last step is done too.
                _finish_previous_step()
                status_container.update(label="Response generated", state="complete")

            reply = result["reply"]
            intent = result["intent"]

            # Update session state from agent result
            st.session_state.persona = result["persona"]
            st.session_state.lc_messages = result["messages"]
            st.session_state.current_chapter = result["current_chapter"]
            st.session_state.anime_name = result["anime_name"]
            st.session_state.spoiler_mode = result["spoiler_mode"]
            st.session_state.last_intent = intent

            # Clean up markdown images from reply just in case
            import re
            clean_reply = re.sub(r'!\[.*?\]\(.*?\)', '', reply).strip()

            # If a tool generated a chart, extract directly from retrieved_context
            chart_path = None
            if intent == "TOOL":
                context_str = result.get("retrieved_context", "")
                for line in context_str.split("\n"):
                    if line.startswith("Chart saved:"):
                        extracted_path = line.replace("Chart saved:", "").strip()
                        if Path(extracted_path).exists():
                            chart_path = extracted_path
                        break
                    
            # Save the message with image to session state
            st.session_state.messages.append({
                "role": "assistant",
                "content": clean_reply,
                "intent": intent,
                "image": chart_path,
                "steps": agent_steps
            })

            # ── Stream response word-by-word for ChatGPT-like feel ────────
            import time as _time
            badge = _intent_badge(intent)
            words = clean_reply.split(" ")
            streamed = ""
            for i, word in enumerate(words):
                streamed += word + (" " if i < len(words) - 1 else "")
                response_placeholder.markdown(
                    badge + f'<div class="bot-bubble">{streamed}▌</div>',
                    unsafe_allow_html=True,
                )
                _time.sleep(0.025)
            # Final render without cursor
            response_placeholder.markdown(
                badge + f'<div class="bot-bubble">{clean_reply}</div>',
                unsafe_allow_html=True,
            )

            
            if chart_path:
                st.image(Image.open(chart_path), use_container_width=True)

        except Exception as e:
            error_msg = f"An error occurred: {e}"
            response_placeholder.error(error_msg)
            clean_reply = error_msg
            intent = "GENERAL"
            st.session_state.messages.append({
                "role": "assistant",
                "content": clean_reply,
                "intent": intent,
                "steps": agent_steps
            })

        # ── Save to DB ────────────────────────────────────────────────────
        try:
            db = get_db()
            db.save_turn(
                session_id=st.session_state.session_id,
                human_msg=user_input,
                ai_msg=reply,
                intent=intent,
                persona=st.session_state.persona,
            )
            db.update_session_meta(
                session_id=st.session_state.session_id,
                anime_name=st.session_state.anime_name,
                persona=st.session_state.persona,
            )
        except Exception:
            pass  # DB errors should never crash the UI

    # Clear image after use
    st.session_state.image_path = None
    st.rerun()

