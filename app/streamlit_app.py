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
from pathlib import Path

# ── Add project root to sys.path ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Anime RAG Bot",
    page_icon="🎌",
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

  /* ── History list ── */
  .history-item {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 8px;
    padding: 0.5rem 0.8rem;
    margin: 0.3rem 0;
    cursor: pointer;
    font-size: 0.82rem;
    color: #94a3b8;
    transition: all 0.2s;
  }
  .history-item:hover {
    background: rgba(99,102,241,0.1);
    border-color: rgba(99,102,241,0.4);
    color: #e8e8f0;
  }

  /* ── Suggestion chips ── */
  .chip-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.6rem;
    justify-content: center;
    margin: 0.8rem 0 1.2rem 0;
  }
  /* Override default Streamlit button styling for chips */
  div[data-testid="column"] > div > div > div > button {
    background: rgba(99,102,241,0.08) !important;
    border: 1px solid rgba(99,102,241,0.3) !important;
    border-radius: 999px !important;
    color: #c4b5fd !important;
    font-size: 0.8rem !important;
    padding: 0.35rem 1rem !important;
    transition: all 0.2s ease !important;
    white-space: nowrap !important;
    font-weight: 500 !important;
  }
  div[data-testid="column"] > div > div > div > button:hover {
    background: rgba(99,102,241,0.25) !important;
    border-color: rgba(168,85,247,0.6) !important;
    color: #e9d5ff !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(99,102,241,0.3) !important;
  }
  .suggestion-chip {
    padding: 0.4rem 0.8rem;
    border-radius: 12px;
    border: 1px solid #334155;
    background: #1e293b;
    color: #cbd5e1;
    font-size: 0.85rem;
    cursor: pointer;
  }

  /* ── Scrollbar ── */
  ::-webkit-scrollbar { width: 4px; }
  ::-webkit-scrollbar-track { background: #0a0a0f; }
  ::-webkit-scrollbar-thumb { background: #4338ca; border-radius: 2px; }
</style>
""", unsafe_allow_html=True)

# ── Session state initialisation ─────────────────────────────────────────────
def _init_session():
    defaults = {
        "messages": [],           # list of {"role": "user"/"assistant", "content": str, "intent": str}
        "lc_messages": [],        # LangChain message objects for the agent
        "session_id": None,       # current DB session ID
        "my_session_ids": [],     # session IDs created in THIS browser session (for privacy)
        "anime_name": "",
        "current_chapter": 9999,
        "spoiler_mode": False,
        "persona": "Default",
        "last_intent": "",
        "image_path": None,
        "suggestion_picks": None, # random picks stable per session
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

_init_session()

# ── Helper: intent badge HTML ─────────────────────────────────────────────────
_INTENT_ICONS = {
    "LORE": ("📖", "badge-lore"),
    "RECOMMEND": ("⭐", "badge-recommend"),
    "TOOL": ("🔧", "badge-tool"),
    "GENERAL": ("💬", "badge-general"),
    "PERSONA_SWITCH": ("🎭", "badge-persona"),
    "EPISODE_UPDATE": ("📺", "badge-persona"),
}

def _intent_badge(intent: str) -> str:
    icon, cls = _INTENT_ICONS.get(intent, ("💬", "badge-general"))
    label = intent.replace("_", " ").title()
    return f'<span class="intent-badge {cls}">{icon} {label}</span>'

# ── Helper: start new session ────────────────────────────────────────────────
def _new_session():
    # Reset UI state entirely to a clean slate (do not create DB entry yet)
    st.session_state.session_id = None
    st.session_state.messages = []
    st.session_state.lc_messages = []
    st.session_state.last_intent = ""
    st.session_state.image_path = None

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    # Header
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0 0.5rem;">
      <span style="font-size:2.5rem;">🎌</span>
      <h2 style="font-family:'Rajdhani',sans-serif; font-size:1.4rem;
                 background:linear-gradient(90deg,#a855f7,#3b82f6);
                 -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                 margin:0.2rem 0 0;">Anime RAG Bot</h2>
      <p style="color:#475569; font-size:0.75rem; margin:0;">Agentic • RAG • Spoiler-Safe</p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── Anime Settings ─────────────────────────────────────────────────────
    st.markdown('<div class="sidebar-title">📺 Anime Settings</div>', unsafe_allow_html=True)

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
        if st.button("🛡️ Set", use_container_width=True, key="set_ep") and ep_input > 0:
            if st.session_state.anime_name:
                # Inject episode update as a user message
                st.session_state._pending_episode_msg = (
                    f"I'm on episode {ep_input} of {st.session_state.anime_name}"
                )
            else:
                st.warning("Select an anime first")

    spoiler_mode = st.toggle(
        "🔓 Spoiler Mode (see everything)",
        value=st.session_state.spoiler_mode,
        key="spoiler_toggle",
    )
    st.session_state.spoiler_mode = spoiler_mode

    st.divider()

    # ── Persona ────────────────────────────────────────────────────────────
    st.markdown('<div class="sidebar-title">🎭 Persona</div>', unsafe_allow_html=True)

    db_chars = get_character_db()
    char_names = ["Default"] + sorted(
        c.get("name", "") for c in db_chars.values()
        if c.get("name")
    )[:200]  # cap at 200 for dropdown performance

    persona_idx = 0
    if st.session_state.persona in char_names:
        persona_idx = char_names.index(st.session_state.persona)

    selected_persona = st.selectbox(
        "Bot Persona",
        options=char_names,
        index=persona_idx,
        key="persona_select",
        help="Choose which character the bot should speak as",
    )
    st.session_state.persona = selected_persona

    if selected_persona != "Default":
        char = db_chars.get(selected_persona.lower())
        if char:
            traits = char.get("personality", {}).get("traits", [])
            if traits:
                st.caption(f"💡 {', '.join(traits[:3])}")

    st.divider()

    # ── Screenshot Upload ──────────────────────────────────────────────────
    st.markdown('<div class="sidebar-title">🖼️ Screenshot ID</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload an anime screenshot",
        type=["jpg", "jpeg", "png", "webp"],
        key="screenshot_upload",
        label_visibility="collapsed",
    )
    if uploaded_file:
        # Save to temp file for trace.moe
        suffix = Path(uploaded_file.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.read())
            st.session_state.image_path = tmp.name
        st.image(uploaded_file, caption="Uploaded screenshot", use_column_width=True)
        if st.button("🔍 Identify this screenshot", use_container_width=True):
            st.session_state._pending_screenshot_msg = (
                "What anime is this screenshot from?"
            )
    else:
        st.session_state.image_path = None

    st.divider()

    # ── Chat History ───────────────────────────────────────────────────────
    st.markdown('<div class="sidebar-title">📜 Chat History</div>', unsafe_allow_html=True)

    if st.button("➕ New Chat", use_container_width=True, type="primary"):
        _new_session()
        st.rerun()

    try:
        db = get_db()
        sessions = db.list_sessions()

        # Filter to only this browser session's conversations (privacy fix)
        my_ids = set(st.session_state.my_session_ids)
        sessions = [s for s in sessions if s["session_id"] in my_ids]

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
                    if st.button("🗑️", key=f"del_{sess['session_id']}", help="Delete chat", use_container_width=False):
                        db.delete_session(sess["session_id"])
                        if is_current:
                            st.session_state.session_id = None
                            st.session_state.messages = []
                            st.session_state.lc_messages = []
                        st.rerun()
    except Exception as _db_err:
        st.caption(f"⚠️ Chat history unavailable: DB connection error.")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN AREA — Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="anime-header">
  <span style="font-size:2.5rem;">🌸</span>
  <div>
    <h1>Anime Agentic RAG</h1>
    <p>Spoiler-safe lore • Smart recommendations • Real-time tools • 738 character personas</p>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Active status bar ─────────────────────────────────────────────────────────
status_parts = []
if st.session_state.anime_name:
    status_parts.append(f"📺 {st.session_state.anime_name}")
if st.session_state.persona != "Default":
    status_parts.append(f"🎭 {st.session_state.persona}")
if st.session_state.spoiler_mode:
    status_parts.append("🔓 Spoiler Mode ON")
elif st.session_state.current_chapter < 9999:
    status_parts.append(f"🛡️ Chapter cap: {st.session_state.current_chapter}")

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
    with st.chat_message(msg["role"], avatar="🙋" if msg["role"] == "user" else "🤖"):
        if msg["role"] == "assistant" and msg.get("intent"):
            st.markdown(_intent_badge(msg["intent"]), unsafe_allow_html=True)
        if msg["role"] == "user":
            st.markdown(
                f'<div class="user-bubble">{msg["content"]}</div>',
                unsafe_allow_html=True,
            )
        else:
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

# Pool — one question randomly picked per category each session
_LORE_POOL = [
    "What happens to Gojo in Shibuya?",
    "What is Gojo Satoru's Six Eyes and Infinity technique?",
    "What happens to Eren Yeager at the end of Attack on Titan?",
    "How does Tanjiro unlock the Sun Breathing style?",
    "Who killed Itachi Uchiha and why?",
    "What is the One Piece and who is Joy Boy?",
    "Explain the Nen system from Hunter x Hunter",
    "What is Chainsaw Man's contract devil power?",
]
_REC_POOL = [
    "Suggest anime like Attack on Titan",
    "Recommend dark psychological anime like Death Note",
    "What should I watch after Fullmetal Alchemist Brotherhood?",
    "Suggest anime similar to Demon Slayer",
    "Recommend anime with strong female leads",
    "Best anime for someone new to the genre?",
]
_TOOL_POOL = [
    "When does JJK next episode air?",
    "When does the next episode of One Piece air?",
    "When does Chainsaw Man next air?",
    "What time does My Hero Academia broadcast?",
]

if not st.session_state.messages:
    # Pick one from each category per session, stable across widget reruns
    if st.session_state.suggestion_picks is None:
        st.session_state.suggestion_picks = [
            _random.choice(_LORE_POOL),
            _random.choice(_REC_POOL),
            _random.choice(_TOOL_POOL),
        ]

    lore_q, rec_q, tool_q = st.session_state.suggestion_picks

    # Original welcome screen HTML (design unchanged)
    st.markdown(f"""
      <div style="text-align:center; padding: 2rem 1rem 0.5rem 1rem; color:#e8e8f0;">
        <div style="font-size:2rem; margin-bottom:0.8rem;">🌸🗡️✨</div>
        <h3 style="font-size:1.6rem; font-weight:700; margin-bottom:0.6rem;
                   background:linear-gradient(90deg,#a855f7,#60a5fa); -webkit-background-clip:text;
                   -webkit-text-fill-color:transparent;">
          Ready to explore the anime world
        </h3>
        <p style="font-size:0.9rem; max-width:500px; margin:0 auto; line-height:1.7; color:#94a3b8;">
          Ask about <b style="color:#a855f7">plot &amp; lore</b>,
          get <b style="color:#34d399">recommendations</b>,
          check <b style="color:#fbbf24">airing schedules</b>,
          or identify a <b style="color:#60a5fa">screenshot</b>.<br><br>
          Set your episode in the sidebar for <b style="color:#f472b6">spoiler-safe</b> answers.
        </p>
      </div>
    """, unsafe_allow_html=True)

    # Clickable pill buttons — styled to match original spans exactly
    st.markdown("""
    <style>
      /* Pill buttons — scoped by key prefix to avoid touching sidebar/chat buttons */
      div[data-testid="stHorizontalBlock"] button[kind="secondary"] {
        border-radius: 8px !important;
        font-size: 0.82rem !important;
        padding: 0.4rem 0.8rem !important;
        font-weight: 400 !important;
        transition: opacity 0.2s !important;
      }
      div[data-testid="stHorizontalBlock"] button[kind="secondary"]:hover {
        opacity: 0.85 !important;
      }
    </style>
    """, unsafe_allow_html=True)

    lore_col, rec_col, tool_col = st.columns(3)
    with lore_col:
        st.markdown("""<style>
          div[data-testid="stHorizontalBlock"] > div:nth-child(1) button {
            background: rgba(168,85,247,0.1) !important;
            border: 1px solid rgba(168,85,247,0.3) !important;
            color: #c084fc !important;
          }
        </style>""", unsafe_allow_html=True)
        if st.button(f"📖 {lore_q}", key="pill_lore", use_container_width=True):
            st.session_state._pending_msg = lore_q
            st.rerun()

    with rec_col:
        st.markdown("""<style>
          div[data-testid="stHorizontalBlock"] > div:nth-child(2) button {
            background: rgba(16,185,129,0.1) !important;
            border: 1px solid rgba(16,185,129,0.3) !important;
            color: #34d399 !important;
          }
        </style>""", unsafe_allow_html=True)
        if st.button(f"⭐ {rec_q}", key="pill_rec", use_container_width=True):
            st.session_state._pending_msg = rec_q
            st.rerun()

    with tool_col:
        st.markdown("""<style>
          div[data-testid="stHorizontalBlock"] > div:nth-child(3) button {
            background: rgba(245,158,11,0.1) !important;
            border: 1px solid rgba(245,158,11,0.3) !important;
            color: #fbbf24 !important;
          }
        </style>""", unsafe_allow_html=True)
        if st.button(f"📅 {tool_q}", key="pill_tool", use_container_width=True):
            st.session_state._pending_msg = tool_q
            st.rerun()


pending = None
if "_pending_msg" in st.session_state:
    pending = st.session_state.pop("_pending_msg")
elif hasattr(st.session_state, "_pending_episode_msg"):
    pending = st.session_state._pending_episode_msg
    del st.session_state._pending_episode_msg
elif hasattr(st.session_state, "_pending_screenshot_msg"):
    pending = st.session_state._pending_screenshot_msg
    del st.session_state._pending_screenshot_msg

# ─────────────────────────────────────────────────────────────────────────────
# CHAT INPUT
# ─────────────────────────────────────────────────────────────────────────────
user_input = st.chat_input(
    placeholder="Ask about lore, get recommendations, check schedules... or just chat!",
    key="chat_input",
)

# Use pending message if no direct input
if pending and not user_input:
    user_input = pending

if user_input:
    # ── Ensure DB session exists before saving ────────────────────────────
    if not st.session_state.session_id:
        db = get_db()
        new_sid = db.create_session(
            anime_name=st.session_state.anime_name,
            persona=st.session_state.persona,
        )
        st.session_state.session_id = new_sid
        # Track this session ID as belonging to this browser user
        st.session_state.my_session_ids.append(new_sid)

    # ── Display user message ──────────────────────────────────────────────
    st.session_state.messages.append({
        "role": "user",
        "content": user_input,
        "intent": "",
    })
    with st.chat_message("user", avatar="🙋"):
        st.markdown(
            f'<div class="user-bubble">{user_input}</div>',
            unsafe_allow_html=True,
        )

    # ── Status indicator while agent runs ────────────────────────────────
    with st.chat_message("assistant", avatar="🤖"):
        response_placeholder = st.empty()
        
        # ── Run agent stream ──────────────────────────────────────────────
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
                        # Display what the agent is currently doing with descriptive text
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
                        
                        status_container.write(f"🔄 **{node_desc}**")
                        
                        if node == "tools_node":
                            context = event["update"].get("retrieved_context", "")
                            tools_called = [line.strip("[]:") for line in context.split("\n") if line.startswith("[") and line.endswith("]:")]
                            if tools_called:
                                status_container.write(f"🔧 **Tools used:** `{', '.join(tools_called)}`")
                        
                    elif event["type"] == "final":
                        result = event["result"]
                        
                status_container.update(label="Response generated", state="complete", expanded=False)

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
                "image": chart_path
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
            error_msg = f"⚠️ An error occurred: {e}"
            response_placeholder.error(error_msg)
            clean_reply = error_msg
            intent = "GENERAL"
            st.session_state.messages.append({
                "role": "assistant",
                "content": clean_reply,
                "intent": intent
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

