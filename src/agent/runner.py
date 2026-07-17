"""
src/agent/runner.py
───────────────────
Public API for running the agent.

run_agent_with_state() is the main function — returns the full
updated state dict so callers can persist persona changes, chapter
updates, and conversation history across turns.

run_agent() is a simplified convenience wrapper that returns just
the reply string. Use it for testing and CLI usage.
"""

from __future__ import annotations

import logging

from langchain_core.messages import AIMessage, HumanMessage

from src.agent.graph import get_agent
from src.agent.state import AgentState

logger = logging.getLogger(__name__)


def run_agent_with_state(
    message: str,
    anime_name: str = "",
    current_chapter: int = 9999,
    spoiler_mode: bool = False,
    persona: str = "Default",
    image_path: str | None = None,
    history: list | None = None,
) -> dict:
    """
    Run the agent and return the full updated state.

    This is the function Streamlit calls every turn. It returns the
    updated persona, intent, and full message history so the frontend
    can persist them in session state.

    Args:
        message:         The user's current message.
        anime_name:      Which anime to filter the Lore DB to.
        current_chapter: Spoiler cap — chapters above this are blocked.
        spoiler_mode:    If True, ignore current_chapter.
        persona:         Bot personality ('Default' or any character name).
        image_path:      Path to uploaded screenshot (optional).
        history:         Previous conversation messages to carry forward.

    Returns:
        {
          'reply':           str  — the bot's text response,
          'persona':         str  — current persona after this turn,
          'intent':          str  — LORE/RECOMMEND/TOOL/GENERAL/...,
          'messages':        list — full updated conversation history,
          'current_chapter': int  — updated chapter cap (may change via episode_node),
          'anime_name':      str  — current anime name,
          'spoiler_mode':    bool — current spoiler mode,
        }
    """
    messages = list(history) if history else []
    messages.append(HumanMessage(content=message))

    initial_state = AgentState(
        messages=messages,
        intent="",
        anime_name=anime_name,
        current_chapter=current_chapter,
        spoiler_mode=spoiler_mode,
        persona=persona,
        image_path=image_path,
        retrieved_context="",
        tool_iteration=0,
    )

    result = get_agent().invoke(initial_state)

    # Extract the last AI message
    reply = "I could not generate a response. Please try again."
    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage):
            if isinstance(msg.content, list):
                text_parts = []
                for block in msg.content:
                    if isinstance(block, dict) and "text" in block:
                        text_parts.append(block["text"])
                    elif isinstance(block, str):
                        text_parts.append(block)
                reply = "\n".join(text_parts) if text_parts else str(msg.content)
            else:
                reply = str(msg.content)
            break

    return {
        "reply": reply,
        "persona": result.get("persona", persona),
        "intent": result.get("intent", "GENERAL"),
        "messages": result["messages"],
        "current_chapter": result.get("current_chapter", current_chapter),
        "anime_name": result.get("anime_name", anime_name),
        "spoiler_mode": result.get("spoiler_mode", spoiler_mode),
    }


def stream_agent_with_state(
    message: str,
    anime_name: str = "",
    current_chapter: int = 9999,
    spoiler_mode: bool = False,
    persona: str = "Default",
    image_path: str | None = None,
    history: list | None = None,
):
    """
    Generator version of run_agent_with_state.
    Yields node execution events so the frontend can display st.status.
    
    Yields:
      {"type": "node", "name": "node_name", "update": {...}}
      {"type": "final", "result": {...}}
    """
    messages = list(history) if history else []
    messages.append(HumanMessage(content=message))

    initial_state = AgentState(
        messages=messages,
        intent="",
        anime_name=anime_name,
        current_chapter=current_chapter,
        spoiler_mode=spoiler_mode,
        persona=persona,
        image_path=image_path,
        retrieved_context="",
        tool_iteration=0,
    )

    agent = get_agent()
    current_state = dict(initial_state)

    # stream_mode="updates" yields {node_name: state_delta}
    for event in agent.stream(initial_state, stream_mode="updates"):
        if not isinstance(event, dict):
            continue
            
        for node_name, state_update in event.items():
            if not state_update:
                yield {"type": "node", "name": node_name, "update": {}}
                continue
                
            # Keep our manual state tracker synced with the graph
            for k, v in state_update.items():
                if k == "messages":
                    if isinstance(v, list):
                        current_state["messages"].extend(v)
                    else:
                        current_state["messages"].append(v)
                else:
                    current_state[k] = v
                    
            yield {"type": "node", "name": node_name, "update": state_update}

    # Extract final reply
    reply = "I could not generate a response. Please try again."
    for msg in reversed(current_state["messages"]):
        if isinstance(msg, AIMessage):
            if isinstance(msg.content, list):
                text_parts = []
                for block in msg.content:
                    if isinstance(block, dict) and "text" in block:
                        text_parts.append(block["text"])
                    elif isinstance(block, str):
                        text_parts.append(block)
                reply = "\n".join(text_parts) if text_parts else str(msg.content)
            else:
                reply = str(msg.content)
            break

    yield {
        "type": "final",
        "result": {
            "reply": reply,
            "persona": current_state.get("persona", persona),
            "intent": current_state.get("intent", "GENERAL"),
            "messages": current_state["messages"],
            "current_chapter": current_state.get("current_chapter", current_chapter),
            "anime_name": current_state.get("anime_name", anime_name),
            "spoiler_mode": current_state.get("spoiler_mode", spoiler_mode),
            "retrieved_context": current_state.get("retrieved_context", ""),
        }
    }


def run_agent(
    message: str,
    anime_name: str = "",
    current_chapter: int = 9999,
    spoiler_mode: bool = False,
    persona: str = "Default",
    image_path: str | None = None,
    history: list | None = None,
) -> str:
    """
    Simplified wrapper — returns only the reply string.
    Useful for testing and CLI usage.
    """
    result = run_agent_with_state(
        message=message,
        anime_name=anime_name,
        current_chapter=current_chapter,
        spoiler_mode=spoiler_mode,
        persona=persona,
        image_path=image_path,
        history=history,
    )
    return result["reply"]
