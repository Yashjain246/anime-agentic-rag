"""
src/agent/nodes.py
──────────────────
All 7 LangGraph node functions.

Node execution order:
  persona_node → episode_node → router_node
      ↓               ↓              ↓
  (PERSONA_SWITCH)  (EPISODE_UPDATE) LORE → lore_node → respond_node
  END              END              RECOMMEND → recs_node → respond_node
                                    TOOL → tools_node → respond_node
                                    GENERAL → respond_node
                                                   ↓
                                                  END
"""

from __future__ import annotations

import logging
from pydantic import ValidationError

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.prebuilt import ToolNode

from config.settings import settings
from src.agent.state import AgentState
from src.episode.detector import detect_episode_progress
from src.episode.mapping import episode_to_chapter
from src.llm.clients import get_agent_llm, get_query_gen_llm
from src.persona.builder import get_persona_prompt
from src.persona.detector import detect_persona_switch
from src.prompts.respond import build_system_prompt
from src.prompts.router import RouterOutput, build_classification_prompt
from src.rag.retriever import build_retriever
from src.rag.vectorstores import get_recs_vectorstore
from src.tools.registry import TOOLS

logger = logging.getLogger(__name__)


def _get_last_human_message(state: AgentState) -> str:
    """Extract the most recent HumanMessage content from state."""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            return msg.content
    return ""


# ── NODE 0: Persona Switch ────────────────────────────────────────────────────
def persona_node(state: AgentState) -> dict:
    """
    Checks the latest user message for a persona-switch request
    ("talk to me like X", "be Y", "go back to normal", etc.).

    If detected:
      - Updates state['persona']
      - Returns a short in-character confirmation
      - Sets intent = 'PERSONA_SWITCH' → graph skips to END

    If NOT detected:
      - Returns {} → graph proceeds to episode_node
    """
    user_message = _get_last_human_message(state)
    new_persona = detect_persona_switch(user_message)

    if new_persona is None:
        return {}

    logger.info(f"[Persona] Switching persona → {new_persona}")
    persona_prompt = get_persona_prompt(new_persona)

    if new_persona == "Default":
        confirm_system = (
            f"{persona_prompt}\n\n"
            "The user asked you to stop roleplaying and go back to normal. "
            "Briefly acknowledge this in 1 short sentence."
        )
    else:
        confirm_system = (
            f"{persona_prompt}\n\n"
            "The user just asked you to start talking like this character. "
            "Respond with a SHORT (1-2 sentence) in-character greeting that "
            "shows off this character's voice and confirms the switch. "
            "Do not explain that you are an AI adopting a persona."
        )

    response = get_agent_llm().invoke([
        SystemMessage(content=confirm_system),
        HumanMessage(content=user_message),
    ])
    return {
        "persona": new_persona,
        "intent": "PERSONA_SWITCH",
        "messages": [AIMessage(content=response.content)],
    }


# ── NODE 0b: Episode Progress ─────────────────────────────────────────────────
def episode_node(state: AgentState) -> dict:
    """
    Checks for "I'm on episode N of <anime>" statements.

    If detected:
      - Converts episode → chapter via episode_to_chapter()
      - Updates current_chapter, anime_name, spoiler_mode = False
      - Returns a confirmation message + sets intent = 'EPISODE_UPDATE'

    If NOT detected:
      - Returns {} → graph proceeds to router_node
    """
    user_message = _get_last_human_message(state)
    detected = detect_episode_progress(user_message)

    if detected is None:
        return {}

    episode_number, detected_anime = detected
    anime_name = detected_anime or state.get("anime_name") or None
    persona_prompt = get_persona_prompt(state.get("persona", "Default"))

    if anime_name is None:
        confirm_system = (
            f"{persona_prompt}\n\n"
            "The user told you what episode they are on, but did not say "
            "which anime, and no anime is currently selected. "
            'Ask them (briefly, in character) which anime they mean — '
            'e.g. "Episode 45 of which anime?"'
        )
        response = get_agent_llm().invoke([
            SystemMessage(content=confirm_system),
            HumanMessage(content=user_message),
        ])
        logger.info(f"[Episode] No anime context for episode {episode_number} — asking user")
        return {
            "intent": "EPISODE_UPDATE",
            "messages": [AIMessage(content=response.content)],
        }

    chapter_cap = episode_to_chapter(anime_name, episode_number)

    if chapter_cap is None:
        confirm_system = (
            f"{persona_prompt}\n\n"
            f"The user said they are on episode {episode_number} of "
            f"{anime_name}, but you don't have an episode-to-chapter "
            f"mapping for that anime. Briefly let them know (in character) "
            f"that you can't set a spoiler cap for that title, but you can "
            f"still chat about it normally."
        )
        response = get_agent_llm().invoke([
            SystemMessage(content=confirm_system),
            HumanMessage(content=user_message),
        ])
        logger.info(f"[Episode] No mapping for anime={anime_name}")
        return {
            "intent": "EPISODE_UPDATE",
            "messages": [AIMessage(content=response.content)],
        }

    logger.info(
        f"[Episode] {anime_name} episode {episode_number} → "
        f"chapter cap {chapter_cap}, spoiler_mode=False"
    )
    confirm_system = (
        f"{persona_prompt}\n\n"
        f"The user just told you they are caught up to episode "
        f"{episode_number} of {anime_name}. You have set their spoiler "
        f"cap to chapter {chapter_cap} — anything beyond that chapter "
        f"will be hidden from them. Briefly (1-2 sentences, in character) "
        f"confirm this and let them know they're safe from spoilers "
        f"beyond that point."
    )
    response = get_agent_llm().invoke([
        SystemMessage(content=confirm_system),
        HumanMessage(content=user_message),
    ])
    return {
        "anime_name": anime_name,
        "current_chapter": chapter_cap,
        "spoiler_mode": False,
        "intent": "EPISODE_UPDATE",
        "messages": [AIMessage(content=response.content)],
    }


# ── NODE 1: Router ────────────────────────────────────────────────────────────
def router_node(state: AgentState) -> dict:
    """
    Classifies user intent using Pydantic-enforced structured output.

    Uses RouterOutput(intent: Literal['LORE','RECOMMEND','TOOL','GENERAL'])
    via .with_structured_output() — guarantees only the 4 allowed values
    are returned. Falls back to GENERAL on ValidationError.
    """
    user_message = _get_last_human_message(state)
    prompt = build_classification_prompt(user_message)

    try:
        structured_llm = get_query_gen_llm().with_structured_output(RouterOutput)
        result: RouterOutput = structured_llm.invoke(prompt)
        intent = result.intent
    except (ValidationError, Exception) as e:
        logger.warning(f"[Router] Structured output failed ({e}), defaulting to GENERAL")
        intent = "GENERAL"

    logger.info(f"[Router] Intent: {intent}")
    print(f"[Router] Intent classified as: {intent}")
    return {"intent": intent}


# ── NODE 2: Lore ──────────────────────────────────────────────────────────────
def lore_node(state: AgentState) -> dict:
    """
    Retrieves relevant manga chapters from the Lore DB.
    Applies the spoiler firewall based on current_chapter and spoiler_mode.
    """
    user_message = _get_last_human_message(state)

    if state.get("spoiler_mode", False):
        print("[Lore] Spoiler mode ON: Searching all chapters...")
        retriever = build_retriever(anime_name=state.get("anime_name") or None)
    else:
        cap = state.get("current_chapter", 9999)
        print(f"[Lore] Searching chapters (Spoiler cap: {cap})...")
        retriever = build_retriever(
            anime_name=state.get("anime_name") or None,
            max_chapter=cap,
        )

    docs = retriever.invoke(user_message)

    if not docs:
        if not state.get("spoiler_mode", False):
            context = (
                "NO_CONTEXT_FOUND: The event the user is asking about "
                "appears to happen after their current chapter. "
                "Inform them politely without revealing spoilers."
            )
        else:
            context = "NO_CONTEXT_FOUND: No relevant chapters found for this query."
    else:
        context_parts = []
        for doc in docs:
            meta = doc.metadata
            context_parts.append(
                f"[{meta['anime_name']} — Chapter {meta['chapter_number']}]\n"
                f"{doc.page_content}"
            )
        context = "\n\n---\n\n".join(context_parts)

    logger.info(f"[Lore] Retrieved {len(docs)} chunks")
    print(f"[Lore] Retrieved {len(docs)} relevant chapter segments.")
    return {"retrieved_context": context}


# ── NODE 3: Recommendations ───────────────────────────────────────────────────
def recs_node(state: AgentState) -> dict:
    """
    Retrieves anime recommendations from the Recs DB (500 anime synopses).
    Simple semantic similarity search — no spoiler filter needed.
    """
    user_message = _get_last_human_message(state)
    print(f"[Recs] Searching anime synopses for top {settings.RECS_TOP_K} matches...")
    recs_retriever = get_recs_vectorstore().as_retriever(
        search_type="similarity",
        search_kwargs={"k": settings.RECS_TOP_K},
    )
    docs = recs_retriever.invoke(user_message)

    if not docs:
        return {"retrieved_context": "NO_RECS_FOUND"}

    context_parts = []
    for doc in docs:
        context_parts.append(
            f"Title: {doc.metadata.get('title', 'Unknown')}\n"
            f"Genres: {doc.metadata.get('genres', 'Unknown')}\n"
            f"Score: {doc.metadata.get('score', 'N/A')}\n"
            f"Synopsis: {doc.page_content[:300]}..."
        )
    context = "\n\n---\n\n".join(context_parts)

    logger.info(f"[Recs] Found {len(docs)} recommendations")
    print(f"[Recs] Found {len(docs)} recommendations.")
    return {"retrieved_context": context}


# ── NODE 4: Tools ─────────────────────────────────────────────────────────────
def tools_node(state: AgentState) -> dict:
    """
    Handles tool requests using an iterative tool-calling loop.

    Supports multi-step tool chains (e.g. anilist_schedule → google_calendar_add).
    Loop runs up to MAX_TOOL_ITERATIONS times — the `for/else` guard ensures
    we log a warning and proceed gracefully instead of looping forever.
    """
    llm_with_tools = get_agent_llm().bind_tools(TOOLS)
    tool_executor = ToolNode(TOOLS)

    user_message = _get_last_human_message(state)
    if state.get("image_path"):
        user_message = f'{user_message} [image:{state["image_path"]}]'

    system = SystemMessage(content=(
        "You are an anime assistant with access to tools. "
        "Use the appropriate tool to answer the user. "
        "For calendar requests, ALWAYS call anilist_schedule first "
        "to get the broadcast day and time, then call google_calendar_add. "
        "When returning the calendar link to the user, tell them to click the link to save it to their personal calendar. "
        "If the user uploaded an image (indicated by [image:path]), "
        "call trace_moe_vision with that path."
    ))

    messages = [system, HumanMessage(content=user_message)]
    tool_results: list[str] = []

    for step in range(settings.MAX_TOOL_ITERATIONS):
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            # LLM is done — no more tools needed
            break

        tool_result_state = tool_executor.invoke({"messages": messages})
        messages.extend(tool_result_state["messages"])

        for msg in tool_result_state["messages"]:
            if isinstance(msg, ToolMessage):
                tool_results.append(f"[{msg.name}]:\n{msg.content}")

        called = [tc["name"] for tc in response.tool_calls]
        logger.info(f"[Tools] Step {step + 1}/{settings.MAX_TOOL_ITERATIONS}: called {called}")
        print(f"[Tools] LLM requested tools: {called}")
    else:
        # Loop exhausted max iterations
        logger.warning(
            f"[Tools] Hit MAX_TOOL_ITERATIONS={settings.MAX_TOOL_ITERATIONS} — "
            "proceeding with partial results"
        )

    context = "\n\n".join(tool_results) if tool_results else "No tool results."
    return {"retrieved_context": context}


# ── NODE 5: Respond ───────────────────────────────────────────────────────────
def respond_node(state: AgentState) -> dict:
    """
    Generates the final user-facing response.
    Reads retrieved_context (from lore/recs/tools node) and generates
    a response in the selected persona's voice.
    """
    user_message = _get_last_human_message(state)
    persona_text = get_persona_prompt(state.get("persona", "Default"))
    context = state.get("retrieved_context", "")
    intent = state.get("intent", "GENERAL")

    system_content, _ = build_system_prompt(intent, persona_text, context)

    response = get_agent_llm().invoke([
        SystemMessage(content=system_content),
        HumanMessage(content=user_message),
    ])

    logger.info(f"[Respond] Generated response ({len(response.content)} chars)")
    print(f"[Respond] Generated response in '{state.get('persona', 'Default')}' persona ({len(response.content)} chars).")
    return {"messages": [AIMessage(content=response.content)]}
