"""
src/agent/graph.py
──────────────────
LangGraph StateGraph wiring and compilation.

Graph structure:
  persona_node ──(PERSONA_SWITCH?)──► END
       │ NO
       ▼
  episode_node ──(EPISODE_UPDATE?)──► END
       │ NO
       ▼
  router_node
       │
       ├──LORE──────► lore_node ──► respond_node ──► END
       ├──RECOMMEND──► recs_node ──► respond_node ──► END
       ├──TOOL──────► tools_node ──► respond_node ──► END
       └──GENERAL───────────────► respond_node ──► END
"""

from __future__ import annotations

from typing import Literal

from langgraph.graph import StateGraph, END

from src.agent.state import AgentState
from src.agent.nodes import (
    persona_node,
    episode_node,
    router_node,
    lore_node,
    recs_node,
    tools_node,
    respond_node,
)


# ── Conditional edge functions ────────────────────────────────────────────────

def _route_after_persona(
    state: AgentState,
) -> Literal["episode_node", "__end__"]:
    """After persona_node: go to END if it handled the request, else episode_node."""
    if state.get("intent") == "PERSONA_SWITCH":
        return "__end__"
    return "episode_node"


def _route_after_episode(
    state: AgentState,
) -> Literal["router_node", "__end__"]:
    """After episode_node: go to END if it handled the request, else router_node."""
    if state.get("intent") == "EPISODE_UPDATE":
        return "__end__"
    return "router_node"


def _route_after_router(
    state: AgentState,
) -> Literal["lore_node", "recs_node", "tools_node", "respond_node"]:
    """After router_node: dispatch to the correct retrieval/tool node."""
    intent = state.get("intent", "GENERAL")
    if intent == "LORE":
        return "lore_node"
    if intent == "RECOMMEND":
        return "recs_node"
    if intent == "TOOL":
        return "tools_node"
    return "respond_node"  # GENERAL goes straight to respond


# ── Build and compile the graph ───────────────────────────────────────────────

def build_graph() -> StateGraph:
    """Build the LangGraph StateGraph. Call compile() on the result."""
    g = StateGraph(AgentState)

    # Register all nodes
    g.add_node("persona_node", persona_node)
    g.add_node("episode_node", episode_node)
    g.add_node("router_node", router_node)
    g.add_node("lore_node", lore_node)
    g.add_node("recs_node", recs_node)
    g.add_node("tools_node", tools_node)
    g.add_node("respond_node", respond_node)

    # Entry point
    g.set_entry_point("persona_node")

    # persona_node → episode_node OR END
    g.add_conditional_edges(
        "persona_node",
        _route_after_persona,
        {"episode_node": "episode_node", "__end__": END},
    )

    # episode_node → router_node OR END
    g.add_conditional_edges(
        "episode_node",
        _route_after_episode,
        {"router_node": "router_node", "__end__": END},
    )

    # router_node → retrieval/tool nodes
    g.add_conditional_edges(
        "router_node",
        _route_after_router,
        {
            "lore_node": "lore_node",
            "recs_node": "recs_node",
            "tools_node": "tools_node",
            "respond_node": "respond_node",
        },
    )

    # All retrieval/tool nodes → respond_node
    g.add_edge("lore_node", "respond_node")
    g.add_edge("recs_node", "respond_node")
    g.add_edge("tools_node", "respond_node")

    # respond_node → END
    g.add_edge("respond_node", END)

    return g


# ── Singleton compiled agent ──────────────────────────────────────────────────
_agent = None


def get_agent():
    """
    Returns the compiled LangGraph agent (singleton).
    The graph is compiled once and reused for all requests.
    """
    global _agent
    if _agent is None:
        _agent = build_graph().compile()
    return _agent
