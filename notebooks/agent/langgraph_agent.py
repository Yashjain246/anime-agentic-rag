"""
agent/langgraph_agent.py
========================
Source notebook: phase4_langgraph_updated (1).ipynb

Full LangGraph agent that combines all pipeline components into a
conversational AI system. This is the top-level orchestration layer.

Architecture:
  ┌──────────────┐
  │  User Query  │
  └──────┬───────┘
         ▼
   inject_persona  ──► detect character from all_characters (4).jsonl
         ▼
    route_query    ──► classify: lore | recs | tools | chat
         │
   ┌─────┼──────────────────────┐
   ▼     ▼                      ▼
lore   recs                  tools
   │     │                      │
retrieve retrieve          call_tools
_lore   _recs                   │
   └─────┴──────────────────────┘
                  ▼
         generate_response  ──► persona-aware LLM reply
                  ▼
         [ append to history ]

Run standalone (interactive CLI chat):
    python -m notebooks.agent.langgraph_agent [--episode N] [--persona "Luffy"]

Sections
--------
1. Configuration
2. AgentState         (LangGraph TypedDict)
3. Node: inject_persona
4. Node: route_query
5. Node: retrieve_lore
6. Node: retrieve_recs
7. Node: call_tools
8. Node: generate_response
9. Graph construction  (StateGraph wiring + compilation)
10. AgentSession       (stateful wrapper with chat history)
11. CLI entry-point
"""

from __future__ import annotations
import argparse
from typing import TypedDict

# TODO: uncomment once dependencies confirmed
# from langgraph.graph import StateGraph, END
# from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

# =============================================================================
# 1. Configuration
# =============================================================================

CHARACTERS_JSONL = "data/all_characters (4).jsonl"  # for persona detection
LLM_MODEL        = "gemini-1.5-flash"               # TODO: verify vs notebook

# Route labels
ROUTE_LORE  = "lore"
ROUTE_RECS  = "recs"
ROUTE_TOOLS = "tools"
ROUTE_CHAT  = "chat"


# =============================================================================
# 2. AgentState
# =============================================================================

class AgentState(TypedDict):
    """
    Shared state flowing through every node in the LangGraph.

    Fields
    ------
    messages : list
        Full conversation history (HumanMessage / AIMessage / ToolMessage).
    query : str
        The latest user message (extracted for retrieval convenience).
    retrieved_chapters : list[dict]
        Chapters returned by the lore RAG pipeline.
    retrieved_recs : list[dict]
        Anime synopses returned by the recommendations retriever.
    user_episode : int | None
        User's current episode — enables the spoiler firewall when set.
    persona_name : str | None
        Active character persona name.
    persona_style : str
        Speaking-style instructions extracted from all_characters (4).jsonl.
    route : str
        Routing decision: one of ROUTE_* constants above.

    TODO: Adjust to match the exact fields in phase4_langgraph_updated (1).ipynb.
    """
    messages:           list
    query:              str
    retrieved_chapters: list
    retrieved_recs:     list
    user_episode:       int | None
    persona_name:       str | None
    persona_style:      str
    route:              str


# =============================================================================
# 3. Node: inject_persona
# =============================================================================

def inject_persona(state: AgentState) -> dict:
    """
    Detect the requested character persona from the user's latest message
    and populate ``persona_name`` and ``persona_style`` in the state.

    Reads speaking styles from all_characters (4).jsonl.

    TODO: Replace with actual persona-detection code from the notebook.
    """
    raise NotImplementedError


# =============================================================================
# 4. Node: route_query
# =============================================================================

def route_query(state: AgentState) -> dict:
    """
    Classify the user's query and set ``state["route"]`` to one of:
      - ROUTE_LORE   → manga chapter RAG
      - ROUTE_RECS   → anime recommendations
      - ROUTE_TOOLS  → external tool call (trace.moe, OMDB, …)
      - ROUTE_CHAT   → general conversation

    TODO: Replace with actual LLM-based routing code from the notebook.
    """
    raise NotImplementedError


# =============================================================================
# 5. Node: retrieve_lore
# =============================================================================

def retrieve_lore(state: AgentState) -> dict:
    """
    Run the HybridSearchPipeline (retrieval/hybrid_search.py) and store
    results in ``state["retrieved_chapters"]``.

    Applies the spoiler firewall if ``state["user_episode"]`` is set.

    TODO: Replace with actual retrieval node from the notebook.
    """
    raise NotImplementedError


# =============================================================================
# 6. Node: retrieve_recs
# =============================================================================

def retrieve_recs(state: AgentState) -> dict:
    """
    Query the chroma_recs_db for anime similar to the user's request
    and store results in ``state["retrieved_recs"]``.

    TODO: Replace with actual recs-retrieval node from the notebook.
    """
    raise NotImplementedError


# =============================================================================
# 7. Node: call_tools
# =============================================================================

def call_tools(state: AgentState) -> dict:
    """
    Execute any LangChain tool calls requested by the LLM in the
    previous turn and append ToolMessages to ``state["messages"]``.

    Binds the TOOLS list from integrations/external_tools.py.

    TODO: Replace with actual tool-execution code from the notebook.
    """
    raise NotImplementedError


# =============================================================================
# 8. Node: generate_response
# =============================================================================

def generate_response(state: AgentState) -> dict:
    """
    Build the final LLM prompt — including retrieved context, tool
    results, persona style, and chat history — and append the AI reply
    to ``state["messages"]``.

    TODO: Replace with actual response-generation node from the notebook.
    """
    raise NotImplementedError


# =============================================================================
# 9. Graph Construction
# =============================================================================

def build_graph():
    """
    Wire all nodes into a LangGraph StateGraph and compile it.

    Returns
    -------
    CompiledGraph
        Ready to invoke via ``graph.invoke(state)``.

    Routing edges (fill in exact conditions from the notebook):
        inject_persona → route_query
        route_query    → [lore | recs | tools | chat]  (conditional)
        retrieve_lore  → generate_response
        retrieve_recs  → generate_response
        call_tools     → generate_response
        generate_response → END

    TODO: Replace with actual graph wiring from
          phase4_langgraph_updated (1).ipynb.
    """
    raise NotImplementedError


# =============================================================================
# 10. AgentSession  (stateful chat wrapper)
# =============================================================================

class AgentSession:
    """
    Manages a single user conversation with the compiled LangGraph agent.

    Parameters
    ----------
    user_episode : int | None
        Latest anime episode watched (enables spoiler firewall).
    persona_name : str | None
        Default character persona.

    TODO: Replace __init__, chat, and reset with actual session code from
          phase4_langgraph_updated (1).ipynb.
    """

    def __init__(
        self,
        user_episode: int | None = None,
        persona_name: str | None = None,
    ) -> None:
        raise NotImplementedError

    def chat(self, user_message: str) -> str:
        """
        Send *user_message* to the agent and return the AI reply string.

        TODO: Replace with actual chat-invocation code from the notebook.
        """
        raise NotImplementedError

    def reset(self) -> None:
        """Clear conversation history and start a fresh session."""
        raise NotImplementedError


# =============================================================================
# 11. CLI Entry-Point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Interactive CLI chat with the anime RAG agent."
    )
    parser.add_argument("--episode", type=int, default=None,
                        help="Your latest anime episode (enables spoiler firewall)")
    parser.add_argument("--persona", type=str, default=None,
                        help='Character persona, e.g. "Luffy" or "Nami"')
    args = parser.parse_args()

    session = AgentSession(user_episode=args.episode, persona_name=args.persona)
    print("Anime RAG Agent ready. Type 'quit' to exit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if user_input.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break
        if not user_input:
            continue
        reply = session.chat(user_input)
        print(f"\nAgent: {reply}\n")
