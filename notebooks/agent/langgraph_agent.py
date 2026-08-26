"""
agent/langgraph_agent.py
==========================
Source notebook: phase4_langgraph_updated (1).ipynb

STATUS: intentionally not reimplemented here. src/agent/ is already a
complete, current implementation of the same design — same 7-node graph
(persona_node, episode_node, router_node, lore_node, recs_node,
tools_node, respond_node), same AgentState shape, same persona-switch and
episode-to-chapter-mapping engines.

THE ACTUAL WORKING VERSION:

    src/agent/state.py    — AgentState TypedDict
    src/agent/nodes.py    — all 7 node functions
    src/agent/graph.py    — graph wiring + compilation
    src/agent/runner.py   — run_agent_with_state(), the public entry point
    src/persona/          — dynamic persona engine (738 characters)
    src/episode/          — episode-to-chapter mapping engine

If the original notebook code ever does need to be ported, diff it
against src/agent/nodes.py first — src/ is very likely still ahead of it.
"""
