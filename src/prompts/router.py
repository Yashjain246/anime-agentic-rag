"""
src/prompts/router.py
─────────────────────
Router classification prompt + Pydantic structured output.

IMPROVEMENT over original notebook:
  Original: response.content.strip().upper() → manual validation with
            a plain `if intent not in [...]` fallback.
  New:      query_gen_llm.with_structured_output(RouterOutput) →
            Pydantic enforces the Literal type. If the LLM returns
            anything other than the 4 allowed values, a ValidationError
            is raised and caught, falling back to GENERAL — no silent
            misclassification possible.

MULTI-LABEL: intents is a list, not a single value. A single message can
genuinely ask for two different things at once ("How did Muzan die AND
when's the next Mushoku Tensei episode?") — one LORE, one TOOL. Forcing
that into one label meant the loser never got answered from its actual
source: a compound LORE+TOOL question used to come back TOOL-only, and
the LORE half got answered from the model's general knowledge (or
whatever tool happened to run) instead of the real Lore RAG. See
_route_after_router in src/agent/graph.py for how the list fans out to
more than one retrieval/tool node in the same turn.

PER-INTENT QUERY: each item also carries its own focused, standalone
`query` — just the slice of the message relevant to that one intent,
not the whole message. Without this, every node received the full raw
message and a name/topic from one part could contaminate another: a
message like "What happens in One Piece and when's the next JJK
episode?" made lore_node's own "is a supported anime mentioned?" guard
see "JJK" from the TOOL half and wrongly conclude the message DID name
a supported anime — which skipped the unsupported-anime refusal and let
it silently retrieve and confidently answer with a completely unrelated
anime's lore instead of declining. Each node uses its own `query` for
retrieval/tool-calling instead of the raw message specifically to
prevent this kind of cross-talk between parts of a compound message.
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Literal


class RouterIntentItem(BaseModel):
    """One classified part of the user's message."""
    intent: Literal["LORE", "RECOMMEND", "TOOL", "GENERAL"]
    query: str = Field(
        description="Just the part of the user's message relevant to "
        "this intent, rewritten as a standalone question/request if "
        "needed so it makes sense on its own. Must NOT include names or "
        "topics that belong only to a different part of the message."
    )


class RouterOutput(BaseModel):
    """Strictly typed router output — one or more classified parts,
    in the order the user asked for them."""
    items: list[RouterIntentItem] = Field(min_length=1)


ROUTER_CLASSIFICATION_TEMPLATE = """\
Classify this user message. List EVERY category below that applies, in
the order the user asked — most messages need only one, but don't force
two genuinely distinct asks into a single label just because the message
is one sentence. For EACH one, also give its own focused `query`: just
that part of the message, standing alone, with no name or topic that
actually belongs to a different part — EXCEPT when the same
show/character is genuinely the subject of more than one part (e.g.
"recommend anime like Frieren and who is Frieren as a character?" —
name it in both queries, don't drop it to a pronoun in the second one
assuming the reader remembers the first).

If two DIFFERENT things in the message both need the same category
(e.g. two separate TOOL asks — a schedule AND a ratings request), do
NOT list that category twice. List it once and combine both asks into
one query for that single entry.

LORE - asking what happens in the story: plot events, character actions,
  abilities, deaths, relationships. Anchored in the source material itself.
  "What happens to Gojo?", "Who is Muzan?", "Explain the Rumbling",
  "How did Rengoku die?"
RECOMMEND - asking what to watch, not what happens in something specific.
  "Suggest anime like AOT", "What should I watch next?",
  "Recommend something with a strong female lead"
TOOL - needs live or external data the model can't know on its own:
  airing schedules, episode ratings, news, or identifying a screenshot.
  "When does JJK air?", "Show me AoT's episode ratings",
  "What anime is this screenshot from?"
GENERAL - everything else: greetings, opinions, comparisons, casual chat
  that doesn't need plot details or live data.
  "Hi!", "Who's your favorite character?", "Is Chainsaw Man good?"

Example of a genuinely compound message needing two labels, each with
its own scoped query:
  "Tell me how Muzan was killed and when will the next episode of
  Mushoku Tensei air?" →
    [{{"intent": "LORE", "query": "How was Muzan killed?"}},
     {{"intent": "TOOL", "query": "When will the next episode of Mushoku Tensei air?"}}]
  The first half is a plot question anchored in the source material
  (LORE), the second needs a live airing schedule (TOOL). Answering the
  LORE half from general knowledge instead of the real story context, or
  dropping it entirely in favor of only the TOOL half, is wrong — both
  must be classified, and Mushoku Tensei must NOT appear in the LORE
  query or Muzan in the TOOL query.

Another example, showing why each query must stay scoped to its own
anime — the LORE query below must NOT mention "JJK", even though the
message does, because that name belongs to the TOOL part:
  "What happens in One Piece and when's the next JJK episode?" →
    [{{"intent": "LORE", "query": "What happens in One Piece?"}},
     {{"intent": "TOOL", "query": "When's the next JJK episode?"}}]

A short reply like "yes", "sure", or "add it" is only meaningful in
context - use RECENT CONVERSATION below to see what it's replying to. If
the assistant just offered to do something tool-based (add a calendar
event, pull a schedule, generate a ratings chart) and the user is
agreeing, classify it as TOOL, not GENERAL — and its query is the thing
being agreed to, resolved from RECENT CONVERSATION (e.g. "yes, add it"
→ query: "Add <that show>'s episode to the calendar").

RECENT CONVERSATION:
{history_text}

User message: "{user_message}"
"""


def build_classification_prompt(user_message: str, history_text: str = "No previous conversation.") -> str:
    """Format the router classification prompt with the user's message and recent history."""
    return ROUTER_CLASSIFICATION_TEMPLATE.format(
        user_message=user_message,
        history_text=history_text,
    )
