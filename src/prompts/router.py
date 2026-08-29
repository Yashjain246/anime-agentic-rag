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

MULTI-LABEL: intent is a list, not a single value. A single message can
genuinely ask for two different things at once ("How did Muzan die AND
when's the next Mushoku Tensei episode?") — one LORE, one TOOL. Forcing
that into one label meant the loser never got answered from its actual
source: a compound LORE+TOOL question used to come back TOOL-only, and
the LORE half got answered from the model's general knowledge (or
whatever tool happened to run) instead of the real Lore RAG. See
_route_after_router in src/agent/nodes.py for how the list fans out to
more than one retrieval/tool node in the same turn.
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Literal


class RouterOutput(BaseModel):
    """Strictly typed router output — one or more of 4 intent labels,
    in the order the user asked for them."""
    intents: list[Literal["LORE", "RECOMMEND", "TOOL", "GENERAL"]] = Field(min_length=1)


ROUTER_CLASSIFICATION_TEMPLATE = """\
Classify this user message. List EVERY category below that applies, in
the order the user asked — most messages need only one, but don't force
two genuinely distinct asks into a single label just because the message
is one sentence.

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

Example of a genuinely compound message needing two labels:
  "Tell me how Muzan was killed and when will the next episode of
  Mushoku Tensei air?" → ["LORE", "TOOL"] — the first half is a plot
  question anchored in the source material (LORE), the second needs a
  live airing schedule (TOOL). Answering the LORE half from general
  knowledge instead of the real story context, or dropping it entirely
  in favor of only the TOOL half, is wrong — both must be classified.

A short reply like "yes", "sure", or "add it" is only meaningful in
context - use RECENT CONVERSATION below to see what it's replying to. If
the assistant just offered to do something tool-based (add a calendar
event, pull a schedule, generate a ratings chart) and the user is
agreeing, classify it as TOOL, not GENERAL.

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
