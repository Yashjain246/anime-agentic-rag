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
"""

from __future__ import annotations

from pydantic import BaseModel
from typing import Literal


class RouterOutput(BaseModel):
    """Strictly typed router output — exactly one of 4 intent labels."""
    intent: Literal["LORE", "RECOMMEND", "TOOL", "GENERAL"]


ROUTER_CLASSIFICATION_TEMPLATE = """\
Classify this user message into exactly one category.

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

When a message could fit two categories: prefer LORE over GENERAL for
anything asking what actually happens in the story, and prefer TOOL over
LORE or GENERAL for anything needing current, real-world information.

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
