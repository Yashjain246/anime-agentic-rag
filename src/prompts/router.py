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
Classify this user message into exactly ONE category.

Categories:
LORE       - Questions about plot, story events, characters, lore from manga/anime
             Examples: "What happens to Gojo?", "Who is Muzan?", "Explain the Rumbling"
RECOMMEND  - Requests for anime recommendations or suggestions
             Examples: "Suggest anime like AOT", "What should I watch next?"
TOOL       - Requests needing real-time data: schedules, news, ratings, screenshots, calendar
             Examples: "When does JJK air?", "Show me AoT ratings", "What is this screenshot?"
GENERAL    - Greetings, opinions, general anime chat not needing a database
             Examples: "Hi!", "Who is your favourite character?", "Is Chainsaw Man good?"

User message: "{user_message}"

Reply with ONLY the category word: LORE, RECOMMEND, TOOL, or GENERAL"""


def build_classification_prompt(user_message: str) -> str:
    """Format the router classification prompt with the user's message."""
    return ROUTER_CLASSIFICATION_TEMPLATE.format(user_message=user_message)
