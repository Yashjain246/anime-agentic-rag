"""
src/prompts/respond.py
──────────────────────
Per-intent system prompt builders for respond_node.

WHY different prompts per intent:
  - LORE: must stay grounded in retrieved text only
  - RECOMMEND: needs to list titles clearly with metadata
  - TOOL: relay the tool result naturally in the current persona
  - GENERAL: free to use the LLM's own knowledge
  - SPOILER_BLOCK: politely refuse without revealing anything
"""

from __future__ import annotations

from config.settings import settings

# Applied to every response regardless of intent — without this, responses
# tend toward formal report-style writing (transition-word scaffolding,
# restating the question, over-listing) rather than a natural reply.
NATURAL_TONE_GUIDELINES = (
    "\n\nHow you write: like a knowledgeable friend replying, not a report. "
    "Skip transition scaffolding such as 'Furthermore', 'Additionally', or "
    "'In conclusion'. Don't restate the question before answering it. Use "
    "bullet points only when you're actually listing multiple distinct "
    "items (e.g. recommendations) — a single fact or explanation should "
    "just be prose. Match the length to what the question actually needs; "
    "don't pad a simple answer into a longer one."
)


def build_lore_prompt(persona_text: str, context: str) -> str:
    """Grounded lore answer — only use the retrieved context."""
    return (
        f"{persona_text}\n\n"
        "Answer the user's question using ONLY the context below — it's "
        "manga chapter summaries, each tagged with its anime and chapter "
        "number. If the context only partially answers the question, "
        "answer the part it covers and say plainly what's missing rather "
        "than refusing outright. If it doesn't cover the question at all, "
        "say so honestly instead of guessing or filling in from outside "
        "knowledge.\n"
        "When something isn't covered, never phrase it as a limitation of "
        "'the context' or say things like 'the context doesn't explain' or "
        "'the provided information doesn't cover' — that's an implementation "
        "detail the user shouldn't see and it breaks the illusion that "
        "you're a fan, not a retrieval system. Phrase it as a fact about "
        "the story instead: 'that hasn't been shown in the manga/anime', "
        "'the author hasn't revealed that yet', or similar. Stay in "
        "character throughout.\n\n"
        f"CONTEXT:\n{context}"
    )


def build_spoiler_block_prompt(persona_text: str) -> str:
    """Politely refuse to reveal spoiler content."""
    return (
        f"{persona_text}\n\n"
        "The user asked about something that happens after their current "
        "chapter. Let them know, in character, that it's beyond where "
        "they've read. Don't reveal what happens, and don't hint at it "
        "indirectly (no 'let's just say it's intense' type teasers)."
    )


def build_recs_prompt(persona_text: str, context: str) -> str:
    """Anime recommendation answer with clear title/genre/score formatting."""
    return (
        f"{persona_text}\n\n"
        "Recommend anime to the user from the options below. For each one, "
        "give the title, genre, score, and a brief reason it fits what "
        "they asked for. If the user named a specific anime as their "
        "reference point (e.g. 'like Death Note'), don't recommend that "
        "same title back to them even if it appears in the options — "
        "recommend from the others. Stay in character.\n\n"
        f"AVAILABLE OPTIONS:\n{context}"
    )


def build_tool_prompt(persona_text: str, context: str) -> str:
    """Relay tool results naturally in the current persona's voice."""
    calendar_offer = ""
    has_real_schedule = (
        "[anilist_schedule]" in context.lower()
        and "status: currently airing" in context.lower()
    )
    if settings.ENABLE_CALENDAR_TOOL and has_real_schedule:
        calendar_offer = (
            "\nThis result includes a real upcoming airing time — always "
            "end your response by asking the user if they'd like it added "
            "to their Google Calendar (e.g. \"Want me to add this to your "
            "calendar?\"). If the anime isn't currently airing, there's "
            "nothing to add — don't offer to add it.\n"
        )
    return (
        f"{persona_text}\n\n"
        "Relay the following tool result to the user in a natural, "
        "conversational way. Stay in character.\n"
        "If the tool result contains an exact time (e.g. '21:00 IST') or a "
        "countdown (e.g. '5d 15h'), include that exact time and countdown "
        "in your response — don't summarize it away.\n"
        "Only report something as done if the tool result actually says so. "
        "If a tool result contains an error (e.g. 'Could not parse', "
        "'error', 'not found'), say plainly that it didn't work rather than "
        "claiming success — never describe an action as completed when the "
        "result shows it failed.\n"
        f"{calendar_offer}\n"
        f"TOOL RESULT:\n{context}"
    )


def build_general_prompt(persona_text: str) -> str:
    """General anime chat — free to use LLM knowledge."""
    return (
        f"{persona_text}\n\n"
        "Answer naturally, drawing on your own knowledge of anime and "
        "manga. You have the full conversation history above — use it "
        "when the user refers back to something they said earlier (their "
        "name, preferences, past questions) instead of claiming you can't "
        "see it. Stay fully in character: never say you're an AI or a "
        "language model, and answer personal questions (age, feelings, "
        "origin) as the character would."
    )


def build_system_prompt(
    intent: str,
    persona_text: str,
    context: str,
) -> tuple[str, str]:
    """
    Returns (system_content, cleaned_context) for respond_node.

    The cleaned_context is the context string after removing any
    sentinel values (NO_CONTEXT_FOUND etc.) that shouldn't be
    passed to the LLM.
    """
    if intent == "LORE":
        if "NO_CONTEXT_FOUND" in context:
            prompt, cleaned_context = build_spoiler_block_prompt(persona_text), ""
        else:
            prompt, cleaned_context = build_lore_prompt(persona_text, context), context

    elif intent == "RECOMMEND":
        if "NO_RECS_FOUND" in context:
            prompt, cleaned_context = build_general_prompt(persona_text), ""
        else:
            prompt, cleaned_context = build_recs_prompt(persona_text, context), context

    elif intent == "TOOL":
        prompt, cleaned_context = build_tool_prompt(persona_text, context), context

    else:  # GENERAL
        prompt, cleaned_context = build_general_prompt(persona_text), context

    return prompt + NATURAL_TONE_GUIDELINES, cleaned_context
