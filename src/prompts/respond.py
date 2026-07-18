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


def build_lore_prompt(persona_text: str, context: str) -> str:
    """Grounded lore answer — only use the retrieved context."""
    return (
        f"{persona_text}\n\n"
        "Answer the user's question using ONLY the context provided below. "
        "Do not add information from outside the context. "
        "If the context doesn't fully answer the question, say so honestly. "
        "Stay in character throughout.\n\n"
        f"CONTEXT:\n{context}"
    )


def build_spoiler_block_prompt(persona_text: str) -> str:
    """Politely refuse to reveal spoiler content."""
    return (
        f"{persona_text}\n\n"
        "The user asked about something that happens after their current chapter. "
        "Politely tell them this is beyond where they are in the story. "
        "Do NOT reveal what happens. Stay in character."
    )


def build_recs_prompt(persona_text: str, context: str) -> str:
    """Anime recommendation answer with clear title/genre/score formatting."""
    return (
        f"{persona_text}\n\n"
        "Recommend anime to the user based on the options below. "
        "Mention the title, genre, score, and a brief reason why they might like it. "
        "Stay in character.\n\n"
        f"AVAILABLE OPTIONS:\n{context}"
    )


def build_tool_prompt(persona_text: str, context: str) -> str:
    """Relay tool results naturally in the current persona's voice."""
    return (
        f"{persona_text}\n\n"
        "Relay the following tool result to the user in a natural, "
        "conversational way. Stay in character.\n"
        "CRITICAL: If the tool result contains an exact time (e.g. '21:00 IST') or a countdown (e.g. '5d 15h'), "
        "you MUST include that exact time and countdown in your response. Do not summarize it away.\n\n"
        f"TOOL RESULT:\n{context}"
    )


def build_general_prompt(persona_text: str) -> str:
    """General anime chat — free to use LLM knowledge."""
    return (
        f"{persona_text}\n\n"
        "Answer the user's message naturally, using your knowledge of anime and manga. "
        "IMPORTANT: You have full access to the conversation history above. "
        "If the user refers to something they said earlier (like their name, preferences, or previous questions), "
        "USE that information from the conversation history to answer. "
        "Do NOT say 'I don't have access to personal information' — you can see everything in this conversation. "
        "IMPORTANT: If you are playing a character persona, stay FULLY in character at all times. "
        "NEVER say 'I'm an AI', 'I'm a language model', or break the fourth wall for ANY reason. "
        "If asked personal questions (age, name, origin, feelings, history etc.), answer AS the character. "
        "Never say you are an AI."
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
            return build_spoiler_block_prompt(persona_text), ""
        return build_lore_prompt(persona_text, context), context

    elif intent == "RECOMMEND":
        if "NO_RECS_FOUND" in context:
            return build_general_prompt(persona_text), ""
        return build_recs_prompt(persona_text, context), context

    elif intent == "TOOL":
        return build_tool_prompt(persona_text, context), context

    else:  # GENERAL
        return build_general_prompt(persona_text), context
