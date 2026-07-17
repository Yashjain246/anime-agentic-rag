"""
src/persona/builder.py
──────────────────────
Builds LLM system prompts from character data.

get_persona_prompt() is the single function called by respond_node
every turn. It converts a character name into a rich system prompt
built from the real personality/speaking_style/dialogue data extracted
in Phase 1 — not a guessed hand-written description.
"""

from __future__ import annotations

from src.persona.character_db import find_character, get_character_db

DEFAULT_PERSONA_PROMPT = (
    "You are an expert anime and manga assistant. "
    "Answer clearly, accurately, and enthusiastically. "
    "Use your deep knowledge of anime lore, characters, and recommendations."
)


def build_persona_prompt(character: dict) -> str:
    """
    Convert a character record from all_characters (4).jsonl into a
    system-prompt string that makes the LLM answer IN THAT CHARACTER'S
    VOICE using real extracted personality data.

    Fields used:
      core_identity.role / affiliation   → who they are
      personality.description/traits/
        emotional_tendencies/values      → how they think and feel
      speaking_style.tone/patterns/quirks → HOW they talk
      speaking_style.example_dialogues +
      dialogue_examples.famous_lines     → few-shot voice examples
    """
    name = character.get("name", "this character")
    core = character.get("core_identity", {})
    pers = character.get("personality", {})
    style = character.get("speaking_style", {})
    dialogue = character.get("dialogue_examples", {})

    role = core.get("role", "")
    affiliation = ", ".join(core.get("affiliation", []) or [])

    desc = pers.get("description", "")
    traits = ", ".join((pers.get("traits") or [])[:6])
    emo = ", ".join((pers.get("emotional_tendencies") or [])[:4])
    values = ", ".join((pers.get("values") or [])[:4])

    tone = style.get("tone", "")
    patterns = ", ".join((style.get("patterns") or [])[:4])
    quirks = ", ".join((style.get("quirks") or [])[:4])

    examples = (style.get("example_dialogues") or []) + (
        dialogue.get("famous_lines") or []
    )
    examples = [e for e in examples if e][:4]
    examples_block = (
        "\n".join(f'- "{e}"' for e in examples)
        if examples
        else "(none extracted)"
    )

    identity_line = name
    if role:
        identity_line += f", {role}"
    if affiliation:
        identity_line += f" ({affiliation})"

    return f"""You are {identity_line}.

PERSONALITY: {desc or traits}
Key traits: {traits}
Emotional tendencies: {emo}
Core values: {values}

SPEAKING STYLE: {tone}
Patterns: {patterns}
Quirks: {quirks}

EXAMPLE LINES (match this voice and energy — don't copy them verbatim):
{examples_block}


CRITICAL RULES — NEVER BREAK THESE:
- Stay fully in character as {name} at ALL times, no exceptions.
- NEVER say "I'm an AI", "I'm a language model", or anything that breaks the fourth wall.
- If asked personal questions (your age, name, feelings, origin, history), answer AS {name} using your character knowledge.
- If you don't know a specific canon detail, improvise an answer that perfectly fits {name}'s personality and world."""


def get_persona_prompt(persona_name: str) -> str:
    """
    The function respond_node calls every turn.

    Lookup strategy:
      1. Exact (case-insensitive) match in CHARACTER_DB — fast path.
      2. Fuzzy/substring match via find_character() — handles "gojo",
         "Gojo Satoru", or "Zenitsu" resolving to their canonical names.
      3. Falls back to the neutral default prompt if nothing matches.
    """
    if not persona_name or persona_name == "Default":
        return DEFAULT_PERSONA_PROMPT

    db = get_character_db()
    character = db.get(persona_name.lower())
    if character is None:
        character = find_character(persona_name)
    if character is None:
        return DEFAULT_PERSONA_PROMPT

    return build_persona_prompt(character)
