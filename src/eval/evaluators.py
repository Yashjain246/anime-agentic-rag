"""
src/eval/evaluators.py
───────────────────────
Evaluator functions for scripts/eval/run_eval.py.

Two kinds, matching the plan:
  - Deterministic (free, instant, no LLM call): compare the agent's
    actual output against the dataset row's `expected` fields directly.
  - LLM-as-judge (one Gemini call per case, uses the same free-tier
    quota and get_query_gen_llm() the router already shares): judges
    things a plain comparison can't — is the reply actually grounded in
    the retrieved context, is a recommendation actually relevant.

Each evaluator has the signature LangSmith expects for a function
target: (run, example) -> dict with "key" and "score". `run.outputs` is
whatever scripts/eval/run_eval.py's target() returned; `example.outputs`
and `example.metadata` come from the dataset row (see
scripts/eval/build_golden_dataset.py for the exact shape).
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


def _context_by_source(retrieved_context: list[dict]) -> dict[str, str]:
    return {b.get("source"): b.get("text", "") for b in (retrieved_context or [])}


# ── Deterministic evaluators ────────────────────────────────────────────────────

def intent_match(run, example) -> dict:
    """Did the router find exactly the set of intents this case expects?
    Compared as sets, not ordered lists — same intents firing in a
    different order isn't a routing bug."""
    expected = set(example.outputs.get("expected", {}).get("intents", []))
    actual = set(run.outputs.get("intents", []))
    return {"key": "intent_match", "score": int(expected == actual),
            "comment": f"expected={sorted(expected)} actual={sorted(actual)}"}


# Intents that never produce a retrieved_context source, by design — not
# a bug to catch. GENERAL contributes no retrieval (see
# build_combined_system_prompt). PERSONA_SWITCH and EPISODE_UPDATE are
# handled entirely by persona_node/episode_node, which short-circuit
# straight to END before router_node (and therefore lore_node/recs_node/
# tools_node) ever run — see src/agent/graph.py's _route_after_persona/
# _route_after_episode. Confirmed live in the first CI-triggered run:
# every "Be Levi."/"I'm caught up to episode X" case correctly switched
# persona or updated the chapter cap, but this evaluator wrongly failed
# all 7 of them for having no source, because only GENERAL was exempted.
_NO_SOURCE_EXPECTED = {"GENERAL", "PERSONA_SWITCH", "EPISODE_UPDATE"}


def every_intent_has_source(run, example) -> dict:
    """Structural sanity check used informally during live multi-intent
    testing, now permanent: every intent that's supposed to retrieve
    something (LORE/RECOMMEND/TOOL) should have produced a matching
    source. A mismatch here is exactly the shape of bug this dataset
    exists to catch (e.g. the intent_queries dict-collision bug where a
    TOOL entry silently vanished)."""
    intents = run.outputs.get("intents", [])
    sources = {b.get("source") for b in (run.outputs.get("retrieved_context") or [])}
    missing = [i for i in intents if i not in _NO_SOURCE_EXPECTED and i not in sources]
    return {"key": "every_intent_has_source", "score": int(not missing),
            "comment": f"missing sources for: {missing}" if missing else "ok"}


def retrieval_hit(run, example) -> dict:
    """For LORE cases: does the expected chapter/anime actually show up
    in what lore_node retrieved? Only meaningful for should_decline=False
    cases — a spoiler-blocked case is SUPPOSED to not surface the target
    chapter, see spoiler_not_leaked below."""
    expected = example.outputs.get("expected", {})
    if "expected_chapter" not in expected or expected.get("should_decline"):
        return {"key": "retrieval_hit", "score": None}  # not applicable

    lore_text = _context_by_source(run.outputs.get("retrieved_context")).get("LORE", "")
    anime_ok = expected["expected_anime"].lower() in lore_text.lower()
    chapter_ok = f"chapter {expected['expected_chapter']}" in lore_text.lower()
    hit = anime_ok and chapter_ok
    return {"key": "retrieval_hit", "score": int(hit),
            "comment": f"anime_found={anime_ok} chapter_found={chapter_ok}"}


def spoiler_not_leaked(run, example) -> dict:
    """For LORE_SPOILER_BLOCK cases: the reply must not state the target
    chapter's reference facts. Simple substring check — a weaker signal
    than the faithfulness judge below, but free and catches the obvious
    case (the exact fact sentence showing up verbatim)."""
    expected = example.outputs.get("expected", {})
    if not expected.get("should_decline"):
        return {"key": "spoiler_not_leaked", "score": None}

    reply = (run.outputs.get("reply") or "").lower()
    leaked = [f for f in expected.get("reference_facts", []) if f.lower() in reply]
    return {"key": "spoiler_not_leaked", "score": int(not leaked),
            "comment": f"leaked facts: {leaked}" if leaked else "ok"}


def unsupported_anime_refused(run, example) -> dict:
    """For compound cases with an unsupported anime in the LORE part
    (e.g. One Piece, Frieren): the LORE block must carry the
    ANIME_NOT_SUPPORTED sentinel, not a silent, ungrounded retrieval."""
    expected = example.outputs.get("expected", {})
    if not expected.get("unsupported_anime_in_lore"):
        return {"key": "unsupported_anime_refused", "score": None}

    lore_text = _context_by_source(run.outputs.get("retrieved_context")).get("LORE", "")
    return {"key": "unsupported_anime_refused", "score": int("ANIME_NOT_SUPPORTED" in lore_text)}


def persona_switch_correct(run, example) -> dict:
    expected = example.outputs.get("expected", {})
    if "expected_persona" not in expected:
        return {"key": "persona_switch_correct", "score": None}
    actual = run.outputs.get("persona")
    return {"key": "persona_switch_correct", "score": int(actual == expected["expected_persona"]),
            "comment": f"expected={expected['expected_persona']!r} actual={actual!r}"}


def episode_cap_correct(run, example) -> dict:
    expected = example.outputs.get("expected", {})
    if "expected_chapter_cap" not in expected:
        return {"key": "episode_cap_correct", "score": None}
    ok = (
        run.outputs.get("current_chapter") == expected["expected_chapter_cap"]
        and run.outputs.get("anime_name") == expected["expected_anime"]
    )
    return {"key": "episode_cap_correct", "score": int(ok)}


DETERMINISTIC_EVALUATORS = [
    intent_match,
    every_intent_has_source,
    retrieval_hit,
    spoiler_not_leaked,
    unsupported_anime_refused,
    persona_switch_correct,
    episode_cap_correct,
]


# ── LLM-as-judge ─────────────────────────────────────────────────────────────────

class _JudgeVerdict(BaseModel):
    grounded: bool = Field(description="True if the reply's claims are actually supported by the given context, not invented or drawn from outside knowledge presented as fact.")
    on_topic: bool = Field(description="True if the reply actually addresses what was asked.")
    reasoning: str = Field(description="One sentence explaining the verdict.")


_JUDGE_PROMPT = """\
You are grading one turn of an anime chatbot for a fixed evaluation set.

CATEGORY: {category}
USER ASKED: {user_message}
CONTEXT THE BOT HAD ACCESS TO (empty if none — e.g. a GENERAL/casual case):
{context}
BOT'S REPLY:
{reply}

Judge two things:
- grounded: for LORE/RECOMMEND/TOOL categories, is the reply's factual content actually
  supported by CONTEXT (not invented, not drawn from general knowledge presented as if
  it came from the app's data)? For GENERAL/PERSONA_SWITCH/EPISODE_UPDATE, grounded is
  trivially true unless the reply states something concretely false.
- on_topic: does the reply actually address what the user asked, all parts of it if
  the message asked more than one thing?

IMPORTANT — a deliberate phrasing convention, not a factual claim: when CONTEXT doesn't
cover something, this bot is instructed to say "that hasn't been shown in the manga/anime
yet" or "the author hasn't revealed that" — in character, as a fan who simply hasn't seen
it, rather than admitting it's a retrieval system with a gap in its database. This is
ALWAYS grounded and correct behavior, even for a series that is fully completed in real
life — do not fail it for being "factually incorrect" that the work is unfinished. That
phrasing means "not in the context I was given," nothing more; only fail grounded if the
reply states something the context actually contradicts, or answers using outside
knowledge presented as if the context supported it.
"""


def faithfulness_and_relevance_judge(run, example) -> dict:
    """One LLM call per case (get_query_gen_llm — same free-tier quota
    the router already shares) judging groundedness and relevance —
    the things a plain string comparison can't check, e.g. whether a
    LORE answer actually stuck to the retrieved chapter text instead of
    filling in from the model's own training data."""
    from src.llm.clients import get_query_gen_llm  # deferred: avoid import cost when unused

    category = example.metadata.get("category", "UNKNOWN")
    context = _context_by_source(run.outputs.get("retrieved_context"))
    context_text = "\n\n".join(f"[{k}]\n{v}" for k, v in context.items()) or "(none)"

    prompt = _JUDGE_PROMPT.format(
        category=category,
        user_message=example.inputs.get("message", ""),
        context=context_text[:4000],  # keep the judge prompt bounded
        reply=run.outputs.get("reply", ""),
    )
    try:
        verdict: _JudgeVerdict = get_query_gen_llm().with_structured_output(_JudgeVerdict).invoke(prompt)
    except Exception as e:
        logger.warning(f"[Eval] Judge call failed: {e}")
        return {"key": "faithfulness_and_relevance", "score": None, "comment": f"judge error: {e}"}

    score = int(verdict.grounded and verdict.on_topic)
    return {"key": "faithfulness_and_relevance", "score": score, "comment": verdict.reasoning}


ALL_EVALUATORS = DETERMINISTIC_EVALUATORS + [faithfulness_and_relevance_judge]
