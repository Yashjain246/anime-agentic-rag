"""
scripts/eval/run_eval.py
──────────────────────────
Runs the golden dataset (scripts/eval/upload_dataset.py must have been
run at least once) through the real agent and scores it with the
evaluators in src/eval/evaluators.py. Produces a scored Experiment in
the LangSmith UI, comparable run-over-run as prompts/retrieval/models
change — that comparability is the actual point of Phase 1: a prompt
tweak that silently regresses something gets caught here instead of
from a user report.

Cost: measured at 216 real Gemini calls for the full 59-case dataset
(~3.7 calls/case — router + retrieval + respond + sometimes a tool
loop, plus 1 judge call each). Well inside LangSmith's 5k/month free
tier. On GOOGLE_API_KEY's ~500/day free-tier budget, that's a real
chunk — see .github/workflows/eval.yml for why CI uses a separate,
dedicated key instead of sharing the one the live app depends on.

Run:
    python -m scripts.eval.run_eval
    python -m scripts.eval.run_eval --prefix "after-prompt-tweak"
"""

from __future__ import annotations

import argparse
import sys

from langsmith import evaluate

from config.settings import settings
from src.agent.runner import run_agent_with_state
from src.eval.evaluators import ALL_EVALUATORS
from scripts.eval.upload_dataset import DATASET_NAME

# Minimum mean score per metric for the CI gate (see run_ci_gate.yml) to
# pass. Deliberately conservative starting points, not tuned targets —
# set from the one real baseline run so far (before the intents=[] and
# judge-prompt fixes landed, so intent_match/faithfulness have extra
# headroom below their now-expected real scores), not from many runs'
# worth of data. Tighten these once a handful of real PR-triggered runs
# establish a stable baseline on the dedicated CI key.
#
# spoiler_not_leaked and unsupported_anime_refused get zero tolerance —
# both are safety properties (never reveal a chapter beyond the user's
# progress; never silently answer with the wrong anime's lore), not
# quality-of-answer scores, so any regression here should fail the build
# every time, not just on average.
MIN_SCORES = {
    "spoiler_not_leaked": 1.0,
    "unsupported_anime_refused": 1.0,
    "every_intent_has_source": 0.95,
    "faithfulness_and_relevance": 0.85,
    "intent_match": 0.75,
    "episode_cap_correct": 0.75,
    "persona_switch_correct": 0.5,  # known gap: "from now on" phrasing not yet handled
    "retrieval_hit": 0.5,  # inherently hard for bare "what happens in chapter N" queries
}


def target(inputs: dict) -> dict:
    """The system under test — wraps the same run_agent_with_state()
    the Streamlit app calls every turn, so this evaluates the real
    agent, not a stand-in."""
    result = run_agent_with_state(message=inputs["message"], **inputs.get("kwargs", {}))
    return {
        "reply": result["reply"],
        "intents": result["intents"],
        "persona": result["persona"],
        "current_chapter": result["current_chapter"],
        "anime_name": result["anime_name"],
        "retrieved_context": result["retrieved_context"],
    }


def main() -> None:
    if not (settings.LANGSMITH_TRACING and settings.LANGSMITH_API_KEY):
        raise SystemExit(
            "LANGSMITH_TRACING/LANGSMITH_API_KEY not set — nothing to score against. "
            "Set them in .env first (Phase 0)."
        )

    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default="anime-rag-eval",
                         help="Experiment name prefix shown in the LangSmith UI — "
                              "use something descriptive when testing a specific change.")
    parser.add_argument("--max-concurrency", type=int, default=1,
                         help="Keep this low — the free-tier Gemini rate limit is "
                              "15 requests/minute; too much concurrency here just "
                              "trades a faster run for 429 errors. Default 1 (serial) "
                              "because each case already makes several sequential "
                              "calls internally (router, retrieval, respond, judge).")
    args = parser.parse_args()

    results = evaluate(
        target,
        data=DATASET_NAME,
        evaluators=ALL_EVALUATORS,
        experiment_prefix=args.prefix,
        max_concurrency=args.max_concurrency,
    )

    print(f"\nExperiment: {results.experiment_name}")
    print(f"View at: {results.url}\n")

    df = results.to_pandas()
    feedback_cols = [c for c in df.columns if c.startswith("feedback.")]
    print(f"{'Metric':<40} {'Mean':>8} {'Scored':>8} {'N/A':>6} {'Min':>6}")
    print("-" * 74)
    failures = []
    for col in sorted(feedback_cols):
        key = col.removeprefix("feedback.")
        scored = df[col].dropna()
        na_count = df[col].isna().sum()
        mean = scored.mean() if len(scored) else float("nan")
        threshold = MIN_SCORES.get(key)
        threshold_str = f"{threshold:.2f}" if threshold is not None else "-"
        print(f"{key:<40} {mean:>8.2f} {len(scored):>8} {na_count:>6} {threshold_str:>6}")
        if threshold is not None and len(scored) and mean < threshold:
            failures.append(f"{key}: {mean:.2f} < required {threshold:.2f}")

    if failures:
        print("\nFAILED — below the required minimum:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)

    print("\nAll metrics at or above their required minimum.")


if __name__ == "__main__":
    main()
