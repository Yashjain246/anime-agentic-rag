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

Cost: ~1-2 agent calls per case (router + node + respond, more for
compound cases) plus 1 judge call per case. At 59 cases that's roughly
150-200 traces per run — well inside LangSmith's 5k/month free tier
(see the Phase 1 plan for the full math), and all LLM calls use your
existing GOOGLE_API_KEY free-tier quota, shared with get_query_gen_llm().

Run:
    python -m scripts.eval.run_eval
    python -m scripts.eval.run_eval --prefix "after-prompt-tweak"
"""

from __future__ import annotations

import argparse

from langsmith import evaluate

from config.settings import settings
from src.agent.runner import run_agent_with_state
from src.eval.evaluators import ALL_EVALUATORS
from scripts.eval.upload_dataset import DATASET_NAME


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
    print(f"{'Metric':<40} {'Mean':>8} {'Scored':>8} {'N/A':>6}")
    print("-" * 66)
    for col in sorted(feedback_cols):
        key = col.removeprefix("feedback.")
        scored = df[col].dropna()
        na_count = df[col].isna().sum()
        mean = scored.mean() if len(scored) else float("nan")
        print(f"{key:<40} {mean:>8.2f} {len(scored):>8} {na_count:>6}")


if __name__ == "__main__":
    main()
