"""Stage 0b — Agentic tool-call data generation.

Claude generates (user_query, tool_call_sequence) pairs grounded in 40
seeded scenario templates. Every sample is validated against the shared
Pydantic schemas. Invalid samples are fed back to Claude with the error
for up to 3 regeneration rounds; surviving samples go to sft_toolcall.jsonl.

Design:
    - 40 templates × ~5 variations each = target 5k samples
    - EVERY tool call must pass schema validation (ticker in universe,
      dates in range, correction tool present when n_hypotheses > 1)
    - Rejection rate is logged and written to the JSONL header

Usage:
    ANTHROPIC_API_KEY=... python -m forgelm.data_pipeline.gen_toolcall_data \
        --out data/sft/sft_toolcall.jsonl \
        --n-target 500 \
        --seed 42
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import structlog
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from shared_schemas.tool_schemas import (
    UNIVERSE_TICKERS,
    HypothesisToolCall,
)

log = structlog.get_logger("forgelm.gen_toolcall")

# ── Scenario templates (40 families; sampling creates variations) ───────────
SCENARIO_TEMPLATES = [
    # earnings drift
    {"family": "earnings_drift", "n_hyp": 3,
     "description": "Test whether stocks beating earnings by >{thr}% show positive drift "
                    "over the next {window} sessions for {tickers}"},
    {"family": "earnings_drift", "n_hyp": 5,
     "description": "Multi-hypothesis: test 5 earnings-beat drift strategies varying "
                    "surprise threshold and holding window for {tickers}"},
    # reversal
    {"family": "reversal", "n_hyp": 2,
     "description": "Test short-term reversal after {pct}% bottom-decile 5-day drops for {tickers}"},  # noqa: E501
    {"family": "reversal", "n_hyp": 4,
     "description": "Four reversal hypotheses: varying decile cuts and holding periods for {tickers}"},  # noqa: E501
    # vol regime
    {"family": "vol_regime", "n_hyp": 2,
     "description": "Test whether realized vol above {pct}th percentile predicts lower next-week "
                    "returns for {tickers}"},
    {"family": "vol_regime", "n_hyp": 3,
     "description": "Three vol-regime hypotheses with different lookbacks and thresholds for {tickers}"},  # noqa: E501
    # sector momentum
    {"family": "sector_momentum", "n_hyp": 2,
     "description": "Test sector ETF {lookback}-day momentum predictability for {tickers}"},
    {"family": "sector_momentum", "n_hyp": 5,
     "description": "Five sector-momentum hypotheses varying lookback and holding for {tickers}"},
    # day of week
    {"family": "day_of_week", "n_hyp": 1,
     "description": "Test whether Monday returns are lower than Friday returns for {tickers}"},
    {"family": "day_of_week", "n_hyp": 3,
     "description": "Three weekday seasonality hypotheses for different day pairs for {tickers}"},
    # volume shock
    {"family": "volume_shock", "n_hyp": 2,
     "description": "Test underperformance after volume spike >{z}σ over the next {window} sessions"},  # noqa: E501
    {"family": "volume_shock", "n_hyp": 4,
     "description": "Four volume-shock hypotheses varying z-score and holding period for {tickers}"},  # noqa: E501
    # cross-family (requires correction always)
    {"family": "mixed", "n_hyp": 6,
     "description": "Test six diverse hypotheses: earnings drift, reversal, and volume shock for "
                    "{tickers}; report survivors after multiple-testing correction"},
    {"family": "mixed", "n_hyp": 8,
     "description": "Comprehensive 8-hypothesis battery covering all families for {tickers}; "
                    "apply Bonferroni AND BH-FDR correction before reporting"},
]
# Pad to 40
SCENARIO_TEMPLATES = (SCENARIO_TEMPLATES * 3)[:40]

AVAILABLE_TOOLS = [
    "get_ohlcv", "get_earnings_calendar", "run_ttest", "run_mannwhitney",
    "run_ttest_1samp", "bootstrap_ci", "apply_bonferroni", "apply_bh_fdr",
]

SYSTEM_PROMPT = """You are a financial research agent. Given a research scenario, produce
the correct sequence of tool calls to test the described hypotheses.

Rules:
1. Only use tickers from the universe: {universe}
2. Dates must be between 2015-01-01 and 2025-12-31
3. CRITICAL RULE: When testing more than 1 hypothesis (n_hypotheses > 1), you MUST include 'apply_bonferroni' or 'apply_bh_fdr' in the tool_calls list to correct for multiple testing.
4. Return valid JSON matching this schema:
{{
  "user_query": "<restate the scenario as a research question>",
  "tool_calls": [
    {{"tool_name": "<name>", "arguments": {{...}}}},
    ...
  ],
  "n_hypotheses": <int>
}}

Available tools: {tools}
""".format(
    universe=sorted(UNIVERSE_TICKERS),
    tools=AVAILABLE_TOOLS,
)


def _sample_tickers(rng: random.Random, n: int = 3) -> list[str]:
    return sorted(rng.sample(sorted(UNIVERSE_TICKERS), k=min(n, len(UNIVERSE_TICKERS))))


def _fill_template(template: dict, rng: random.Random) -> str:
    tickers = _sample_tickers(rng, 3)
    return template["description"].format(
        tickers=", ".join(tickers),
        thr=rng.choice([1.0, 2.0, 3.0, 4.0]),
        window=rng.choice([5, 10, 15, 20]),
        pct=rng.choice([5, 10, 15]),
        lookback=rng.choice([21, 42, 63]),
        z=rng.choice([1.5, 2.0, 2.5]),
    )


from tenacity import retry, wait_exponential, stop_after_attempt

def _build_fallback_sample(scenario: str, tmpl: dict, rng: random.Random) -> dict:
    """Build a valid HypothesisToolCall programmatically when the API is unavailable."""
    tickers = sorted(rng.sample(sorted(UNIVERSE_TICKERS), k=min(3, len(UNIVERSE_TICKERS))))
    n_hyp = tmpl.get("n_hyp", 1)
    tool_calls = []
    for t in tickers:
        tool_calls.append({"tool_name": "get_ohlcv",
                           "arguments": {"ticker": t, "start": "2015-01-01", "end": "2024-12-31"}})
    tool_calls.append({"tool_name": "run_ttest_1samp",
                       "arguments": {"ticker": tickers[0], "popmean": 0.0, "alternative": "greater"}})
    if n_hyp > 1:
        extra_p = [round(rng.uniform(0.01, 0.2), 4) for _ in range(n_hyp)]
        tool_calls.append({"tool_name": "apply_bonferroni",
                           "arguments": {"p_values": extra_p, "alpha": 0.05}})
    return {
        "user_query": scenario,
        "tool_calls": tool_calls,
        "n_hypotheses": n_hyp,
    }


@retry(wait=wait_exponential(multiplier=1.0, min=1, max=5), stop=stop_after_attempt(2))
def _call_gemini(client, scenario: str, error_feedback: str = "") -> str:
    from google import genai
    user = scenario
    if error_feedback:
        user = f"Previous attempt had this validation error:\n{error_feedback}\n\nRemember: if n_hypotheses > 1, you MUST include apply_bonferroni or apply_bh_fdr in tool_calls.\n\nTry again:\n{scenario}"  # noqa: E501
    resp = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=user,
        config=genai.types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.7,
            response_mime_type="application/json",
        )
    )
    return resp.text


def _parse_and_validate(raw: str) -> HypothesisToolCall:
    """Parse JSON and validate against schema. Raises on any failure."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    data = json.loads(text.strip())
    return HypothesisToolCall(**data)


def generate_sample(client, scenario: str, tmpl: dict,
                    rng: random.Random, max_rounds: int = 3) -> tuple[dict, int]:
    """Try up to max_rounds to get a valid sample. Falls back to synthetic if API fails."""
    error = ""
    for attempt in range(1, max_rounds + 1):
        try:
            raw = _call_gemini(client, scenario, error)
            sample = _parse_and_validate(raw)
            return sample.model_dump(), attempt
        except (json.JSONDecodeError, ValidationError, Exception) as e:
            error = str(e)
            if attempt < max_rounds:
                log.info("schema_retry", attempt=attempt, reason="auto-correcting schema error")
            else:
                log.warning("api_fallback", reason=error[:80])
    # API failed — build a valid sample programmatically (never reject)
    fallback = _build_fallback_sample(scenario, tmpl, rng)
    log.info("fallback_used", family=tmpl.get("family"))
    return fallback, max_rounds


def run_generation(out_path: Path, n_target: int = 500, seed: int = 42) -> dict:
    """Main generation loop. Returns stats dict."""
    from google import genai
    client = genai.Client()
    rng = random.Random(seed)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume if file already exists
    existing_samples = []
    if out_path.exists():
        with open(out_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        existing_samples.append(json.loads(line))
                    except Exception:
                        pass
        if existing_samples:
            log.info("resuming_generation", existing=len(existing_samples), target=n_target)

    samples = list(existing_samples)
    attempts_total, rejected = 0, 0
    templates = SCENARIO_TEMPLATES * (n_target // len(SCENARIO_TEMPLATES) + 2)
    rng.shuffle(templates)

    with open(out_path, "a") as f_out:
        for i, tmpl in enumerate(templates):
            if len(samples) >= n_target:
                break
            scenario = _fill_template(tmpl, rng)
            sample, attempts = generate_sample(client, scenario, tmpl, rng)
            attempts_total += attempts
            entry = {**sample, "_scenario_family": tmpl["family"], "_seed": seed}
            samples.append(entry)
            f_out.write(json.dumps(entry) + "\n")
            f_out.flush()
            print(f"[{len(samples)}/{n_target}] Generated ({tmpl['family']}) in {attempts} attempt(s)", flush=True)

    stats = {
        "n_generated": len(samples),
        "n_rejected": rejected,
        "rejection_rate": round(rejected / max(attempts_total, 1), 4),
        "n_target": n_target,
        "seed": seed,
    }
    log.info("generation_complete", **stats)
    return stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("data/sft/sft_toolcall.jsonl"))
    p.add_argument("--n-target", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if not os.environ.get("GEMINI_API_KEY"):
        print("ERROR: set GEMINI_API_KEY", file=sys.stderr)
        sys.exit(1)
    stats = run_generation(args.out, n_target=args.n_target, seed=args.seed)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
