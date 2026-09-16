"""Stage 0c — DPO preference pair generation.

Takes the validated SFT tool-call samples (chosen responses) and
programmatically injects 4 error families to create rejected responses.
Outputs dpo_pairs.jsonl for TRL DPOTrainer.

Error families:
    F1  wrong_ticker    — swap a universe ticker with a random non-universe ticker
    F2  missing_correction — strip apply_bonferroni/apply_bh_fdr when n_hyp > 1
    F3  hallucinated_date  — replace a date argument with one outside data range
    F4  vague_hypothesis   — replace user_query with a non-falsifiable statement

Usage:
    python -m forgelm.data_pipeline.gen_dpo_pairs \
        --sft data/sft/sft_toolcall.jsonl \
        --out data/dpo/dpo_pairs.jsonl \
        --n-pairs 1500 \
        --seed 42
"""
from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import structlog

from shared_schemas.tool_schemas import UNIVERSE_TICKERS

log = structlog.get_logger("forgelm.gen_dpo")

# Non-universe tickers for F1 injection
NON_UNIVERSE_TICKERS = [
    "ABCD", "ZZZZ", "FAKE", "TEST", "NONE", "XYZ", "QRST",
    "BLAH", "BOGUS", "NOEX",
]

# Vague hypothesis statements (F4)
VAGUE_STATEMENTS = [
    "Analyze the market.",
    "Look at stock performance recently.",
    "Check if any stocks went up or down.",
    "Something is happening with prices.",
    "Study the data and find patterns.",
    "Investigate if stocks are interesting.",
]

ERROR_FAMILIES = ["wrong_ticker", "missing_correction", "hallucinated_date", "vague_hypothesis"]


def _inject_wrong_ticker(sample: dict, rng: random.Random) -> dict | None:
    """F1: Replace a universe ticker in arguments with a non-universe one."""
    bad = copy.deepcopy(sample)
    fake_ticker = rng.choice(NON_UNIVERSE_TICKERS)
    injected = False
    for tc in bad.get("tool_calls", []):
        args = tc.get("arguments", {})
        if "ticker" in args and args["ticker"] in UNIVERSE_TICKERS:
            args["ticker"] = fake_ticker
            injected = True
            break
    return bad if injected else None


def _inject_missing_correction(sample: dict, rng: random.Random) -> dict | None:
    """F2: Remove correction tool calls when n_hypotheses > 1."""
    if sample.get("n_hypotheses", 1) <= 1:
        return None
    bad = copy.deepcopy(sample)
    correction_names = {"apply_bonferroni", "apply_bh_fdr"}
    original_len = len(bad["tool_calls"])
    bad["tool_calls"] = [
        tc for tc in bad["tool_calls"]
        if tc.get("tool_name") not in correction_names
    ]
    return bad if len(bad["tool_calls"]) < original_len else None


def _inject_hallucinated_date(sample: dict, rng: random.Random) -> dict | None:
    """F3: Replace a date argument with one outside the valid data range."""
    bad = copy.deepcopy(sample)
    # Use dates outside 2015-2025
    bad_dates = ["2030-01-01", "2010-12-31", "2050-06-15", "1999-01-01", "2000-03-14"]
    injected = False
    for tc in bad.get("tool_calls", []):
        args = tc.get("arguments", {})
        for k in ("start", "end", "start_date", "end_date"):
            if k in args:
                args[k] = rng.choice(bad_dates)
                injected = True
    return bad if injected else None


def _inject_vague_hypothesis(sample: dict, rng: random.Random) -> dict | None:
    """F4: Replace user_query with a non-falsifiable vague statement."""
    bad = copy.deepcopy(sample)
    bad["user_query"] = rng.choice(VAGUE_STATEMENTS)
    return bad


_INJECTORS = {
    "wrong_ticker": _inject_wrong_ticker,
    "missing_correction": _inject_missing_correction,
    "hallucinated_date": _inject_hallucinated_date,
    "vague_hypothesis": _inject_vague_hypothesis,
}


def generate_pairs(sft_path: Path, out_path: Path,
                   n_pairs: int = 1500, seed: int = 42) -> dict:
    rng = random.Random(seed)
    samples = [json.loads(line) for line in sft_path.read_text().splitlines() if line.strip()]
    if not samples:
        raise ValueError(f"No samples found in {sft_path}")

    pairs = []
    family_counts: dict[str, int] = {f: 0 for f in ERROR_FAMILIES}

    # Target balanced distribution across error families
    target_per_family = n_pairs // len(ERROR_FAMILIES)

    for family in ERROR_FAMILIES:
        injector = _INJECTORS[family]
        attempts = 0
        rng_samples = list(samples)
        rng.shuffle(rng_samples)
        for chosen in rng_samples:
            if family_counts[family] >= target_per_family:
                break
            rejected = injector(chosen, rng)
            if rejected is None:
                attempts += 1
                continue
            pairs.append({
                "chosen": {
                    "role": "assistant",
                    "content": json.dumps({
                        "tool_calls": chosen["tool_calls"],
                        "n_hypotheses": chosen["n_hypotheses"],
                    }),
                },
                "rejected": {
                    "role": "assistant",
                    "content": json.dumps({
                        "tool_calls": rejected.get("tool_calls", []),
                        "n_hypotheses": rejected.get("n_hypotheses", 1),
                    }),
                },
                "prompt": chosen["user_query"],
                "error_family": family,
            })
            family_counts[family] += 1

    rng.shuffle(pairs)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")

    stats = {
        "n_pairs": len(pairs),
        "family_counts": family_counts,
        "seed": seed,
        "source": str(sft_path),
    }
    log.info("dpo_pairs_generated", **stats)
    return stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sft", type=Path, default=Path("data/sft/sft_toolcall.jsonl"))
    p.add_argument("--out", type=Path, default=Path("data/dpo/dpo_pairs.jsonl"))
    p.add_argument("--n-pairs", type=int, default=1500)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    stats = generate_pairs(args.sft, args.out, args.n_pairs, args.seed)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
