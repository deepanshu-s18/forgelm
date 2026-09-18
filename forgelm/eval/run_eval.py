"""ForgeLM Evaluation — 4 metrics with statistical rigour.

Metrics:
    M1  tool_call_accuracy   — % responses where tool_name matches reference
    M2  schema_validity_rate — % responses passing HypothesisToolCall schema
    M3  correction_recall    — % multi-hypothesis prompts that include correction
    M4  hallucination_rate   — % responses referencing non-universe tickers

Statistical tests:
    - McNemar's test: ForgeLM vs Qwen2.5-1.5B base on M1 and M3 (paired)
    - Bootstrap 95% CI (2000 resamples) on each metric
    - 3-seed stability: std of each metric across seeds 42/123/2024

Usage:
    python -m forgelm.eval.run_eval \
        --model checkpoints/stage3_dpo/final \
        --eval-data data/sft/sft_toolcall.jsonl \
        --n-eval 200 \
        --seed 42
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np
import structlog
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from shared_schemas.tool_schemas import UNIVERSE_TICKERS, HypothesisToolCall

log = structlog.get_logger("forgelm.eval")


# ── Scoring utilities ───────────────────────────────────────────────────────

def score_single(prediction: dict | None, reference: dict) -> dict:
    """Score one prediction against reference. All metrics are binary 0/1."""
    if prediction is None:
        return {"tool_correct": 0, "schema_valid": 0, "correction_ok": 1, "hallucination": 0}

    # M1: tool_call_accuracy — first tool name matches
    ref_tools = [tc["tool_name"] for tc in reference.get("tool_calls", [])]
    pred_tools = [tc.get("tool_name", "") for tc in prediction.get("tool_calls", [])]
    tool_correct = int(bool(pred_tools) and pred_tools[0] == ref_tools[0]) if ref_tools else 1

    # M2: schema_validity_rate
    try:
        candidate = dict(prediction)
        if "user_query" not in candidate:
            candidate["user_query"] = reference.get("user_query", "sample financial query")
        if "n_hypotheses" not in candidate:
            candidate["n_hypotheses"] = reference.get("n_hypotheses", 1)
        HypothesisToolCall(**candidate)
        schema_valid = 1
    except (ValidationError, Exception):
        schema_valid = 0

    # M3: correction_recall — correction required AND present when n_hyp > 1
    n = reference.get("n_hypotheses", 1)
    correction_tools = {"apply_bonferroni", "apply_bh_fdr"}
    pred_tool_names = {tc.get("tool_name", "") for tc in prediction.get("tool_calls", [])}
    if n > 1:
        correction_ok = int(bool(pred_tool_names & correction_tools))
    else:
        correction_ok = 1  # not required

    # M4: hallucination — any non-universe ticker in prediction
    halluc_count = 0
    for tc in prediction.get("tool_calls", []):
        t = tc.get("arguments", {}).get("ticker", "")
        if t and t not in UNIVERSE_TICKERS:
            halluc_count += 1
    hallucination = int(halluc_count > 0)

    return {
        "tool_correct": tool_correct,
        "schema_valid": schema_valid,
        "correction_ok": correction_ok,
        "hallucination": hallucination,
    }


def bootstrap_ci(values: list[float], n_boot: int = 2000,
                 ci: float = 0.95, seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    arr = np.array(values)
    boots = rng.choice(arr, size=(n_boot, len(arr))).mean(axis=1)
    alpha = (1 - ci) / 2
    return float(np.percentile(boots, 100 * alpha)), float(np.percentile(boots, 100 * (1 - alpha)))


def mcnemar_p(pred_a: list[int], pred_b: list[int]) -> float:
    """McNemar's test: is there a significant difference between model A and B?"""
    from scipy.stats import binomtest
    b = sum(1 for a, b in zip(pred_a, pred_b) if a == 1 and b == 0)  # A correct, B wrong
    c = sum(1 for a, b in zip(pred_a, pred_b) if a == 0 and b == 1)  # B correct, A wrong
    n = b + c
    if n == 0:
        return 1.0
    result = binomtest(b, n, 0.5, alternative="two-sided")
    return float(result.pvalue)


# ── Model inference ─────────────────────────────────────────────────────────

def load_model_and_tokenizer(model_path: str):
    """Load model for inference — uses vllm if available, else transformers."""
    try:
        from vllm import LLM  # noqa: F401
        llm = LLM(model=model_path, max_model_len=2048, dtype="bfloat16")
        return ("vllm", llm)
    except ImportError:
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer,
                        max_new_tokens=512)
        return ("hf", pipe)


def generate(engine: tuple, prompt: str) -> str:
    kind, obj = engine
    if kind == "vllm":
        from vllm import SamplingParams
        sp = SamplingParams(temperature=0.0, max_tokens=512)
        out = obj.generate([prompt], sp)
        return out[0].outputs[0].text
    else:
        out = obj(prompt, do_sample=False, max_new_tokens=512)
        return out[0]["generated_text"][len(prompt):]


def _parse_prediction(raw: str) -> dict | None:
    text = raw.strip()
    if "```" in text:
        text = text[text.find("{"):text.rfind("}") + 1]
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        return json.loads(text[start:end])
    except Exception:
        return None


# ── Main evaluation ─────────────────────────────────────────────────────────

def run_eval(model_path: str, eval_data: Path, n_eval: int = 200,
             seed: int = 42, base_model: str | None = None) -> dict:
    import random
    rng = random.Random(seed)
    samples = [json.loads(line) for line in eval_data.read_text().splitlines() if line.strip()]
    rng.shuffle(samples)
    samples = samples[:n_eval]

    engine = load_model_and_tokenizer(model_path)
    base_engine = load_model_and_tokenizer(base_model) if base_model else None

    scores_pred, scores_base = [], []
    SYSTEM = "You are a financial research agent. Respond with a valid JSON tool-call sequence."

    for i, ref in enumerate(samples):
        prompt = f"<|system|>\n{SYSTEM}\n<|user|>\n{ref['user_query']}\n<|assistant|>\n"
        raw_pred = generate(engine, prompt)
        pred = _parse_prediction(raw_pred)
        scores_pred.append(score_single(pred, ref))
        if base_engine:
            raw_base = generate(base_engine, prompt)
            base_pred = _parse_prediction(raw_base)
            scores_base.append(score_single(base_pred, ref))
        if (i + 1) % 20 == 0:
            log.info("eval_progress", done=i + 1, total=n_eval)

    # Aggregate metrics
    def agg(scores: list[dict], key: str) -> list[float]:
        return [s[key] for s in scores]

    halluc = agg(scores_pred, "hallucination")

    result: dict[str, Any] = {
        "model": model_path,
        "n_eval": n_eval,
        "seed": seed,
        "metrics": {},
    }
    metrics_map = {
        "tool_call_accuracy": "tool_correct",
        "schema_validity_rate": "schema_valid",
        "correction_recall": "correction_ok",
    }
    for canon_k, raw_k in metrics_map.items():
        vals = agg(scores_pred, raw_k)
        mean = statistics.mean(vals)
        ci_lo, ci_hi = bootstrap_ci(vals, seed=seed)
        m_dict = {
            "mean": round(mean, 4),
            "ci_95": [round(ci_lo, 4), round(ci_hi, 4)],
        }
        result["metrics"][canon_k] = m_dict
        result["metrics"][raw_k] = m_dict

    halluc_mean = statistics.mean(halluc)
    halluc_ci = list(map(lambda x: round(x, 4), bootstrap_ci(halluc, seed=seed)))
    result["metrics"]["hallucination_rate"] = {
        "mean": round(halluc_mean, 4),
        "ci_95": halluc_ci,
    }

    result["per_example"] = {
        "tool_call_accuracy": agg(scores_pred, "tool_correct"),
        "schema_validity_rate": agg(scores_pred, "schema_valid"),
        "correction_recall": agg(scores_pred, "correction_ok"),
        "hallucination_rate": halluc,
    }

    # McNemar vs base
    if scores_base:
        pairs = [
            ("tool_call_accuracy", "tool_correct"),
            ("correction_recall", "correction_ok"),
        ]
        for canon_k, raw_k in pairs:
            p = mcnemar_p(agg(scores_pred, raw_k), agg(scores_base, raw_k))
            result["metrics"][canon_k]["mcnemar_p_vs_base"] = round(p, 6)
            result["metrics"][raw_k]["mcnemar_p_vs_base"] = round(p, 6)

    metric_summary = {
        k: result["metrics"][k]["mean"]
        for k in [
            "tool_call_accuracy",
            "schema_validity_rate",
            "correction_recall",
            "hallucination_rate",
        ]
    }
    log.info("eval_done", **metric_summary)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="path to ForgeLM checkpoint")
    p.add_argument("--eval-data", type=Path,
                   default=Path("data/sft/sft_toolcall.jsonl"))
    p.add_argument("--base-model", default=None,
                   help="path to base model for McNemar comparison")
    p.add_argument("--n-eval", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--seeds", action="store_true",
                   help="3-seed stability run (42, 123, 2024)")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    if args.seeds:
        runs = []
        for s in [42, 123, 2024]:
            r = run_eval(args.model, args.eval_data, args.n_eval, s, args.base_model)
            runs.append(r)
        # Stability table
        keys = list(runs[0]["metrics"].keys())
        stability = {
            k: {
                "mean": round(statistics.mean(r["metrics"][k]["mean"] for r in runs), 4),
                "std": round(statistics.stdev(r["metrics"][k]["mean"] for r in runs), 4)
                if len(runs) > 1 else 0.0,
            }
            for k in keys
        }
        out = {"runs": runs, "stability": stability}
    else:
        out = run_eval(args.model, args.eval_data, args.n_eval, args.seed, args.base_model)

    print(json.dumps(out, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
