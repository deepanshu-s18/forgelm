"""Decontamination — remove eval samples that overlap with training data.

Uses 8-gram overlap as the contamination signal (same threshold as the spec).
A sample is flagged if any 8-gram from it appears in ANY training file.

Usage:
    python -m forgelm.data_pipeline.decontaminate \
        --train-files data/sft/sft_toolcall.jsonl data/events/sft_events.jsonl \
        --eval-in  eval/eval_data_raw.jsonl \
        --eval-out eval/eval_data.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import structlog

log = structlog.get_logger("forgelm.decontaminate")


def _ngrams(text: str, n: int = 8) -> set[tuple]:
    words = text.lower().split()
    return set(zip(*[words[i:] for i in range(n)])) if len(words) >= n else set()


def build_train_ngrams(train_files: list[Path], n: int = 8) -> set[tuple]:
    """Collect all n-grams from all training files."""
    grams: set[tuple] = set()
    for path in train_files:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            # Collect text from all string fields
            text = " ".join(
                str(v) for v in _iter_strings(obj)
            )
            grams |= _ngrams(text, n)
    log.info("train_ngrams_built", n_grams=len(grams), n_files=len(train_files))
    return grams


def _iter_strings(obj):
    """Recursively yield all string values from a JSON object."""
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _iter_strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _iter_strings(v)


def decontaminate(train_files: list[Path], eval_in: Path,
                  eval_out: Path, n: int = 8) -> dict:
    """Filter eval_in, write clean samples to eval_out. Returns stats."""
    train_grams = build_train_ngrams(train_files, n)
    raw = [json.loads(line) for line in eval_in.read_text().splitlines() if line.strip()]

    clean, flagged = [], []
    for sample in raw:
        text = " ".join(str(v) for v in _iter_strings(sample))
        sample_grams = _ngrams(text, n)
        overlap = sample_grams & train_grams
        if overlap:
            flagged.append({
                "sample": sample,
                "overlapping_grams": len(overlap),
            })
        else:
            clean.append(sample)

    eval_out.parent.mkdir(parents=True, exist_ok=True)
    with open(eval_out, "w") as f:
        for s in clean:
            f.write(json.dumps(s) + "\n")

    stats = {
        "n_raw": len(raw),
        "n_clean": len(clean),
        "n_flagged": len(flagged),
        "contamination_rate": round(len(flagged) / max(len(raw), 1), 4),
        "n_gram_size": n,
    }
    log.info("decontamination_done", **stats)

    # Write flagged log for audit
    flagged_path = eval_out.parent / "decontam_flagged.jsonl"
    with open(flagged_path, "w") as f:
        for item in flagged:
            f.write(json.dumps(item) + "\n")
    log.info("flagged_log_written", path=str(flagged_path))
    return stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-files", nargs="+", type=Path,
                   default=[Path("data/sft/sft_toolcall.jsonl"),
                            Path("data/events/sft_events.jsonl")])
    p.add_argument("--eval-in",  type=Path, default=Path("eval/eval_data_raw.jsonl"))
    p.add_argument("--eval-out", type=Path, default=Path("eval/eval_data.jsonl"))
    p.add_argument("--n", type=int, default=8)
    args = p.parse_args()
    stats = decontaminate(args.train_files, args.eval_in, args.eval_out, args.n)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
