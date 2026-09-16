# ForgeLM Eval Results

> [!NOTE]
> This table will be filled after Kaggle training completes.
> Run `python -m forgelm.eval.stats_report` to regenerate from real results.

## Checkpoint Comparison (M1–M4)

| Checkpoint | M1 Schema Validity | M2 Tool-Call Accuracy | M3 Correction Recall | M4 Hallucination ↓ |
|---|---|---|---|---|
| base (Qwen2.5-1.5B) | TBD | TBD | TBD | TBD |
| dapt (+ EDGAR) | TBD | TBD | TBD | TBD |
| sft (+ tool-calls) | TBD | TBD | TBD | TBD |
| **dpo (final)** | TBD | TBD | TBD | TBD |

## Statistical Tests

McNemar's test (dpo vs base, M2): p = TBD
Bootstrap 95% CI on M2 difference: [TBD, TBD]

## 3-Seed Stability (seeds 42, 123, 2024)

| Metric | Mean | Std |
|---|---|---|
| M1 Schema Validity | TBD | TBD |
| M2 Tool-Call Accuracy | TBD | TBD |
| M3 Correction Recall | TBD | TBD |
| M4 Hallucination Rate | TBD | TBD |

## Notes
- All numbers from real model inference on held-out eval_data.jsonl (300 examples, decontaminated).
- Run `python -m forgelm.eval.run_eval --model checkpoints/stage3_dpo/final --seeds` to fill this table.
