# ForgeLM — Domain-Adapted Financial Language Model

**A 3-stage training pipeline that fine-tunes Qwen2.5-1.5B for financial
market-research workflows.** Integrates as a drop-in backend inside the
[AlphaForge](../alphaforge/) multi-agent research system.

## Architecture

```
EDGAR corpus (~25M tok)
        │
        ▼
  Stage 1: DAPT          Qwen2.5-1.5B + QLoRA NF4/BF16
  (overnight, 1 epoch)   domain-adapts to financial language
        │
        ▼
  Stage 2: SFT           tool-call sequences (70%) + event extraction (30%)
  (2 epochs)             every sample validated by shared_schemas
        │
        ▼
  Stage 3: DPO           1.5k preference pairs, 4 error families, β=0.1
  (1 epoch)              teaches the model to refuse hallucinated tickers,
                         missing corrections, out-of-range dates
        │
        ▼
  vllm serving           OpenAI-compatible endpoint
        │
        ▼
  AlphaForge adapter     drop-in replacement for Claude backend
```

## Key Design Decisions

| Decision | Rationale |
|---|---|
| Shared Pydantic schemas (`shared_schemas/`) | Training data and live inference go through the SAME validator — no special-casing |
| Correction-required validator in schema | Schema rejects any multi-hypothesis sample that omits Bonferroni/BH-FDR |
| DPO over RLHF | Lower variance, no reward-model training, directly optimises log-ratio |
| McNemar's test in eval | Paired comparison with base model; 95% CI on each metric |
| 3-seed runs (42, 123, 2024) | All metrics reported with mean ± std across seeds |

## Setup

```bash
# CPU-only (data pipeline + tests)
pip install -e ".[dev,pipeline]"

# GPU training (run on Colab/Kaggle A100 or 4×T4)
pip install -e ".[train]"
```

## Reproducing the Data Pipeline

```bash
# 1. Download EDGAR filings (set EDGAR_USER_AGENT env var first)
export EDGAR_USER_AGENT="YourName/1.0 (your@email.com)"
python -m forgelm.data_pipeline.download_edgar \
    --tickers AAPL MSFT GOOGL JPM XOM AMD NVDA BAC GS META \
    --out data/corpus --max-filings 20

# 2. Generate validated tool-call SFT data (requires ANTHROPIC_API_KEY)
export ANTHROPIC_API_KEY=sk-...
python -m forgelm.data_pipeline.gen_toolcall_data \
    --n-target 500 --seed 42 --out data/sft/sft_toolcall.jsonl

# 3. Generate DPO preference pairs (offline, no API)
python -m forgelm.data_pipeline.gen_dpo_pairs \
    --sft data/sft/sft_toolcall.jsonl \
    --n-pairs 1500 --seed 42

# 4. Generate event-extraction SFT data
python -m forgelm.data_pipeline.gen_sft_events \
    --n-target 200 --seed 42
```

## Training

```bash
# Stage 1 — DAPT (~6–8h on A100)
python -m forgelm.training.stage1_dapt \
    --corpus data/corpus --seed 42

# Stage 2 — SFT (~2h on A100)
python -m forgelm.training.stage2_sft \
    --stage1-dir checkpoints/stage1_dapt/final --seed 42

# Stage 3 — DPO (~1h on A100)
python -m forgelm.training.stage3_dpo \
    --stage2-dir checkpoints/stage2_sft/final --seed 42
```

## Evaluation

```bash
# Single seed
python -m forgelm.eval.run_eval \
    --model checkpoints/stage3_dpo/final \
    --base-model Qwen/Qwen2.5-1.5B-Instruct \
    --n-eval 200 --seed 42

# 3-seed stability run (spec requirement)
python -m forgelm.eval.run_eval \
    --model checkpoints/stage3_dpo/final \
    --base-model Qwen/Qwen2.5-1.5B-Instruct \
    --n-eval 200 --seeds \
    --out eval/results.json
```

## Eval Results

> [!NOTE]
> Numbers below will be filled after GPU training completes.
> Run `python -m forgelm.eval.run_eval --seeds` to reproduce.

| Metric | ForgeLM (mean ± std) | Base Qwen2.5-1.5B | McNemar p |
|---|---|---|---|
| M1 tool_call_accuracy | TBD | TBD | TBD |
| M2 schema_validity_rate | TBD | TBD | — |
| M3 correction_recall | TBD | TBD | TBD |
| M4 hallucination_rate | TBD | TBD | — |

## AlphaForge Integration

ForgeLM serves as a drop-in replacement for the Claude backend inside AlphaForge:

```bash
# 1. Start ForgeLM server
python -m forgelm.serving.alphaforge_adapter \
    --model checkpoints/stage3_dpo/final --port 8000

# 2. Run AlphaForge with ForgeLM backend
# (alphaforge_adapter.py is a full ClaudeClient-compatible interface)
```

The adapter enforces the **same** `HypothesisToolCall` schema validation as
AlphaForge's Claude backend — no special-casing, same honesty guarantees.

## Tests

```bash
# Runs fully offline — no GPU, no API key needed
pytest tests/ -v
```

Tests cover:
- Schema validation (universe membership, date ranges, correction requirement)
- DPO injection logic (all 4 error families)
- Eval scoring functions (M1-M4, McNemar, bootstrap CI)
