# CLAUDE.md — ForgeLM

## What this repo is

A 3-stage LLM training pipeline (DAPT → SFT → DPO) that domain-adapts
Qwen2.5-1.5B for financial hypothesis research. Integrates with AlphaForge
as a drop-in Claude replacement via `serving/alphaforge_adapter.py`.

## Commands

```bash
pip install -e ".[dev]"                          # CPU-only, for tests
pip install -e ".[pipeline]"                     # + data pipeline deps (no GPU)
pip install -e ".[train]"                        # + GPU training deps
pytest tests/                                    # offline tests (no API, no GPU)
ruff check .                                     # lint

# Data pipeline (requires ANTHROPIC_API_KEY)
python -m forgelm.data_pipeline.download_edgar --tickers AAPL MSFT GOOGL
python -m forgelm.data_pipeline.gen_toolcall_data --n-target 500 --seed 42
python -m forgelm.data_pipeline.gen_dpo_pairs --n-pairs 1500 --seed 42
python -m forgelm.data_pipeline.gen_sft_events --n-target 200 --seed 42

# Training (GPU required — run on Colab/Kaggle A100/T4)
python -m forgelm.training.stage1_dapt --corpus data/corpus --seed 42
python -m forgelm.training.stage2_sft --seed 42
python -m forgelm.training.stage3_dpo --seed 42

# Eval
python -m forgelm.eval.run_eval --model checkpoints/stage3_dpo/final --seeds
```

## Conventions

- All randomness seeded. 3-seed runs: 42, 123, 2024.
- ALL generated training data passes `shared_schemas.tool_schemas.HypothesisToolCall` before being written.
- DPO: 4 error families (wrong_ticker, missing_correction, hallucinated_date, vague_hypothesis).
- Eval: McNemar p-value required for M1/M3 comparisons.
- Numbers in README come from real runs only.
