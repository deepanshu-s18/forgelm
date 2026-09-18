# Kaggle Setup Guide — ForgeLM Training

Step-by-step to go from local data → trained model on Kaggle in one session.

---

## Step 1 — API Key / Offline Setup (~2 min)

ForgeLM supports three data generation routes:
- **Option A (Google Gemini, Free)**: Get a free key at [aistudio.google.com](https://aistudio.google.com), run `export GEMINI_API_KEY=...`
- **Option B (Anthropic Claude)**: Go to [console.anthropic.com](https://console.anthropic.com), copy key, run `export ANTHROPIC_API_KEY=...`
- **Option C (Fully Offline / Pre-generated)**: Skip generation entirely — the repo already includes full pre-generated, schema-validated training and eval datasets in `data/sft/`, `data/dpo/`, and `eval/`!

---

## Step 2 — Data Pipeline Overview

If regenerating data from scratch:

```bash
cd /Users/deepanshusingh/Desktop/Tower/forgelm

# 2a. Download EDGAR filings (free, ~30 min)
export EDGAR_USER_AGENT="Deepanshu Singh/1.0 (your@email.com)"
python -m forgelm.data_pipeline.download_edgar \
    --tickers AAPL MSFT GOOGL NVDA AMD JPM GS BAC META XOM CVX JNJ PFE UNH NFLX \
    --out data/corpus --max-filings 20

# 2b. Generate SFT tool-call data (supports Gemini or Claude or programmatic fallback)
python -m forgelm.data_pipeline.gen_toolcall_data \
    --n-target 500 --seed 42 --out data/sft/sft_toolcall.jsonl

# 2c. Generate event-extraction SFT data
python -m forgelm.data_pipeline.gen_sft_events \
    --n-target 200 --seed 42 --out data/events/sft_events.jsonl

# 2d. Generate DPO preference pairs (FREE — offline, ~2 min)
python -m forgelm.data_pipeline.gen_dpo_pairs \
    --sft data/sft/sft_toolcall.jsonl \
    --n-pairs 1500 --seed 42 --out data/dpo/dpo_pairs.jsonl

# 2e. Generate held-out eval data (different seed = no contamination)
python -m forgelm.data_pipeline.gen_toolcall_data \
    --n-target 300 --seed 999 --out eval/eval_data_raw.jsonl

# 2f. Decontaminate eval vs training data
python -m forgelm.data_pipeline.decontaminate \
    --train-files data/sft/sft_toolcall.jsonl data/events/sft_events.jsonl \
    --eval-in eval/eval_data_raw.jsonl \
    --eval-out eval/eval_data.jsonl

# Verify counts
echo "Corpus chunks: $(ls data/corpus/*.txt 2>/dev/null | wc -l)"
echo "SFT samples:   $(wc -l < data/sft/sft_toolcall.jsonl)"
echo "DPO pairs:     $(wc -l < data/dpo/dpo_pairs.jsonl)"
echo "Eval examples: $(wc -l < eval/eval_data.jsonl)"
```

---

## Step 3 — Create Kaggle Dataset (~10 min)

1. Go to https://kaggle.com/datasets → **New Dataset**
2. Name it: `forgelm-data`
3. Upload these folders:
   ```
   data/corpus/      (all *.txt files + manifest.json)
   data/sft/         (sft_toolcall.jsonl)
   data/dpo/         (dpo_pairs.jsonl)
   data/events/      (sft_events.jsonl)
   eval/             (eval_data.jsonl)
   ```
4. Set visibility: **Private**
5. Click **Create**

> [!TIP]
> Zip each subfolder first for faster upload:
> `zip -r corpus.zip data/corpus/ && zip sft.zip data/sft/ ...`

---

## Step 4 — Create Kaggle Notebook (~5 min)

1. Go to https://kaggle.com/code → **New Notebook**
2. Upload `notebooks/kaggle_train.ipynb`
3. **Settings** panel (right side):
   - Accelerator: **GPU T4 × 2** ← important
   - Internet: **ON** (needed to download base model from HuggingFace)
4. **Add Data** → search your `forgelm-data` dataset → **Add**
5. **Add Secret** → Name: `ANTHROPIC_API_KEY`, Value: `sk-ant-...`
   (Only needed if you want to run data gen inside the notebook)

> [!IMPORTANT]
> Update **Cell 2** in the notebook:
> ```python
> REPO_URL = 'https://github.com/YOUR_GITHUB/forgelm.git'
> ```
> Replace `YOUR_GITHUB` with your actual GitHub username.

---

## Step 5 — Run the Notebook (~8–10h)

1. Click **Run All** (or Shift+Enter cell by cell)
2. Kaggle auto-saves output to `/kaggle/working/`
3. Training stages run sequentially with automated adapter merging:
   - **Stage 1 DAPT**: ~3h → automatically merged into `/kaggle/working/checkpoints/stage1_merged`
   - **Stage 2 SFT**: ~2h (trained with `max_seq_len=512` on merged DAPT base) → merged into `/kaggle/working/checkpoints/stage2_merged`
   - **Stage 3 DPO**: ~1.5h (policy + reference initialized on clean SFT base)
   - **Eval all 4 checkpoints**: ~1h (evaluates base, dapt, sft, dpo on 300 decontaminated examples)
   - **3-Seed Stability Check**: runs seeds 42, 123, 2024
   - **Stats report**: ~2 min (`python -m forgelm.eval.stats_report` creates `eval/results.md`)

> [!NOTE]
> If you hit the 30h/week GPU limit, save checkpoints after each stage
> and resume in a new session by commenting out completed stages.

---

## Step 6 — Download Results (~10 min)

1. After notebook finishes, go to **Output** tab
2. Download `forgelm_results.zip`
3. Unzip and copy into your local repo:
   ```bash
   unzip forgelm_results.zip -d /tmp/forgelm_results/
   cp /tmp/forgelm_results/results.md \
      /Users/deepanshusingh/Desktop/Tower/forgelm/eval/results.md
   cp /tmp/forgelm_results/eval_all.json \
      /Users/deepanshusingh/Desktop/Tower/forgelm/eval/results.json
   ```
4. Update the README eval table with real numbers
5. Commit: `git commit -m 'feat: real Kaggle training results (DAPT→SFT→DPO)'`

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `CUDA out of memory` | Reduce `--batch-size` to 2, `--grad-accum` to 16 |
| `ModuleNotFoundError: bitsandbytes` | Run: `pip install bitsandbytes --upgrade` |
| HuggingFace rate limit on model download | Add `HF_TOKEN` secret in Kaggle settings |
| Stage 1 loss > 3.0 | Check corpus: `ls data/corpus/*.txt | wc -l` should be > 100 |
| vLLM not available for eval | The eval harness falls back to HuggingFace pipeline automatically |
