"""Stage 1 — Domain-Adaptive Pre-Training (DAPT).

Trains Qwen2.5-1.5B on ~25M tokens of cleaned EDGAR corpus (MD&A +
risk-factor sections) using QLoRA NF4 + BF16 for ~1 epoch overnight on
a single A100 or overnight on 4×T4.

This stage adapts the model's vocabulary distribution toward financial
language before any task-specific fine-tuning.

Usage (GPU required):
    python -m forgelm.training.stage1_dapt \
        --corpus data/corpus \
        --output-dir checkpoints/stage1_dapt \
        --seed 42
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import structlog

from forgelm.training.common import StageConfig, get_qlora_config, load_base_model, set_seed

log = structlog.get_logger("forgelm.stage1")


def load_corpus(corpus_dir: Path) -> list[str]:
    """Load all *.txt chunks from the corpus directory."""
    files = sorted(corpus_dir.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No .txt files in {corpus_dir}")
    texts = [f.read_text().strip() for f in files if f.stat().st_size > 100]
    log.info("corpus_loaded", n_chunks=len(texts), total_tokens=sum(len(t.split()) for t in texts))
    return texts


def build_dataset(texts: list[str], tokenizer, max_length: int = 2048):
    """Tokenize the corpus into fixed-length blocks for causal LM pre-training."""
    from datasets import Dataset

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length,
                         padding=False, return_special_tokens_mask=True)

    raw = Dataset.from_dict({"text": texts})
    tokenized = raw.map(tokenize, batched=True, remove_columns=["text"])
    log.info("dataset_built", n_examples=len(tokenized))
    return tokenized


def train(cfg: StageConfig, corpus_dir: Path) -> dict:
    from peft import get_peft_model, prepare_model_for_kbit_training
    from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments

    set_seed(cfg.seed)
    bnb_config, lora_config = get_qlora_config()
    model, tokenizer = load_base_model(cfg.base_model, bnb_config)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=cfg.gradient_checkpointing)  # noqa: E501
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    texts = load_corpus(corpus_dir)
    dataset = build_dataset(texts, tokenizer, max_length=cfg.max_seq_len)

    out_dir = cfg.output_dir / "stage1_dapt"
    out_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.lr,
        bf16=cfg.bf16,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        logging_steps=50,
        save_steps=500,
        save_total_limit=2,
        seed=cfg.seed,
        dataloader_num_workers=2,
        report_to="none",
    )
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )
    log.info("dapt_train_start", model=cfg.base_model, corpus_chunks=len(texts))
    result = trainer.train()
    trainer.save_model(str(out_dir / "final"))
    tokenizer.save_pretrained(str(out_dir / "final"))

    metrics = {
        "stage": 1,
        "seed": cfg.seed,
        "train_loss": round(result.training_loss, 4),
        "train_steps": result.global_step,
        "n_corpus_chunks": len(texts),
        "checkpoint": str(out_dir / "final"),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    log.info("dapt_done", **metrics)
    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", type=Path, default=Path("data/corpus"))
    p.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=4)
    args = p.parse_args()
    cfg = StageConfig(
        stage=1, seed=args.seed, output_dir=args.output_dir,
        lr=args.lr, num_epochs=args.epochs, batch_size=args.batch_size,
    )
    metrics = train(cfg, args.corpus)
    import json
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
