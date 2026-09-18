"""Stage 2 — Supervised Fine-Tuning (SFT).

Fine-tunes the DAPT checkpoint on:
    - sft_toolcall.jsonl (70%) — tool-call sequences
    - sft_events.jsonl   (30%) — event extraction

Uses TRL SFTTrainer with QLoRA. Inherits the DAPT LoRA weights as the
starting point (loading the PEFT adapter from stage1 checkpoint).

Usage (GPU required):
    python -m forgelm.training.stage2_sft \
        --stage1-dir checkpoints/stage1_dapt/final \
        --sft-toolcall data/sft/sft_toolcall.jsonl \
        --sft-events data/events/sft_events.jsonl \
        --output-dir checkpoints \
        --seed 42
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import structlog

from forgelm.training.common import (
    StageConfig,
    get_qlora_config,
    set_seed,
)

log = structlog.get_logger("forgelm.stage2")

TOOLCALL_FRACTION = 0.70
SYSTEM_PREFIX = "You are a financial research agent with access to market data and statistical tools."  # noqa: E501


def _format_toolcall(sample: dict, tokenizer) -> dict:
    """Convert a HypothesisToolCall sample to the chat format."""
    messages = [
        {"role": "system", "content": SYSTEM_PREFIX},
        {"role": "user", "content": sample["user_query"]},
        {"role": "assistant", "content": json.dumps({
            "tool_calls": sample["tool_calls"],
            "n_hypotheses": sample["n_hypotheses"],
        })},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": text}


def _format_event(sample: dict, tokenizer) -> dict:
    """Convert an EventRecord sample to the chat format."""
    messages = [
        {"role": "system", "content": "Extract structured financial event data from earnings text."},  # noqa: E501
        {"role": "user", "content": sample["input"]},
        {"role": "assistant", "content": json.dumps(sample["output"])},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": text}


def load_sft_dataset(toolcall_path: Path, events_path: Path, tokenizer, seed: int = 42):
    """Load, format, shuffle, and merge the two SFT sources."""
    from datasets import Dataset, concatenate_datasets

    rng = random.Random(seed)

    tc_raw = [json.loads(line) for line in toolcall_path.read_text().splitlines() if line.strip()]
    ev_raw = [json.loads(line) for line in events_path.read_text().splitlines() if line.strip()]

    # 70/30 split by count
    n_tc = int((len(tc_raw) + len(ev_raw)) * TOOLCALL_FRACTION)
    n_ev = len(tc_raw) + len(ev_raw) - n_tc
    rng.shuffle(tc_raw)
    rng.shuffle(ev_raw)
    tc_raw = tc_raw[:n_tc]
    ev_raw = ev_raw[:n_ev]

    tc_ds = Dataset.from_list([_format_toolcall(s, tokenizer) for s in tc_raw])
    ev_ds = Dataset.from_list([_format_event(s, tokenizer) for s in ev_raw])
    combined = concatenate_datasets([tc_ds, ev_ds]).shuffle(seed=seed)
    log.info("sft_dataset", toolcall=len(tc_raw), events=len(ev_raw), total=len(combined))
    return combined


def train(cfg: StageConfig, stage1_dir: Path,
          toolcall_path: Path, events_path: Path) -> dict:
    from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    set_seed(cfg.seed)
    bnb_config, lora_config = get_qlora_config()

    # Load tokenizer from stage1 (preserves any special tokens added)
    tokenizer = AutoTokenizer.from_pretrained(str(stage1_dir), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model (either merged model directly or base + adapter)
    adapter_cfg = stage1_dir / "adapter_config.json"
    if adapter_cfg.exists():
        log.info("loading_peft_adapter", adapter_dir=str(stage1_dir))
        base = AutoModelForCausalLM.from_pretrained(
            cfg.base_model, quantization_config=bnb_config, device_map="auto",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base, str(stage1_dir))
    else:
        log.info("loading_merged_base", model_dir=str(stage1_dir))
        model = AutoModelForCausalLM.from_pretrained(
            str(stage1_dir), quantization_config=bnb_config, device_map="auto",
            trust_remote_code=True,
        )
        model = get_peft_model(model, lora_config)

    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=cfg.gradient_checkpointing
    )

    dataset = load_sft_dataset(toolcall_path, events_path, tokenizer, seed=cfg.seed)
    out_dir = cfg.output_dir / "stage2_sft"
    out_dir.mkdir(parents=True, exist_ok=True)

    sft_cfg = SFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.lr,
        bf16=cfg.bf16,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        max_seq_length=cfg.max_seq_len,
        logging_steps=20,
        save_steps=200,
        save_total_limit=2,
        seed=cfg.seed,
        report_to="none",
        dataset_text_field="text",
    )
    trainer = SFTTrainer(model=model, args=sft_cfg, train_dataset=dataset)
    log.info("sft_train_start", n=len(dataset), seed=cfg.seed, max_seq_len=cfg.max_seq_len)
    result = trainer.train()
    trainer.save_model(str(out_dir / "final"))
    tokenizer.save_pretrained(str(out_dir / "final"))

    metrics = {
        "stage": 2,
        "seed": cfg.seed,
        "train_loss": round(result.training_loss, 4),
        "train_steps": result.global_step,
        "n_sft_samples": len(dataset),
        "toolcall_fraction": TOOLCALL_FRACTION,
        "checkpoint": str(out_dir / "final"),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    log.info("sft_done", **metrics)
    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage1-dir", type=Path, default=Path("checkpoints/stage1_dapt/final"))
    p.add_argument("--sft-toolcall", type=Path, default=Path("data/sft/sft_toolcall.jsonl"))
    p.add_argument("--sft-events", type=Path, default=Path("data/events/sft_events.jsonl"))
    p.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--max-seq-len", type=int, default=512)
    args = p.parse_args()
    cfg = StageConfig(
        stage=2, seed=args.seed, output_dir=args.output_dir,
        lr=args.lr, num_epochs=args.epochs, max_seq_len=args.max_seq_len,
    )
    metrics = train(cfg, args.stage1_dir, args.sft_toolcall, args.sft_events)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
