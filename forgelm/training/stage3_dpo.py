"""Stage 3 — Direct Preference Optimisation (DPO).

Fine-tunes the SFT checkpoint on the DPO preference pairs (chosen vs
rejected tool calls) using TRL DPOTrainer with beta=0.1.

Usage (GPU required):
    python -m forgelm.training.stage3_dpo \
        --stage2-dir checkpoints/stage2_sft/final \
        --dpo-pairs data/dpo/dpo_pairs.jsonl \
        --output-dir checkpoints \
        --seed 42
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import structlog

from forgelm.training.common import StageConfig, get_qlora_config, set_seed

log = structlog.get_logger("forgelm.stage3")

DPO_BETA = 0.1
DPO_LR = 5e-5


def load_dpo_dataset(pairs_path: Path, tokenizer, seed: int = 42):
    import random

    from datasets import Dataset

    pairs = [json.loads(line) for line in pairs_path.read_text().splitlines() if line.strip()]
    random.Random(seed).shuffle(pairs)

    # DPO format: prompt, chosen, rejected
    records = [
        {
            "prompt": p["prompt"],
            "chosen": p["chosen"]["content"],
            "rejected": p["rejected"]["content"],
        }
        for p in pairs
    ]
    ds = Dataset.from_list(records)
    log.info("dpo_dataset", n=len(ds))
    return ds


def train(cfg: StageConfig, stage2_dir: Path, pairs_path: Path) -> dict:
    from peft import PeftModel, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    set_seed(cfg.seed)
    bnb_config, lora_config = get_qlora_config()

    tokenizer = AutoTokenizer.from_pretrained(str(stage2_dir), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Reference model (frozen) — the SFT checkpoint
    ref_model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model, quantization_config=bnb_config, device_map="auto",
        trust_remote_code=True,
    )
    ref_model = PeftModel.from_pretrained(ref_model, str(stage2_dir))

    # Policy model (trainable)
    policy = AutoModelForCausalLM.from_pretrained(
        cfg.base_model, quantization_config=bnb_config, device_map="auto",
        trust_remote_code=True,
    )
    policy = PeftModel.from_pretrained(policy, str(stage2_dir))
    policy = prepare_model_for_kbit_training(policy, use_gradient_checkpointing=cfg.gradient_checkpointing)  # noqa: E501

    dataset = load_dpo_dataset(pairs_path, tokenizer, seed=cfg.seed)
    out_dir = cfg.output_dir / "stage3_dpo"
    out_dir.mkdir(parents=True, exist_ok=True)

    dpo_cfg = DPOConfig(
        output_dir=str(out_dir),
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=DPO_LR,
        bf16=cfg.bf16,
        warmup_ratio=cfg.warmup_ratio,
        max_length=cfg.max_seq_len,
        max_prompt_length=512,
        beta=DPO_BETA,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        seed=cfg.seed,
        report_to="none",
    )
    trainer = DPOTrainer(
        model=policy,
        ref_model=ref_model,
        args=dpo_cfg,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )
    log.info("dpo_train_start", n=len(dataset), beta=DPO_BETA, seed=cfg.seed)
    result = trainer.train()
    trainer.save_model(str(out_dir / "final"))
    tokenizer.save_pretrained(str(out_dir / "final"))

    metrics = {
        "stage": 3,
        "seed": cfg.seed,
        "train_loss": round(result.training_loss, 4),
        "train_steps": result.global_step,
        "n_dpo_pairs": len(dataset),
        "beta": DPO_BETA,
        "checkpoint": str(out_dir / "final"),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    log.info("dpo_done", **metrics)
    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage2-dir", type=Path, default=Path("checkpoints/stage2_sft/final"))
    p.add_argument("--dpo-pairs", type=Path, default=Path("data/dpo/dpo_pairs.jsonl"))
    p.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=1)
    args = p.parse_args()
    cfg = StageConfig(stage=3, seed=args.seed, output_dir=args.output_dir,
                      lr=DPO_LR, num_epochs=args.epochs)
    metrics = train(cfg, args.stage2_dir, args.dpo_pairs)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
