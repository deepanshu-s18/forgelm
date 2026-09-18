"""Shared training utilities: QLoRA config, dtype selection, reproducibility.

These are imported by all three training stages so the config lives in one
place and is trivially auditable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    pass

log = structlog.get_logger("forgelm.training")

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
HF_REPO_PREFIX = "deepanshusingh/forgelm"  # HuggingFace Hub upload target

# LoRA / QLoRA hyperparameters
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj"]

# Quantization: NF4, double-quant, BF16 compute
BNBCONFIG = {
    "load_in_4bit": True,
    "bnb_4bit_use_double_quant": True,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_compute_dtype": "bfloat16",
}


@dataclass
class StageConfig:
    """Runtime config for a training stage — CLI overridable."""
    stage: int = 1
    seed: int = 42
    base_model: str = BASE_MODEL
    output_dir: Path = Path("checkpoints")
    batch_size: int = 4
    grad_accum: int = 8
    lr: float = 2e-4
    num_epochs: int = 1
    max_seq_len: int = 2048
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    bf16: bool = True
    gradient_checkpointing: bool = True
    push_to_hub: bool = False
    hub_model_id: str = ""
    seeds: list[int] = field(default_factory=lambda: [42, 123, 2024])


def set_seed(seed: int) -> None:
    import random

    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_qlora_config():
    """Build a BitsAndBytesConfig + LoraConfig for QLoRA training."""
    import torch
    from peft import LoraConfig, TaskType
    from transformers import BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=BNBCONFIG["load_in_4bit"],
        bnb_4bit_use_double_quant=BNBCONFIG["bnb_4bit_use_double_quant"],
        bnb_4bit_quant_type=BNBCONFIG["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    return bnb_config, lora_config


def load_base_model(model_id: str, bnb_config):
    """Load the base model with QLoRA quantization."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    return model, tokenizer


def merge_and_save_adapter(base_model_path: str, adapter_path: str, output_dir: Path | str) -> str:
    """Merge a LoRA adapter into base weights and save as a standalone model.

    Solves the PEFT chaining problem across multi-stage training (DAPT -> SFT -> DPO)
    by ensuring each downstream stage loads a unified model rather than stacking
    multiple unmerged adapters over a frozen 4-bit base.
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    log.info(
        "merging_adapter",
        base=str(base_model_path),
        adapter=str(adapter_path),
        out=str(out_path),
    )
    # Load unquantized in float16/bfloat16 to merge losslessly
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    base = AutoModelForCausalLM.from_pretrained(
        str(base_model_path),
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, str(adapter_path))
    merged = model.merge_and_unload()
    merged.save_pretrained(str(out_path))

    tokenizer = AutoTokenizer.from_pretrained(str(adapter_path), trust_remote_code=True)
    tokenizer.save_pretrained(str(out_path))
    log.info("merge_complete", out=str(out_path))
    return str(out_path)
