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
