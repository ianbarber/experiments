"""Model loading and small shared helpers."""

import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "LiquidAI/LFM3-1.7B"
MODEL_REVISION = "1da06bec6a819da10b54c67e5c90972b46d81871"
CODE_DIR = Path(__file__).resolve().parent.parent
RUNS_DIR = Path(os.environ.get("DELTUNE_RUNS", CODE_DIR / "runs"))


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_tokenizer():
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    tok.padding_side = "left"
    return tok


def load_model(dtype=torch.bfloat16, device="cuda"):
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=MODEL_REVISION, dtype=dtype)
    # plain greedy decoding everywhere; the shipped config samples with a repetition penalty
    gc = model.generation_config
    gc.do_sample, gc.repetition_penalty, gc.temperature, gc.top_k, gc.top_p = False, 1.0, None, None, None
    return model.to(device) if device != "cpu" else model


def gate_modules(model):
    """The SwiGLU gate projection of every block: w2(silu(w1(x)) * w3(x)), so w1."""
    return [(f"model.layers.{i}.feed_forward.w1", layer.feed_forward.w1) for i, layer in enumerate(model.model.layers)]


def dump_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=str))
