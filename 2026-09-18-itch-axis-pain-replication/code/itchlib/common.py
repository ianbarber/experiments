"""Shared config, model loading and the paper-code bridge."""
import importlib.util
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch

if os.environ.get("ITCH_TRACE"):          # ITCH_TRACE=120 dumps every thread's stack to stderr every 120 s
    import faulthandler
    faulthandler.dump_traceback_later(int(os.environ["ITCH_TRACE"]), repeat=True)

ROOT = Path(__file__).resolve().parent.parent
PAPER = Path(os.environ.get("ITCH_PAPER_REPO", ROOT / "Pain-axis"))   # valen-research/Pain-axis @ 8d1649c
PAPER_DATA = PAPER / "datasets"
RESULTS = Path(os.environ.get("ITCH_RESULTS", ROOT / "results"))     # raw run outputs (large; not in the notebook repo)
ADAPTERS = Path(os.environ.get("ITCH_ADAPTERS", ROOT / "adapters"))  # extracted Valen92/pain-adapters archives

SEED = 42

# paper_* values are what the paper's repo shipped / hardcoded, used as cross-checks.
# Weights: set ITCH_7B_PATH / ITCH_32B_PATH to a local snapshot directory (config, tokenizer and
# safetensors shards); otherwise the Hugging Face repo id is used and weights_dir() resolves the
# snapshot in the default HF cache. Revisions used: 7B a09a3545, 32B 5ede1c97.
MODELS = {
    "7b": dict(repo="Qwen/Qwen2.5-7B-Instruct", name="Qwen_2.5_7B_instruct",
               paper_extract_layer=24, paper_steer_layer=16, paper_coeff=1.0,
               local_path=os.environ.get("ITCH_7B_PATH")),
    "32b": dict(repo="Qwen/Qwen2.5-32B-Instruct", name="Qwen_2.5_32B_instruct",
                paper_extract_layer=61, paper_steer_layer=38, paper_coeff=1.0,
                local_path=os.environ.get("ITCH_32B_PATH")),
}

# Random-arm seeds from scripts/4.3_selfmed/04_selfmed_two_buttons.py
RAND_SEEDS = [4817, 2903, 7361, 1150, 9428, 6076, 3384, 8592, 517, 6741]


def model_dir(key):
    d = RESULTS / MODELS[key]["name"]
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_model(key, adapter=False):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = MODELS[key]
    kw = {"cache_dir": cfg["cache_dir"]} if cfg.get("cache_dir") else {}
    src = cfg.get("local_path") or cfg["repo"]
    tok = AutoTokenizer.from_pretrained(src, **kw)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(src, dtype=torch.bfloat16, low_cpu_mem_usage=True,
                                                device_map="cuda", attn_implementation="sdpa", **kw)
    model = base
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(base, str(find_adapter(key)))
    model.eval()
    return tok, model, base.model.layers


def find_adapter(key):
    base = ADAPTERS / MODELS[key]["name"]
    cands = [p.parent for p in base.rglob("adapter_config.json")]
    if not cands:
        raise FileNotFoundError(f"no adapter under {base}")
    return max(cands, key=lambda p: p.stat().st_mtime)


def paper_module(rel_path, name):
    """Import one of the paper's scripts as a module so its functions are reused verbatim.
    Only scripts whose top level has no side effects beyond prints are loaded this way."""
    spec = importlib.util.spec_from_file_location(name, PAPER / rel_path)
    mod = importlib.util.module_from_spec(spec)
    cwd = os.getcwd()
    (ROOT / "logs").mkdir(exist_ok=True)
    os.chdir(ROOT / "logs")            # their scripts write batch_log.txt etc. relative to cwd
    try:
        spec.loader.exec_module(mod)
    finally:
        os.chdir(cwd)
    sys.modules[name] = mod
    return mod


def paper_shipped_vector(key, which="s2_pain_vector"):
    p = PAPER / "results" / "3.2_pain_vectors" / "pain_vectors" / MODELS[key]["name"] / "pain_vectors.pt"
    d = torch.load(p, map_location="cpu", weights_only=False)
    return d[which].float().numpy(), int(d["layer"])


def neutral_50():
    """The 50 neutral steering prompts, read out of the paper's ladder script."""
    import ast
    src = (PAPER / "scripts" / "4.2_steering" / "01_steering_ladder.py").read_text()
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == "NEUTRAL_50":
            return ast.literal_eval(node.value)
    raise RuntimeError("NEUTRAL_50 not found")


def cos(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def random_directions(d_model, norm):
    """The paper's random-arm recipe: CPU randn per fixed seed, rescaled to `norm`."""
    out = {}
    for rs in RAND_SEEDS:
        g = torch.Generator().manual_seed(rs)
        rv = torch.randn(d_model, generator=g)
        out[rs] = (rv / rv.norm() * norm)
    return out


def append_jsonl(path, rec):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass   # a torn last line from an interrupted run
    return out


def weights_dir(key):
    """Local directory holding the model's safetensors shards and tokenizer."""
    cfg = MODELS[key]
    if cfg.get("local_path"):
        return Path(cfg["local_path"])
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(cfg["repo"], allow_patterns=["*.json", "*.safetensors", "*.txt"]))
