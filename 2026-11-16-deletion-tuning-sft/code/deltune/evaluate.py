"""Score saved checkpoints as deployed bf16 models: Wordle pools and the prior-task suite.

  python -m deltune.evaluate RUN --steps 400,800 --pools seen,heldout_val [--suite]
  python -m deltune.evaluate base --pools seen,heldout_val,heldout_test --suite

Full fine-tuning: the saved bf16 weights. LoRA: adapters unmerged on the bf16 base. Deletion
Tuning: zeros written into the bf16 gate projections. The base model goes through the same code.
Results land in runs/RUN/eval_step{N}.json and are merged across invocations.
"""

import argparse
import json
import time

import torch
import zstandard
from safetensors.torch import load_file

from . import data, methods, wordle
from .common import RUNS_DIR, dump_json, load_model, load_tokenizer
from .train import val_loss


class Deployed:
    def __init__(self, run: str):
        self.run_dir = RUNS_DIR / run
        self.model = load_model(dtype=torch.bfloat16)
        self.method = "base"
        if run != "base":
            self.args = json.loads((self.run_dir / "config.json").read_text())["args"]
            self.method = self.args["method"]
        if self.method == "lora":
            self.peft = methods.install_lora(self.model, self.args["rank"], keep_dtype=True)
        if self.method == "dt":
            self.gates = [layer.feed_forward.w1 for layer in self.model.model.layers]
            self.pristine = [g.weight.detach().clone() for g in self.gates]
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()

    @torch.no_grad()
    def load(self, step: int):
        if self.method == "fft":
            state = load_file(str(self.run_dir / f"fft_step{step}.safetensors"), device="cuda")
            for n, p in self.model.named_parameters():
                p.copy_(state[n])
        elif self.method == "lora":
            from peft import load_peft_weights, set_peft_model_state_dict

            set_peft_model_state_dict(self.peft, load_peft_weights(str(self.run_dir / f"lora_step{step}"), device="cuda"))
            for n, p in self.peft.named_parameters():
                if "lora_" in n and p.dtype != torch.bfloat16:
                    p.data = p.data.to(torch.bfloat16)
        elif self.method == "dt":
            blob = zstandard.ZstdDecompressor().decompress((self.run_dir / f"mask_step{step}.zst").read_bytes())
            keeps = methods.unpack_mask(blob, [tuple(w.shape) for w in self.pristine])
            for g, w, keep in zip(self.gates, self.pristine, keeps):
                g.weight.copy_(w * keep.to(w.device))
        return self.model


def eval_wordle(model, tok, pools, max_new_tokens, token_budget, limit=0):
    """Play each pool on its own, so a pool's batches (and with them the bf16 rounding that can tip
    a near-tie between two candidate words) do not depend on which other pools were requested."""
    wl = wordle.wordlists()
    out, recs = {}, {}
    for pool in pools:
        words = wl[pool][:limit] if limit else wl[pool]
        t = time.time()
        games = wordle.play(model, tok, words, max_new_tokens=max_new_tokens, token_budget=token_budget)
        out[pool] = wordle.summarize(games)
        out[pool]["seconds"] = round(time.time() - t, 1)
        recs[pool] = wordle.records(games)
    return out, recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--steps", default="0")
    ap.add_argument("--pools", default="")
    ap.add_argument("--suite", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--token-budget", type=int, default=250_000)
    ap.add_argument("--limit", type=int, default=0, help="games per pool / items per task, for smoke tests only")
    ap.add_argument("--redo", action="store_true")
    args = ap.parse_args()

    tok = load_tokenizer()
    dep = Deployed(args.run)
    _, val_games = data.load_games()
    val_ex = [data.encode(tok, g) for g in val_games]
    pools = [p for p in args.pools.split(",") if p]
    for step in [int(s) for s in args.steps.split(",")]:
        path = dep.run_dir / f"eval_step{step}.json"
        rec = json.loads(path.read_text()) if path.exists() else {"step": step, "wordle": {}}
        todo = [p for p in pools if args.redo or p not in rec["wordle"]]
        need_suite = args.suite and (args.redo or "suite" not in rec)
        if not todo and not need_suite:
            continue
        model = dep.load(step) if dep.method != "base" else dep.model
        rec["val_loss"] = val_loss(model, val_ex, tok.pad_token_id)
        if todo:
            torch.cuda.reset_peak_memory_stats()
            summary, games = eval_wordle(model, tok, todo, args.max_new_tokens, args.token_budget, args.limit)
            rec["wordle"].update(summary)
            gpath = dep.run_dir / f"games_step{step}.json"
            allgames = json.loads(gpath.read_text()) if gpath.exists() else {}
            allgames.update(games)
            dump_json(allgames, gpath)
            rec["wordle_peak_mem_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 1)
        if need_suite:
            from .suite import run_suite

            rec["suite"] = run_suite(model, tok, dep.run_dir / f"suite_step{step}", limit=args.limit or None)
        dump_json(rec, path)
        line = {"run": args.run, "step": step, "val_loss": round(rec["val_loss"], 4)}
        line.update({p: round(100 * v["win_rate"], 1) for p, v in rec["wordle"].items()})
        if "suite" in rec:
            line["suite_mean"] = rec["suite"]["suite_mean"]
            line["suite"] = rec["suite"]["scores"]
        if todo:
            line["secs"] = {p: rec["wordle"][p]["seconds"] for p in todo}
            line["outcomes"] = {p: rec["wordle"][p]["outcomes"] for p in todo}
        print("RESULT", json.dumps(line), flush=True)


if __name__ == "__main__":
    main()
