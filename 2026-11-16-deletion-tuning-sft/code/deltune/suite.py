"""The prior-task suite: the paper's six benchmarks, run with lm-evaluation-harness on a live model.

Multiple-choice tasks and HumanEval are scored completion-style (no chat template). IFEval goes
through the chat template with thinking disabled. Greedy decoding throughout. `core` is the bf16
deployed model built by deltune.evaluate, or the bf16 base model.
"""

import gzip
import json
import os
import time
from pathlib import Path

from .common import CODE_DIR

os.environ.setdefault("HF_ALLOW_CODE_EVAL", "1")
os.environ.setdefault("NLTK_DATA", str(CODE_DIR / "data" / "nltk_data"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

MC_TASKS = ["hellaswag", "mmlu", "truthfulqa_mc1", "winogrande"]
# the per-item 0/1 metric that defines each task's score
METRIC = {
    "hellaswag": "acc",
    "mmlu": "acc",
    "truthfulqa_mc1": "acc",
    "winogrande": "acc",
    "ifeval": "prompt_level_strict_acc",
    "humaneval": "pass@1",
}
SUITE = list(METRIC)


def _item_scores(samples: dict, task: str) -> dict:
    """{item id: 0/1} for one suite task, merging MMLU's 57 subjects."""
    out = {}
    for name, rows in samples.items():
        if not (name == task or (task == "mmlu" and name.startswith("mmlu_"))):
            continue
        for r in rows:
            key = METRIC[task] if METRIC[task] in r else next(k for k in r if k.startswith("pass"))
            v = r[key]
            if isinstance(v, (list, tuple)):
                v = v[0]
            if isinstance(v, dict):
                v = next(iter(v.values()))
            out[f"{name}:{r['doc_id']}"] = int(bool(v)) if isinstance(v, bool) else float(v)
    return out


def run_suite(core, tok, out_dir, limit=None, tasks=None, batch_size=16, gen_batch_size=96) -> dict:
    import lm_eval
    from lm_eval.models.huggingface import HFLM

    tasks = tasks or SUITE
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    was_training = core.training
    core.eval()
    tok.padding_side = "left"
    items, timing = {}, {}

    def evaluate(names, chat, bs):
        lm = HFLM(pretrained=core, tokenizer=tok, batch_size=bs, softmax_dtype="float32", max_length=4096, enable_thinking=False)
        t = time.time()
        res = lm_eval.simple_evaluate(
            model=lm, tasks=names, limit=limit, log_samples=True, apply_chat_template=chat,
            bootstrap_iters=0, confirm_run_unsafe_code=True, verbosity="ERROR",
        )
        for n in names:
            items[n] = _item_scores(res["samples"], n)
            timing[n] = round(time.time() - t, 1)
        return res

    mc = [t for t in MC_TASKS if t in tasks]
    if mc:
        evaluate(mc, False, batch_size)
    if "humaneval" in tasks:
        res = evaluate(["humaneval"], False, gen_batch_size)
        try:  # keep the raw completions for reading; never let this break a run
            rows = [{"doc_id": r["doc_id"], "resp": str(r["resps"][0][0])[:3000], "ok": items["humaneval"][f"humaneval:{r['doc_id']}"]} for r in res["samples"]["humaneval"]]
            (out_dir / "humaneval_responses.json").write_text(json.dumps(rows))
        except Exception as err:  # noqa: BLE001
            print("could not save humaneval responses:", repr(err))
    if "ifeval" in tasks:
        res = evaluate(["ifeval"], True, gen_batch_size)
        (out_dir / "ifeval_responses.json").write_text(
            json.dumps([{"doc_id": r["doc_id"], "resp": r["resps"][0][0][:4000], "ok": r[METRIC["ifeval"]]} for r in res["samples"]["ifeval"]])
        )
    scores = {t: sum(v.values()) / len(v) for t, v in items.items()}
    summary = {
        "scores": {t: round(100 * s, 3) for t, s in scores.items()},
        "n": {t: len(v) for t, v in items.items()},
        "suite_mean": round(100 * sum(scores.values()) / len(scores), 3),
        "seconds": timing,
    }
    with gzip.open(out_dir / "items.json.gz", "wt") as f:
        json.dump(items, f)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    if was_training:
        core.train()
    return summary
