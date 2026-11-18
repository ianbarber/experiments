"""Why do the adapters score higher on HumanEval than base? Counts completions that stall
(`pass`, or no code at all) and the problems gained and lost against base.
Needs scripts/humaneval_read.py to have been run for the three checkpoints. Writes ../results/humaneval_reading.json.
"""
import json
import re
from pathlib import Path

CODE = Path(__file__).resolve().parent.parent
RUNS = CODE / "runs"
CHECKPOINTS = (("base", "base", 0), ("dt", "dt_lr1e-3_a100-1000_s42", 400), ("lora", "lora_r4_lr3e-4_s42", 1000))


def load(run, step):
    return {r["doc_id"]: r for r in json.loads((RUNS / run / f"humaneval_step{step}" / "humaneval_responses.json").read_text())}


def stall(text):
    code = [l for l in text.strip().splitlines() if l.strip() and not l.strip().startswith("#")]
    return (not code) or bool(re.match(r"^(pass\b|\.\.\.|raise NotImplementedError)", code[0].strip()))


def main():
    data = {name: load(run, step) for name, run, step in CHECKPOINTS}
    base, out = data["base"], {}
    for name, m in data.items():
        out[name] = {"n": len(m), "pass": sum(bool(r["ok"]) for r in m.values()), "stalls": sum(stall(r["resp"]) for r in m.values()),
                     "mean_chars": round(sum(len(r["resp"]) for r in m.values()) / len(m))}
        if name != "base":
            gained = [d for d in m if m[d]["ok"] and not base[d]["ok"]]
            lost = [d for d in m if not m[d]["ok"] and base[d]["ok"]]
            out[name].update(gained=len(gained), lost=len(lost), gained_where_base_stalled=sum(stall(base[d]["resp"]) for d in gained))
        print(name, out[name])
    (CODE.parent / "results" / "humaneval_reading.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
