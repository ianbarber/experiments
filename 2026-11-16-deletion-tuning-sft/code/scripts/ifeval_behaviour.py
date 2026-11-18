"""What the IFEval answers look like after tuning.

For every scored checkpoint with saved IFEval responses, writes per-answer flags next to them
(runs/RUN/suite_stepN/ifeval_flags.json: passed, opens a thinking block, contains Wordle text or
a tool call) and a summary to ../results/ifeval_behaviour.json. The flags are what the analysis
and the evidence archive use; the responses themselves stay out of the repository.
"""
import json
import re
from pathlib import Path

CODE = Path(__file__).resolve().parent.parent
RUNS, RESULTS = CODE / "runs", CODE.parent / "results"
WORDLE = re.compile(r"<tool_call>|\b(TRADE|STARE|CRANE|SLATE|RAISE)\b|Words that fit|No feedback yet|opener")


def flags(path):
    rows = json.loads(path.read_text())
    return [{"doc_id": r["doc_id"], "ok": bool(r["ok"]), "think": "<think>" in r["resp"] or "</think>" in r["resp"],
             "wordle": bool(WORDLE.search(r["resp"])), "chars": len(r["resp"])} for r in rows]


def main():
    out = {}
    for path in sorted(RUNS.glob("*/suite_step*/ifeval_responses.json")) + sorted(RUNS.glob("session2/*/suite_step*/ifeval_responses.json")):
        fl = flags(path)
        (path.parent / "ifeval_flags.json").write_text(json.dumps(fl, separators=(",", ":")))
        think = [f for f in fl if f["think"]]
        rest = [f for f in fl if not f["think"]]
        key = f"{path.parent.parent.relative_to(RUNS)}/step{path.parent.name.removeprefix('suite_step')}"
        out[key] = {
            "n": len(fl), "pass": sum(f["ok"] for f in fl),
            "opens_think_block": len(think), "pass_among_think_block": sum(f["ok"] for f in think),
            "pass_among_rest": sum(f["ok"] for f in rest), "pass_rate_among_rest": round(100 * sum(f["ok"] for f in rest) / max(1, len(rest)), 1),
            "wordle_text_or_tool_call": sum(f["wordle"] for f in fl), "mean_chars": round(sum(f["chars"] for f in fl) / len(fl)),
        }
    (RESULTS / "ifeval_behaviour.json").write_text(json.dumps(out, indent=1))
    for k, v in out.items():
        print(f"{k:40s} pass {v['pass']:3d}/{v['n']}  opens think block {v['opens_think_block']:3d}  pass rate of the rest {v['pass_rate_among_rest']:5.1f}%  "
              f"wordle text or tool call {v['wordle_text_or_tool_call']:3d}  mean chars {v['mean_chars']}")


if __name__ == "__main__":
    main()
