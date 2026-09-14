"""Human inspection helpers (docs/03 §4 'Pause and inspect'): print trajectories, funnels, episode tables."""
from __future__ import annotations
import json
from pathlib import Path
from rsi.harness.prompt import split_thinking

def print_trajectory(trial_dir: Path, *, max_chars: int = 300, show_reasoning: bool = False) -> None:
    trial_dir = Path(trial_dir)
    rp = trial_dir / "verifier" / "reward.txt"
    res = json.loads((trial_dir / "result.json").read_text()) if (trial_dir / "result.json").exists() else {}
    meta = (res.get("agent_result") or {}).get("metadata") or {}
    print(f"### {trial_dir.name}  task={res.get('task_name')}  reward={rp.read_text().strip() if rp.exists() else 'n/a'}  "
          f"turns={meta.get('turns')} stop={meta.get('stop_reason')} compactions={meta.get('compactions')} wall={meta.get('wall_s')}s "
          f"tokens in/out={meta.get('n_input_tokens')}/{meta.get('n_output_tokens')}")
    tp = trial_dir / "agent" / "turns.jsonl"
    if not tp.exists():
        print("  (no turn records)"); return
    for line in open(tp):
        r = json.loads(line)
        if "error" in r:
            print(f"  !! {r['error']}"); continue
        reasoning, visible = split_thinking(r.get("content", ""))
        tag = "SUMMARY" if r.get("is_summary") else f"seg{r.get('segment',0)}/t{r.get('turn',0)}"
        print(f"--- {tag} prompt={len(r.get('prompt_token_ids') or [])} gen={len(r.get('token_ids') or [])} finish={r.get('finish_reason')} "
              f"exit={r.get('exit_code')} llm={r.get('llm_latency_s')}s cmd={r.get('cmd_latency_s')}s")
        if show_reasoning and reasoning:
            print("   think:", reasoning[:max_chars].replace("\n", " "))
        print("   reply:", visible[:max_chars].replace("\n", " | "))
        if r.get("observation"):
            print("   obs  :", r["observation"][:max_chars].replace("\n", " | "))

def job_trials(job_dir: Path):
    return sorted(p for p in Path(job_dir).iterdir() if p.is_dir() and not p.name.startswith("_"))
