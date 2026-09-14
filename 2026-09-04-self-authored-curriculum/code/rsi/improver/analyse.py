"""ANALYSE (docs/03 §4 step 5): the policy reads its own failed trajectories (rollout episodes of this iteration,
never dev/held-out), clusters them by failure type, and emits schema-validated edits to the improver:
append-only rows for failure_playbook.md, a new themes.json, bounded edits to thresholds.json, and optional
rewrites of generator_prompt.md / mutation_ops.md. Every edit is diffed (git) and logged."""
from __future__ import annotations
import json, re, time
from collections import Counter, defaultdict
from pathlib import Path
from rsi.db import DB
from rsi.harness.prompt import split_thinking
from rsi.harness.vllm_client import VLLMChat

FAILURE_TYPES = ["command_loop", "turn_exhaustion", "wrong_tool", "misread_spec", "env_error", "gave_up_early", "partial_output", "unknown"]

def classify(records: list[dict], stop_reason: str | None) -> str:
    cmds = [r.get("command") for r in records if r.get("command")]
    if stop_reason in ("max_turns",): return "turn_exhaustion"
    if stop_reason in ("no_command", "prompt_too_long_for_window", "max_compactions"): return "env_error" if stop_reason != "no_command" else "gave_up_early"
    if len(cmds) >= 4 and len(set(cmds[-4:])) == 1: return "command_loop"
    exits = [r.get("exit_code") for r in records if r.get("exit_code") is not None]
    if exits and sum(1 for e in exits if e not in (0, None)) / len(exits) > 0.6: return "wrong_tool"
    if stop_reason == "task_complete" and len(cmds) <= 3: return "misread_spec"
    if stop_reason == "task_complete": return "partial_output"
    return "unknown"

ANALYSE_PROMPT = """You are improving the task generator of a self-training loop. Below are failure clusters from the policy's own
attempts at its self-written tasks this iteration, with example trajectories (commands and final observations), plus the
current playbook and themes.

Failure clusters (type: count):
{clusters}

Example failed trajectories:
{examples}

Current playbook (append-only table; you may ADD rows):
{playbook}

Current themes.json:
{themes}

Write your reply as exactly these sections, each with one fenced block:

### PLAYBOOK_ROWS
```markdown
| <failure cluster> | <diagnosis in one sentence> | <task template that would train this weakness, one sentence> |
```
(1-5 new rows; never delete existing rows)

### THEMES
```json
{{"themes": [<8-14 short theme strings, keep the good ones, add themes that target the failure clusters>],
 "domain_targets": {{<same 9 domain keys as before, fractions summing to 1, each between 0.03 and 0.25>}}}}
```

### THRESHOLDS
```json
{{"learning_band": [lo, hi], "anchor_fraction": x, "candidates_per_iteration": n}}
```
(only change a value if the clusters justify it; values must stay inside the allowed ranges: {ranges})
"""

SECTION = re.compile(r"###\s*(PLAYBOOK_ROWS|THEMES|THRESHOLDS)\s*\n+```[a-z]*\n(.*?)```", re.DOTALL)

def analyse_and_edit(db: DB, run, h_dir: Path, k: int, *, base_url: str, model: str, thinking: bool = True, max_examples: int = 6, log=print) -> dict:
    rows = db.q("SELECT e.episode_id, e.task_id, e.reward, e.stop_reason, e.turns, t.record_path FROM episodes e "
                "JOIN (SELECT episode_id, MIN(record_path) record_path FROM turns GROUP BY episode_id) t ON t.episode_id=e.episode_id "
                "WHERE e.purpose='rollout' AND e.iteration=? AND e.exception IS NULL AND e.reward IS NOT NULL AND e.reward < 1.0 AND COALESCE(e.quarantined,0)=0", k)
    clusters = Counter(); examples = defaultdict(list)
    for r in rows:
        recs = [json.loads(l) for l in open(r["record_path"]) if l.strip()]
        recs = [x for x in recs if "error" not in x]
        ft = classify(recs, r["stop_reason"]); clusters[ft] += 1
        if len(examples[ft]) < 2:
            cmds = [x.get("command") for x in recs if x.get("command")][:12]
            last_obs = next((x.get("observation", "") for x in reversed(recs) if x.get("observation")), "")
            instr = ""
            tp = db.q("SELECT path FROM tasks WHERE task_id=?", r["task_id"])
            if tp and (Path(tp[0]["path"]) / "instruction.md").exists(): instr = (Path(tp[0]["path"]) / "instruction.md").read_text()[:800]
            examples[ft].append({"task": r["task_id"], "instruction": instr, "commands": cmds, "last_observation": last_obs[:600], "stop": r["stop_reason"]})
    rep = {"iteration": k, "failed_episodes": len(rows), "clusters": dict(clusters), "edits": {}, "accepted": False}
    for ft, n in clusters.items(): db.metric(f"failure_{ft}", n, k, "ANALYSE")
    if not rows:
        log("analyse: no failed episodes; improver unchanged"); return rep
    ex_text = "\n\n".join(f"[{ft}] task {e['task']}\nInstruction: {e['instruction'][:500]}\nCommands: " + " ; ".join(c[:120] for c in e["commands"]) +
                          f"\nLast observation: {e['last_observation'][:300]}\nStop: {e['stop']}"
                          for ft, exs in examples.items() for e in exs)[:12000]
    themes = json.loads((h_dir / "themes.json").read_text()); playbook = (h_dir / "failure_playbook.md").read_text()
    th = json.loads((h_dir / "thresholds.json").read_text())
    prompt = ANALYSE_PROMPT.format(clusters="\n".join(f"{t}: {n}" for t, n in clusters.most_common()), examples=ex_text, playbook=playbook[-4000:],
                                   themes=json.dumps(themes, indent=1), ranges=json.dumps(th["allowed_ranges"]))
    llm = VLLMChat(base_url, model)
    t0 = time.time()
    # 4,000 tokens with thinking on left nothing for the structured output once the failure list grew: iteration 4
    # spent the whole budget reasoning and was cut off mid-sentence, so every edit that round was lost (docs/07).
    res = llm.complete([{"role": "user", "content": prompt}], max_tokens=12000, temperature=0.7, top_p=1.0, enable_thinking=thinking, seed=k)
    if getattr(res, "finish_reason", "") == "length":
        log("ANALYSE output hit the token limit; edits from this round are probably truncated")
    text = split_thinking(res.content)[1]
    secs = {m.group(1): m.group(2) for m in SECTION.finditer(text)}
    (run.root / "iterations" / f"{k:02d}").mkdir(parents=True, exist_ok=True)
    (run.root / "iterations" / f"{k:02d}" / "analyse-raw.txt").write_text(res.content)
    # playbook rows: append-only, validated shape
    new_rows = [l for l in secs.get("PLAYBOOK_ROWS", "").splitlines() if l.startswith("|") and l.count("|") >= 4 and "---" not in l and "failure cluster" not in l.lower()]
    if new_rows:
        (h_dir / "failure_playbook.md").write_text(playbook.rstrip("\n") + "\n" + "\n".join(new_rows[:5]) + "\n"); rep["edits"]["playbook_rows"] = len(new_rows[:5])
    # themes: validated
    try:
        t2 = json.loads(secs.get("THEMES", ""))
        ok = isinstance(t2.get("themes"), list) and 6 <= len(t2["themes"]) <= 16 and all(isinstance(x, str) and 3 < len(x) < 120 for x in t2["themes"])
        dt = t2.get("domain_targets", {})
        ok = ok and set(dt) == set(themes["domain_targets"]) and all(0.03 <= float(v) <= 0.25 for v in dt.values()) and abs(sum(map(float, dt.values())) - 1) < 0.05
        if ok:
            (h_dir / "themes.json").write_text(json.dumps({"themes": t2["themes"], "domain_targets": {d: float(v) for d, v in dt.items()}}, indent=1)); rep["edits"]["themes"] = True
        else: rep["edits"]["themes"] = "rejected:schema"
    except (json.JSONDecodeError, TypeError, ValueError, AttributeError) as ex:
        rep["edits"]["themes"] = f"rejected:{type(ex).__name__}"
    # thresholds: bounded
    try:
        t3 = json.loads(secs.get("THRESHOLDS", "{}")); changed = {}
        rng = th["allowed_ranges"]
        lb = t3.get("learning_band")
        if isinstance(lb, list) and len(lb) == 2 and rng["learning_band"][0][0] <= lb[0] <= rng["learning_band"][0][1] and rng["learning_band"][1][0] <= lb[1] <= rng["learning_band"][1][1] and lb[0] < lb[1]:
            th["learning_band"] = [float(lb[0]), float(lb[1])]; changed["learning_band"] = th["learning_band"]
        af = t3.get("anchor_fraction")
        if isinstance(af, (int, float)) and rng["anchor_fraction"][0] <= af <= rng["anchor_fraction"][1]:
            th["anchor_fraction"] = float(af); changed["anchor_fraction"] = af
        cp = t3.get("candidates_per_iteration")
        if isinstance(cp, int) and rng["candidates_per_iteration"][0] <= cp <= rng["candidates_per_iteration"][1]:
            th["candidates_per_iteration"] = cp; changed["candidates_per_iteration"] = cp
        if changed: (h_dir / "thresholds.json").write_text(json.dumps(th, indent=1))
        rep["edits"]["thresholds"] = changed
    except (json.JSONDecodeError, TypeError) as ex:
        rep["edits"]["thresholds"] = f"rejected:{type(ex).__name__}"
    rep["finish_reason"] = getattr(res, "finish_reason", "")
    rep["latency_s"] = round(time.time() - t0, 1); rep["accepted"] = bool(rep["edits"])
    db.event("analyse", iteration=k, **{k2: v for k2, v in rep.items() if k2 != "iteration"})
    log(f"analyse: clusters {dict(clusters)} edits {rep['edits']}")
    return rep
