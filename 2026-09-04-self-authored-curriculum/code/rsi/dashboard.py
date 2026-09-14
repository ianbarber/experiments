"""Static HTML dashboard from SQLite (docs/03 §10, v0). Rebuilt at every step boundary; no server needed."""
from __future__ import annotations
import html, json, time
from collections import defaultdict
from pathlib import Path
from rsi.db import DB

def _table(rows, cols, fmt=None):
    fmt = fmt or {}
    h = "<table><tr>" + "".join(f"<th>{html.escape(c)}</th>" for c in cols) + "</tr>"
    for r in rows:
        h += "<tr>" + "".join(f"<td>{html.escape(str(fmt.get(c, lambda v: v)(r[c]) if r[c] is not None else ''))}</td>" for c in cols) + "</tr>"
    return h + "</table>"

def _sparkline(values, w=260, h=48):
    if not values: return ""
    lo, hi = min(values), max(values); rng = (hi - lo) or 1.0
    pts = " ".join(f"{i * w / max(len(values) - 1, 1):.1f},{h - (v - lo) / rng * (h - 6) - 3:.1f}" for i, v in enumerate(values))
    return f'<svg width="{w}" height="{h}"><polyline fill="none" stroke="#2b6cb0" stroke-width="2" points="{pts}"/></svg>'

def build_dashboard(run, db: DB) -> Path:
    out = run.dashboard / "index.html"; out.parent.mkdir(parents=True, exist_ok=True)
    r3 = lambda v: f"{v:.3f}" if isinstance(v, float) else v
    parts = [f"<h1>RSI at home: run {html.escape(run.root.name)}</h1><p>generated {time.strftime('%Y-%m-%d %H:%M:%S')}</p>"]
    st = run.root / "state.json"
    if st.exists(): parts.append("<pre>" + html.escape(st.read_text()) + "</pre>")
    # 1. per-iteration headline: rollout pass rate, dev, gate funnel, training diagnostics
    its = db.q("SELECT iteration, AVG(reward) mean_reward, COUNT(*) n, SUM(exception IS NOT NULL) errors, AVG(turns) turns, AVG(n_output_tokens) gen_tokens FROM episodes WHERE purpose='rollout' GROUP BY iteration ORDER BY iteration")
    parts.append("<h2>Rollouts per iteration (in-distribution, never the headline)</h2>" + _table(its, ["iteration", "n", "mean_reward", "errors", "turns", "gen_tokens"], {"mean_reward": r3, "turns": r3, "gen_tokens": lambda v: f"{v:.0f}"}))
    dev = db.q("SELECT policy, COUNT(*) n, AVG(reward) mean_reward FROM episodes WHERE purpose='dev' GROUP BY policy ORDER BY policy")
    parts.append("<h2>Dev set (regression gate)</h2>" + _table(dev, ["policy", "n", "mean_reward"], {"mean_reward": r3}))
    ho = db.q("SELECT policy, purpose, COUNT(*) n, AVG(reward) mean_reward FROM episodes WHERE purpose LIKE 'heldout%' OR purpose LIKE 'tw%' OR purpose LIKE 'tb%' GROUP BY policy, purpose ORDER BY purpose, policy")
    parts.append("<h2>Held-out evaluations</h2>" + _table(ho, ["purpose", "policy", "n", "mean_reward"], {"mean_reward": r3}))
    # 2. funnel
    g = db.q("SELECT iteration, check_name, verdict, COUNT(*) n FROM gates GROUP BY iteration, check_name, verdict ORDER BY iteration, check_name, verdict")
    parts.append("<h2>Task funnel (gate verdicts)</h2>" + _table(g, ["iteration", "check_name", "verdict", "n"]))
    tk = db.q("SELECT iteration_added, source, status, COUNT(*) n FROM tasks GROUP BY iteration_added, source, status ORDER BY iteration_added, source, status")
    parts.append("<h2>Task pool</h2>" + _table(tk, ["iteration_added", "source", "status", "n"]))
    tiers = db.q("SELECT tier, COUNT(*) n FROM tasks WHERE status LIKE 'gated%' GROUP BY tier")
    parts.append("<h3>Tiers (gated tasks)</h3>" + _table(tiers, ["tier", "n"]))
    # 3. training diagnostics
    m = db.q("SELECT iteration, name, value, extra FROM metrics WHERE step='UPDATE' ORDER BY id")
    series = defaultdict(list)
    for r in m: series[r["name"]].append(r["value"])
    parts.append("<h2>Training diagnostics (per optimiser step, all iterations)</h2>" + "".join(f"<div><b>{html.escape(n)}</b> last={v[-1]:.4f} {_sparkline(v)}</div>" for n, v in series.items()))
    # 4. integrity
    q = db.q("SELECT iteration, SUM(quarantined) quarantined, SUM(flags IS NOT NULL) flagged, COUNT(*) n FROM episodes WHERE purpose='rollout' GROUP BY iteration")
    parts.append("<h2>Integrity</h2>" + _table(q, ["iteration", "n", "flagged", "quarantined"]))
    # 5. improver diffs
    an = db.q("SELECT ts, payload FROM events WHERE kind='analyse' ORDER BY id")
    parts.append("<h2>Improver edits (ANALYSE)</h2>" + "".join(f"<pre>{html.escape(r['payload'][:1500])}</pre>" for r in an))
    # 6. steps
    steps = db.q("SELECT k, step, status, started_at, finished_at, notes FROM iterations ORDER BY k, started_at")
    parts.append("<h2>Steps</h2>" + _table(steps, ["k", "step", "status", "notes"]))
    css = "<style>body{font-family:system-ui;margin:24px;max-width:1200px}table{border-collapse:collapse;margin:8px 0}td,th{border:1px solid #ccc;padding:3px 8px;font-size:13px}pre{background:#f4f4f4;padding:8px;font-size:12px;white-space:pre-wrap}</style>"
    out.write_text("<!doctype html><meta charset='utf-8'><title>RSI at home</title>" + css + "\n".join(parts))
    return out
