# CodeAnchor-style anchors on grep output: does passive structural injection help a local coding agent refactor?

**Date:** 2026-09-02 → 2026-09-05 (variance rerun in progress) · **Fleet:** DGX Spark (GB10)
serving + two x86 docker workers (worker-a, worker-b) + NAS · **Benchmark:** SWE-Bench ProMax python
subset · **Scaffold:** mini-swe-agent 2.4.6 · **Model:** Qwen3.8-27B-FP8 (local)

Follow-up to [LSP tools for a local coding agent](../2026-08-22-lsp-agents-promax/),
which found the model never adopts `lsp refs` at any prompt strength while its dominant
failure is incomplete refactoring.

## Brief

CodeAnchor (arXiv 2606.26979) injects static "used by" facts into source files as
comments so a grep-first agent sees structure without adopting a new tool, and reports
small localization gains on SWE-bench. Here the same facts come from the warm language
server and are appended to the agent's grep observations (in-file comments would leak into
graded patches). Prompt byte-identical to the baseline; paired two-round runs against a
same-period control (A8), hosts swapped.

## Headline results

- **Null on outcome, and not cheaper.** E 32/50 (64%) vs control 34/50 (68%); paired E
  better 2, control better 3, 20 tied (p = 1.0). Raw totals favour E by 13% (tokens) and 5%
  (wall), but paired per instance the ratios are 0.95 (p = 0.76) and 1.03 (p = 0.40) — two
  outlier control episodes, not a saving. ~7% fewer steps (p = 0.07) is cancelled by ~8%
  higher per-step latency from the longer context. Edit recall identical.
- **No variance win on two rounds** (per-instance round-to-round dispersion equal; extreme tail
  shorter with anchors — variance rerun queued to test it).
- **The mechanism works and is cheap** (half of greps anchored, ≈4% of prompt tokens,
  5.6 s/episode, zero failures) **and the agent acts on it** (99 of 99 flagged gold files
  opened, 92 patched) — but both arms miss the same 64 gold files.
- **Why:** of the round-1 misses, 41% are non-Python targets and 39% are Python files
  never referenced by any symbol the agent chose to search. Passive injection is bounded
  by the questions the agent asks.
- **Deviation audited:** my display caps (the paper caps nothing) hid only 3 of 69
  missed files, measured by an uncapped replay of every anchored grep.
- **E3** (definition-site placement on file views, uncapped — the paper's own trigger
  surface): same outcome (32/50, paired 1-vs-4), fewer steps (−4.5%, p = 0.043) but +35%
  wall-clock (p = 0.015) and +36% median tokens; only 9 of the 64 jointly-missed gold files
  were ever named. More exposure is a net cost on a local server.

## Contents

| Path | What |
|---|---|
| `REPORT.md` | Full report: mechanism, results, uptake/coverage diagnostics, cap counterfactual, deviations, follow-ups |
| `LABNOTES.md` | Chronological lab notebook (four launches of E-r1, the cap concern, the replay) |
| `NOTES.md` | Protocol amendments + run ledger for E / A8 / E3 |
| `code/` | Environment subclass + launcher shim, arm configs, runner, daemon `anchor` op, analysis scripts (diff vs the earlier entry) |
| `results/` | Per-run per-instance outcomes (E, A8, E3), per-grep/view anchor telemetry, summaries, episode CSVs, uncapped replay + counterfactual, cost/consistency tables |
