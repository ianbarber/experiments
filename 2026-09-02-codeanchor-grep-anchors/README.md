# CodeAnchor-style anchors on grep output: does passive structural injection help a local coding agent refactor?

**Date:** 2026-09-02 → 2026-09-06 · **Fleet:** DGX Spark (GB10)
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
- **Variance win plausible, unproven:** a 7-episode rerun on the five largest-gap instances shows
  the control producing the runaway episodes (three 300-step-cap hits vs none) at equal typical
  cost, but the spread reduction is not significant (bootstrap CI 0.42–1.57) and the wall-clock
  tail is not shorter locally.
- **The mechanism works and is cheap** (half of greps anchored, ≈4% of prompt tokens,
  5.6 s/episode, zero failures) **and the agent acts on it** (99 of 99 flagged gold files
  opened, 92 patched) — but both arms miss the same 64 gold files.
- **Why (corrected 2026-09-14):** of the round-1 misses, 41% are non-Python targets, 32%
  are Python files never referenced by any symbol the agent chose to search, and 12% would
  have been named by a working, uncapped server. Passive injection is bounded by the
  questions the agent asks.
- **Deviation audited:** my display caps (the paper caps nothing) hid only 3 of 69
  missed files, measured by an uncapped replay of every anchored grep.
- **Post-hoc corrections (2026-09-14, no new runs):** two of the 29 rollout images ship
  `PYTHONPATH=/testbed` on a `src/`-layout repo, which makes Pyrefly's find-references
  return only the relative-import neighbourhood of the opened file; the anchors in
  lerobot-2808 and transformers-38332 under-reported users throughout (e.g. "used by 3
  sites in 2 files" for a class used in 20). The other 27 instances match a clean replay
  file-for-file. And 65 of the 69 jointly-missed gold files were omitted by at least one
  patch that resolved the instance: the coverage gap is mostly docs, squashed unrelated
  changes and untested parts of the refactor, not a correctness gap. See the report's
  addendum.
- **The surprise: the failures are not hidden-test surprises.** Against the files the
  tests actually need, both arms find them (recall 0.96 E vs 0.94 A8; 0.68 against gold
  files). And ProMax task statements say what the tests check: django's quotes the
  duplicate-partial error message verbatim, lerobot's says "raise a `TimeoutError`",
  optuna's names `inverse_squared_lengthscales`. The patches implement most of a long
  enumerated spec and drop one item, and both arms drop the same items on the same
  instances. The dominant failure is partial compliance with a long, explicit
  specification (8 of 18 E failures, 5 of 16 A8), not incomplete refactoring; 4 per arm
  are requirements only the tests pin, 2 per arm need a live site's JSON layout.
- **E3** (definition-site placement on file views, uncapped — the paper's own trigger
  surface): same outcome (32/50, paired 1-vs-4), fewer steps (−4.5%, p = 0.043) but +35%
  wall-clock (p = 0.015) and +36% median tokens; only 9 of the 64 jointly-missed gold files
  were ever named. More exposure is a net cost on a local server.

## Contents

| Path | What |
|---|---|
| `REPORT.md` | Full report: mechanism, results, uptake/coverage diagnostics, cap counterfactual, deviations |
| `LABNOTES.md` | Chronological lab notebook (four launches of E-r1, the cap concern, the replay) |
| `NOTES.md` | Protocol amendments + run ledger for E / A8 / E3 |
| `code/` | Environment subclass + launcher shim, arm configs, runner, daemon `anchor` op, analysis scripts (diff vs the earlier entry) |
| `results/` | Per-run per-instance outcomes (E, A8, E3), per-grep/view anchor telemetry, summaries, episode CSVs, uncapped replay + counterfactual, cost/consistency tables; 2026-09-14 audit: `replay_uncapped_local_e_r1.jsonl` (clean-server replay), `base_image_env.txt` (PYTHONPATH per base image), `missed_audit_2026-09-14.txt` (corrected classification + necessity) |
