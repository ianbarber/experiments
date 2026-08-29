# LSP tools for a local coding agent: SWE-Bench ProMax × Qwen3.8-27B

**Date:** 2026-08-22 → 2026-08-28 · **Fleet:** DGX Spark (GB10) serving + chunklebox/leejr
x86 workers + NAS · **Benchmark:** SWE-Bench ProMax python subset · **Scaffold:**
mini-swe-agent 2.4.6

## Brief

SWE-Bench ProMax scores agents on large multi-file refactors — exactly the task
find-references tooling should help. This experiment gives a local Qwen3.8-27B agent LSP
capability (Pyrefly-backed `lsp` CLI) at escalating strength — available, documented,
politely preferred, imperatively mandated, and finally enforced via a submission-time
type-check acceptance gate (hard, then silent-soft, then hosted) — and measures resolve
rate with paired two-round runs per arm.

## Headline results

- **The model never adopts navigation tools, at any instruction strength** — 1 `lsp refs`
  call in ~180 tool-equipped episodes (only verification-shaped `lsp diag` asks get
  compliance). Traces show entry-point localization is already instant (gold file by step
  ~2, edit precision ≈1.0) while failures are reference-completeness recall (~⅓ of needed
  files) — i.e. **the one thing the tool is demonstrably good for is the one thing the
  agent won't use it for**. Whether that's ask-shape, grep familiarity, or trigger
  density is untested.
- **Type-check acceptance gates didn't help and probably hurt**: A 72% · B 71% · C2 68% ·
  D (hard gate) 58% (p≈0.016 nominal, ≈0.08 Holm; wall-confounded) · D2 (silent soft)
  64% · hosted (clean, wall-free): A 94% vs D2 84% (n.s.). Retroactive measurement shows
  why: the gate would have rejected 35% of patches that pass the full test suite — type cleanliness is nearly orthogonal to the scoring criterion.
- **Two transferable harness findings** (`HARNESS-NOTES.md`): mini-swe-agent's default 2h
  `container_timeout` kills long episodes mid-flight and masquerades as step-cap
  exhaustion (deaths monotone in episode length, A 4 → D2 16; zero after the 8h fix);
  and the local/hosted serving gap (72% vs 94%, same everything else) dwarfs every
  intervention tested.
- The hosted 94% is treated as an **unvalidated anomaly** (5× the published field for
  this subset/scaffold; hosted resolved patches overlap gold at 0.62 mean, 18/47 > 0.75
  — close to reproduction). Decisive next step: symbol/path perturbation re-run.

## Contents

| Path | What |
|---|---|
| `REPORT.md` | Full report: setup, arms, results, findings, benchmark notes, verdict |
| `HARNESS-NOTES.md` | Standalone write-up of the two transferable harness findings |
| `LABNOTES.md` | Chronological lab notebook — everything in order, including failures, self-corrections, and the cost incident |
| `PLAN.md` / `NOTES.md` | Original plan; protocol amendments + run ledger |
| `code/` | Arm configs, batch runner, `lsp` CLI + gate + image layers, serving notes, analysis scripts |
| `results/` | Per-run per-instance outcomes, summary.csv, localization analysis output |
