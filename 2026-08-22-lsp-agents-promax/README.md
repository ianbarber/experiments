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

- **The dominant failure mode — incomplete refactoring — is invisible to a type
  checker.** The agent localizes instantly (gold file by step ~2, edit precision ≈1.0)
  but failed episodes cover only ~⅓ of the needed files; since unwritten code emits no
  diagnostics, a delta-scoped Pyrefly gate flagged *passing* patches at 35% vs *failing*
  ones at 14%. The tool aimed at exactly this failure (`lsp refs`) is the one the agent
  never voluntarily used.
- **Navigation tools go unadopted at any instruction strength** — 1 `lsp refs` call in
  ~180 tool-equipped episodes; only verification-shaped `lsp diag` asks get compliance
  (non-adoption, not mechanistic uselessness; grep-familiarity and trigger-density
  confounds untested).
- **Acceptance gates didn't help and probably hurt**: A 72% · B 71% · C2 68% · D 58% ·
  D2 64% local; hosted (wall-free): A 94% vs D2 84% (n.s.). The 2×2 above explains the
  economics.
- Both headline rates are ~4–5× the published field for this subset/scaffold and are
  treated as unvalidated pending a reference-model harness control; paired comparisons
  are unaffected. Two operational asides — mini-swe-agent's 2h container wall and a
  ~20-point local-serving understatement — are in `HARNESS-NOTES.md`.

## Contents

| Path | What |
|---|---|
| `REPORT.md` | Full report: setup, arms, results, findings, benchmark notes, verdict |
| `HARNESS-NOTES.md` | Standalone write-up of the two transferable harness findings |
| `LABNOTES.md` | Chronological lab notebook — everything in order, including failures, self-corrections, and the cost incident |
| `PLAN.md` / `NOTES.md` | Original plan; protocol amendments + run ledger |
| `code/` | Arm configs, batch runner, `lsp` CLI + gate + image layers, serving notes, analysis scripts |
| `results/` | Per-run per-instance outcomes, summary.csv, localization analysis output |
