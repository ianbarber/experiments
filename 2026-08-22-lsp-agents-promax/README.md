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

- **Nothing helped; gates hurt.** A 72% · B (lsp available) 71% · C2 (imperative) 68% ·
  D (hard gate) 58% — the study's only significant paired effect (p≈0.016) · D2 (silent
  soft gate) 64% · hosted A 94% vs hosted D2 84%.
- **Navigation tools are unadoptable by prompting**: 1 `lsp refs` call in ~180
  tool-equipped episodes; only the verification-shaped "run diag on edited files" clause
  gets compliance. Traces show why the offer is moot: localization is instant (gold file
  touched by step ~2, edit precision ≈1.0); failures are *recall* — the right files, but
  only ⅓ of the needed set.
- **Why gates backfire**: on dynamically-typed repos, new type errors frequently don't
  fail the suite — the gate pays real steps for problems the benchmark never charges.
  The repair loop itself works (silent gate: 12/17 gate-fired episodes still resolved).
- **Harness landmine**: mini-swe-agent's default 2h `container_timeout` kills long
  episodes mid-flight (deaths: A 4 → D2 16, monotone in episode length) and masqueraded
  as step-cap exhaustion; at local decode speeds this confounded the gate arms until
  fixed (hosted reruns, 8h wall: zero deaths).
- **Serving stack > intervention**: same model, instances, and scaffold scored 72% local
  vs 94% hosted (fp8 providers, default xhigh thinking) — with a contamination caveat:
  hosted resolved patches overlap gold added-lines at 0.62 mean (local: 0.39).

## Contents

| Path | What |
|---|---|
| `REPORT.md` | Full report: setup, arms, results, the five findings, benchmark notes, verdict |
| `LABNOTES.md` | Chronological lab notebook — everything in order, including failures, self-corrections, and the cost incident |
| `PLAN.md` / `NOTES.md` | Original plan; protocol amendments + run ledger |
| `code/` | Arm configs, batch runner, `lsp` CLI + gate + image layers, serving notes, analysis scripts |
| `results/` | Per-run per-instance outcomes, summary.csv, localization analysis output |
