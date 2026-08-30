# Do LSP tools help a local coding agent refactor? (Qwen3.8-27B × SWE-Bench ProMax)

**Dates:** 2026-08-22 → 2026-08-28 · **Fleet:** DGX Spark (GB10, aarch64) serving + 2× x86
workers + NAS · **Benchmark:** SWE-Bench ProMax (arXiv 2608.09802), python subset (29
instances) · **Scaffold:** mini-swe-agent 2.4.6 · **Model:** Qwen3.8-27B (FP8 local /
OpenRouter fp8 hosted)

## TL;DR

- **The central finding: the dominant failure mode — incomplete refactoring — is
  invisible to a type checker.** Traces show the agent finds the right files instantly
  (edit precision ≈1.0) but covers only ~⅓ of the needed set when it fails; and because
  unwritten code emits no diagnostics, a delta-scoped type gate flags *passing* patches
  more often than failing ones (35% vs 14%). The tool most aimed at this failure
  (`lsp refs`) is the one the agent never voluntarily uses (§4–6).
- **Provisioning LSP navigation did nothing at any instruction strength** — 1 navigation
  call in ~180 tool-equipped episodes; only verification-shaped asks (`lsp diag`) get
  compliance. This is non-adoption, not evidence the tools are mechanistically useless
  (§4–5).
- **Type-check acceptance gates didn't help and probably hurt**: −14 (local hard), −8
  (local soft), −10 (hosted soft) — one nominally significant, wall-confounded; the
  clean hosted comparison is a non-significant −10; the 2×2 above explains the
  economics (§6).
- Validity notes: both headline rates (72% local / 94% hosted) are ~4–5× the published
  field for this subset+scaffold and are treated as unvalidated pending a
  reference-model harness control; the paired intervention comparisons are unaffected.
  The local/hosted gap itself traced to local serving implementation (effort knob +
  stack) — an aside for this study, written up separately in `HARNESS-NOTES.md`
  alongside the 2h-container-wall pitfall (§7).

## 1. Setup

- **Benchmark**: SWE-Bench ProMax — 170 expert-curated multilingual refactoring
  instances; we ran the 29-instance python subset. Gold patches avg 11.4 source files.
  Resolve = full test suite passes in the instance's Docker image, using the authors'
  own eval harness (`test_run.py`), which re-validates the golden patch per run and
  drops golden-invalid instances from the denominator (2–5 per run on our LAN; several
  images bake in an unreachable ByteDance proxy — `sys-proxy-rd-relay.byted.org` —
  which breaks their network-dependent tests outside the authors' infra).
- **Model**: Qwen3.8-27B (dense hybrid Gated-DeltaNet, released 2026-08-13; HF arch
  `qwen3_5`). Local: official FP8 checkpoint on the GB10 via `lmsysorg/sglang:qwen38-27b`
  + DSpark speculative decoding (`code/serving/SERVING.md`; pip vLLM/SGLang are dead ends
  on sm_121). Hosted: OpenRouter pinned to CoreWeave/Parasail (fp8).
- **Protocol**: mini-swe-agent (tool-calling mode), 300-step cap (paper protocol), temp
  1.0 / top_p 0.95 / top_k 20 (model-card thinking defaults; the paper doesn't state
  temperature), `reasoning_effort: medium` locally; the knob does not pass through
  OpenRouter, so hosted ran at default xhigh (§7). Two rounds per arm, hosts swapped
  between rounds.
- **Statistics**: all paired tests are **instance-level** (each instance's two-round
  score, 0–2, per arm; discordant counts below are counts of *distinct instances*), on
  the intersection of golden-valid sets; exact two-sided sign tests. Five paired
  comparisons are reported; Holm-corrected thresholds are noted where relevant. Rate
  columns carry 95% Wilson intervals computed at n≈50 episode-slots; because those are
  25 instances × 2 correlated rounds, effective n lies between 25 and 50 and the
  intervals are **optimistic** — at the n=25 cluster level, A's 72% is [52–86]. Read
  3–8 point gaps through the paired tests, not the rate column.
- **LSP tooling**: `code/lsp-tool/` — a bash CLI (`lsp def/refs/hover/sym/outline/diag/
  rename --dry-run`) over a warm daemon wrapping solidlsp (serena-agent 1.7.0), Pyrefly
  1.2.0 as the Python server, content-enriched output. Baked into per-instance image
  variants; verified in-container (refs 2.6 s cold, ~50 ms warm).

## 2. Arms

| Arm | Intervention |
|-----|--------------|
| A | Baseline (stock template) |
| B | `lsp` CLI present + neutral tool docs in prompt |
| C (pilot) | B + polite "prefer lsp" paragraph — abandoned after 5 episodes: 0 lsp calls |
| C2 | B + imperative, workflow-integrated instruction (lsp-first analysis; refs before change; diag before submit) |
| D | A's prompt + `submit` wrapper running a delta-scoped Pyrefly acceptance gate — **rejects on ≥1 new error fingerprint** (the "cap 8" is display-only); fail-closed: rejection blocks submission |
| D2 | Silent soft gate: prompt byte-identical to A; submission marker intercepted in the environment; after 2 rejections the next attempt passes silently |
| A/D2 hosted | A and D2 re-run via OpenRouter with `container_timeout: 8h` |

## 3. Results

**Local (GB10 serving), golden-valid python, 2 rounds each:**

| Arm | Resolved | Rate [95% CI] | Paired vs A (distinct instances) |
|-----|----------|---------------|----------------------------------|
| A | 36/50 | 72% [58–83] | — |
| B | 35/49 | 71% [58–82] | 3-vs-4, p=1.0 |
| C2 | 34/50 | 68% [54–79] | 4-vs-2, p≈0.69 |
| D | 29/50 | 58% [44–71] | 7-vs-0, p≈0.016 (Holm-adjusted ≈0.08) |
| D2 | 32/50 | 64% [50–76] | 6-vs-3, p≈0.51 |

> ⚠ **Read with §6.5 in mind**: local runs are contaminated by the 2-hour container
> wall, which disproportionately killed long (gate-arm) episodes. The clean gate
> comparison is the hosted pair below, where wall deaths were zero.

**Hosted (OpenRouter fp8, 8h wall), 2 rounds each:**

| Arm | Resolved | Rate [95% CI] | Paired |
|-----|----------|---------------|--------|
| A hosted | 47/50 | 94% [84–98] | — |
| D2 hosted | 42/50 | 84% [71–92] | A-better 6-vs-1, p≈0.125 |

Denominators: 50 = 25 golden-valid instances × 2 rounds (B loses one slot to a
host-varying golden failure). Where §4–6 cite "58 episodes," that is all episodes
including golden-invalid ones — behavioral measures (tool calls, gate firings) don't
require a valid golden. Per-run detail in `results/summary.csv`; per-instance outcomes in
`results/*.json`.

## 4. Finding: the model doesn't adopt navigation tools — with two untested confounds

- **B (available + documented): 0 lsp calls in 58 episodes** (~5,000+ commands); docs
  verified present in every prompt.
- **C pilot (polite preference): 0 calls in 5 episodes.**
- **C2 (imperative): 61 calls across 58 episodes — 60 `lsp diag`, 1 `lsp refs`.**

The pattern — verification-shaped compliance, zero navigation adoption — extends
arXiv 2608.13568's frontier finding (0–6% unprompted semantic-tool use) to an open 27B.
But two mundane explanations remain open, and our design cannot rule either out:

1. **Familiarity, not tool-shape**: arm B/C2 environments still contain grep — a
   navigation tool with enormous pretraining exposure — competing against a CLI the
   model has seen once, in a prompt. "Unadoptable by prompting" and "won't switch from a
   familiar tool to an unfamiliar one" predict identical data. The discriminating arm
   (remove or degrade grep) is cheap and was not run.
2. **Trigger specificity, not ask-shape**: "run diag on edited files" fires at one fixed
   workflow point; "use refs before changing any symbol" requires interrupting an
   established habit at dozens of unpredictable moments. The compliance asymmetry may be
   about trigger density rather than verification-vs-navigation framing.

What stands regardless: if you want this model using semantic navigation, prompt
engineering at any strength we tried is not the lever.

## 5. Finding: entry-point localization is a non-problem; the real failure went untested

Trace analysis over round 1 (`code/analysis/localization.py`, output in
`results/localization-round1.txt`):

| Split | first gold-file touch (step) | edit precision | edit recall | gold files missed |
|-------|------------------------------|----------------|-------------|-------------------|
| A resolved | 1.5 | 0.97 | 0.76 | 1.9 |
| A unresolved | 0.7 | 1.00 | 0.32 | 6.2 |
| B/C2 (same pattern) | 0.8–2.1 | 0.94–1.00 | 0.74–0.75 / 0.35–0.43 | 2.4–4.9 |

The agent touches a gold-patch file within ~2 steps in every episode and almost never
edits a wrong file; failures edit the *right* files but cover only ~⅓ of the needed set —
independently confirming, at trace level, the ProMax paper's claim that the dominant
failure is incomplete refactoring, "not a matter of failing to find the right files."

Two distinct conclusions follow, and only the first is a null:

- *Finding-the-entry-point* tooling (goto/sym as a substitute for grep) addresses a
  non-problem here.
- *Reference-completeness* tooling (`lsp refs` before changing a symbol) targets exactly
  the measured failure — and its value is **untested by this study, because the model
  never invoked it** (1 call total). The sharpest summary of §4+§5 together: **the one
  thing the tool is demonstrably good for is the one thing the agent won't use it for.**
  Testing refs' actual value requires forced integration (harness-injected reference
  reports on edited symbols) or retraining, not prompts.

## 6. Finding: acceptance gates didn't help, and probably hurt

Honest decomposition of "the gate arms trailed":

- **D local, −14 pts, p≈0.016** (Holm ≈0.08): confounded — 13 container-wall deaths vs
  A's 4 (§6.5) — and fail-closed by design, so any unrepaired rejection becomes a
  guaranteed empty-patch failure. Evidence about *this gate design under a wall-clock
  budget*, not about type-checking per se.
- **D2 local, −8, p≈0.51**: same wall confound (16 deaths vs 4); not significant.
- **D2 hosted, −10, p≈0.125**: the clean comparison — same serving, same 8h wall, zero
  deaths on both sides — and not significant.

So the defensible claim is: **no gate variant showed benefit; all three point estimates
are negative; the one clean comparison is a non-significant −10.** A caveat on that
clean comparison: budget pressure — our causal story for gate cost — is weakest in the
hosted regime (fast decode, 8h wall), so the hosted pair under-tests the mechanism. We
did not re-run the local pair under an 8h wall for the honest reason that at ~15 tok/s
it is a multi-day run; instead the retroactive measurement below establishes the
mechanism directly, without needing the arm.

Mechanism evidence that the negative direction is real and not just noise:

1. **Repair works; economics don't.** Silent-gate rejections were understood and acted
   on (D2 local: 6/9 gate-fired episodes still resolved; hosted: 12/17). One rejection
   caught a genuinely broken patch (43 new type errors). The loop functions — and still
   doesn't pay.
2. **The gate's signal points the wrong way (measured 2×2).** Retroactively applying
   the identical gate predicate (reject on ≥1 new delta-scoped fingerprint; the "cap 8"
   is display-only) to every golden-valid baseline-A patch:
   **P(flag | patch passed suite) = 29/83 (35%)** — consistent across all four runs
   (29–39%), median 4 new type errors per false-blocked patch (max 43) —
   versus **P(flag | patch failed suite) = 2/14 (14%)** (Fisher exact p≈0.21: no
   evidence the gate discriminates outcomes; the point estimate runs *inverse*).
   The inversion is mechanistically expected given §5: the dominant failure mode is
   under-editing (~⅓ of needed files), and code the agent never wrote produces no new
   diagnostics — **the gate scores the risk of what the agent did, while the benchmark
   punishes what it didn't do.** Flagging also concentrates in 11 of 24 instances,
   suggesting type-messiness is substantially a repo property; if gating anywhere, a
   per-repo policy beats a blanket one.
   (`results/retro_gate.jsonl`, `results/retro_gate_unresolved.jsonl`;
   `code/analysis/retro_gate.sh`.)
3. **A third, gate-free data point.** C2's imperative prompt produced 60 *voluntary*
   `lsp diag` verifications — a non-gate mechanism for the same type-feedback — and C2
   ran −4 vs A (n.s.). In [ianbarber/lsps-for-llms](https://github.com/ianbarber/lsps-for-llms), two distinct
   arms bear on this: the *imperative-prompting* delivery arm ("treat diagnostics as
   squigglies, fix before moving on") **hurt** (−0.131, the only FDR-significant delivery
   effect), while the *acceptance-gate* arm **won** — on seeded single-defect revision
   tasks where the type error was the bug and billability was perfect. Our setting
   (test-scored multi-file refactors, 35% of passing patches type-dirty) sits at the
   opposite billability pole, and both studies' results are consistent with the same
   rule: diagnostic pressure pays exactly in proportion to how billable the diagnostics
   are under the task's scoring.

### 6.5 The 2-hour wall (transferable harness finding)

mini-swe-agent's `environment.container_timeout` defaults to `"2h"`. Episodes exceeding
it lose their container mid-flight and grind remaining steps against "No such container"
until the step cap (one episode logged 188 dead submission attempts). Container-death
counts, local runs, 2 rounds: **A 4 · B 7 · C2 10 · D 13 · D2 16** — monotone in episode
length; nearly every local "step-cap exhaustion" was actually a wall death. At ~15 tok/s
local decode the wall binds hard; at API speeds it doesn't (hosted, 8h: 0 deaths in 4
runs). Any slow-serving agent evaluation on mini-swe-agent defaults should audit this
before interpreting step-cap statistics. See `HARNESS-NOTES.md`.

## 7. Validity: the headline rates need external validation; the serving gap is an aside

Local A resolved 72% [58–83] and hosted A 94% [84–98] — roughly 4–5× the best published
number for this subset and scaffold (§8). The correct prior for a gap that size is a
measurement or setup difference, not a 5× better model. Static checks (same harness,
eval scripts, images, step cap, dataset revision) all match the paper; none of that is
an empirical control. **The decisive check is a reference-model run**: one of the
paper's tied models (GLM-5 is cheapest) through this exact pipeline for one round —
~17% validates the harness; much higher collapses every headline rate here to
internally-paired evidence only. That is follow-up #1 (not yet run; requires an
OpenRouter top-up). The **paired intervention comparisons are unaffected** either way —
identical harness and instances on both sides of every pair.

**Contamination is the leading model-side candidate.** All 29 instances' source commits
(2025-02 → 2026-01) predate the model's 2026-08 release. The informative signal is the
shift: on identical instances under identical scoring, resolved-patch gold overlap moves
from 0.39 (local, medium effort) to 0.62 (hosted, default xhigh), with 18/47 hosted
patches above 0.75 and one verbatim reproduction (transformers-38332). The absolute
level is confounded by task mechanicalness (propagation refactors admit few distinct
correct solutions), but mechanicalness is constant across the comparison — "longer
thinking retrieves memorized commits" fits the shift. A symbol/path perturbation re-run
is follow-up #2.

**The local/hosted gap (72% → 94% on identical everything else) is, for this study's
question, an implementation aside**: it says nothing about LSPs or refactoring — it says
our local serving stack (and a reasoning-effort knob that silently fails to pass through
OpenRouter) understated the model by ~20 points. It matters operationally — calibrate a
local stack against a reference endpoint before attributing anything to interventions —
and is written up with the container-wall finding in `HARNESS-NOTES.md`.

## 8. Benchmark notes (SWE-Bench ProMax, python subset)

- Paper Table 3, python column, mini-swe-agent: best published = **17.2%**
  (GPT-5.2 / Gemini-3-Pro / Kimi-K2.5 / Qwen3.5 tied). Our local 72% [58–83] exceeds
  the field even before the hosted anomaly; the §7 caveats apply to any headline use.
- Golden validity varies by host/network: dspy-9193, dspy-1801, transformers-38788,
  langextract-239 fail their own gold patches on a normal LAN (baked-in corporate
  proxy); transformers-38332 was golden-valid on one worker and not the other. Per-run
  golden re-validation handles this; cross-run analyses must intersect golden-valid sets.

## 9. Verdict

1. Prompting — at any strength tried — does not get this model to adopt semantic
   navigation tools; whether that is tool-unfamiliarity, trigger density, or
   ask-shape is untested (§4). The refactoring failure such tools target (reference
   completeness) is real and measured, but its tooling value is untested because
   adoption never happened (§5). Forced integration or training are the remaining
   levers.
2. **Incomplete refactoring is the dominant failure mode, and it produces no type
   errors** — the study's central observation. Consequently type-check gating in
   test-scored, dynamically-typed settings showed no benefit and trended negative in
   every variant: the gate flagged 35% of passing patches vs 14% of failing ones.
   Reserve gates for settings where diagnostics are billable under the task's scoring. Reserve gates for
   settings where type cleanliness is part of the acceptance criterion.
3. Audit hidden harness budgets (container lifetime vs decode speed) before trusting
   step-cap statistics; quantify your serving stack against a reference endpoint before
   attributing differences to interventions (`HARNESS-NOTES.md`).
4. Treat **both** headline rates (local 72%, hosted 94%) as unvalidated against the
   published field pending the reference-model harness control; the intervention
   comparisons are internally paired on identical harness/instances and survive this
   concern — which is precisely why the study's within-arm conclusions stand regardless.

## 10. Reproduction map

| Path | What |
|---|---|
| `PLAN.md` / `NOTES.md` | Original experiment plan; protocol amendments + run ledger |
| `LABNOTES.md` | Full chronological lab notebook, including failures, fixes, and incidents |
| `HARNESS-NOTES.md` | Standalone summary of the two transferable harness findings |
| `code/agent/` | All arm configs, waved batch runner, image fetch/cache scripts |
| `code/lsp-tool/` | `lsp` CLI + daemon, acceptance-gate scripts (`gate/`), Docker layer build |
| `code/analysis/` | Localization trace analysis; retroactive gate false-block analysis |
| `code/serving/` | GB10 serving notes + exact `.env` |
| `results/` | Per-run per-instance outcomes, summary.csv, localization + retro-gate outputs |
