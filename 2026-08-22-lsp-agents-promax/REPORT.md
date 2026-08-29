# Do LSP tools help a local coding agent refactor? (Qwen3.8-27B × SWE-Bench ProMax)

**Dates:** 2026-08-22 → 2026-08-28 · **Fleet:** DGX Spark (GB10, aarch64) serving + 2× x86
workers + NAS · **Benchmark:** SWE-Bench ProMax (arXiv 2608.09802), python subset (29
instances) · **Scaffold:** mini-swe-agent 2.4.6 · **Model:** Qwen3.8-27B (FP8 local /
OpenRouter fp8 hosted)

## TL;DR

- **Provisioning LSP navigation to this model did nothing, at any instruction strength**
  — 1 navigation call in ~180 tool-equipped episodes. What the data shows is
  *non-adoption*, not mechanistic uselessness: the completeness tool (`lsp refs`) targets
  exactly the measured failure mode, and the agent never picks it up (§4–5).
- **Type-check acceptance gates didn't help and probably hurt**: three same-direction
  point estimates (−14 local hard, −8 local soft, −10 hosted soft); only the local hard
  gate is nominally significant, and it is partly a fail-closed design artifact (§6).
  Retroactive measurement: the gate would have rejected **35% (29/83)** of patches that
  passed the full test suite — type cleanliness and test success are nearly orthogonal
  here.
- **Two transferable harness findings** (see also `HARNESS-NOTES.md`): mini-swe-agent's
  default 2h `container_timeout` silently kills long episodes and masquerades as step-cap
  exhaustion (§6.5); and the same model/scaffold/instances scored **72% local vs 94%
  hosted** — serving configuration and reasoning-effort defaults move this benchmark more
  than any intervention we tested (§7).
- The hosted 94% is an **anomaly requiring validation, not a headline**: it is 5× the
  best published number for this subset/scaffold, and hosted resolved patches overlap
  gold added-lines at 0.62 mean (18/47 > 0.75) — close to reproduction on 11-file
  refactors (§7).

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
  columns carry 95% Wilson intervals — at n≈50 these are ±11–13 points, which is the
  right lens for reading 3–8 point gaps.
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
| D | A's prompt + `submit` wrapper running a delta-scoped Pyrefly acceptance gate (fail-closed: rejection blocks submission) |
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

## 5. Finding: entry-point localization is a non-problem; reference completeness is the
failure — and it went untested

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
are negative; the one clean comparison is a non-significant −10.** We did not re-run the
local pair under an 8h wall: the hosted pair already provides a wall-free, within-regime
A-vs-D2 comparison, and a local repeat would only probe whether the (consistent)
direction generalizes across serving regimes.

Mechanism evidence that the negative direction is real and not just noise:

1. **Repair works; economics don't.** Silent-gate rejections were understood and acted
   on (D2 local: 6/9 gate-fired episodes still resolved; hosted: 12/17). One rejection
   caught a genuinely broken patch (43 new type errors). The loop functions — and still
   doesn't pay.
2. **Type errors ≠ test failures (measured).** Retroactively applying the gate to every
   *resolved* baseline-A patch (local + hosted, both rounds): **the gate would have rejected 29 of 83 (35%), consistently across all four runs (29–39%), with a median of 4 new type errors per false-blocked patch (max 43), spanning 11 of 24 distinct instances** (`results/retro_gate.jsonl`; `code/analysis/retro_gate.sh`).
   Each of those is a patch the benchmark scores as success that the gate would have
   turned into repair detours or, unrepaired, a certain failure. On dynamically-typed
   repos with outcome-focused suites, type cleanliness is close to orthogonal to the
   scoring criterion.
3. **A third, gate-free data point.** C2's imperative prompt produced 60 *voluntary*
   `lsp diag` verifications — a non-gate mechanism for the same type-feedback — and C2
   ran −4 vs A (n.s.). Together with the −0.131 imperative-diagnostic arm in
   [ianbarber/lsps-for-llms](https://github.com/ianbarber/lsps-for-llms) (the only
   FDR-significant arm in that delivery grid), that is two studies and three mechanisms
   pointing the same way: on tasks scored by tests, pushing this class of model to chase
   type diagnostics costs more than it saves. The regime matters: the same gate design
   *won* in lsps-for-llms on seeded single-defect revision, where the type error was the
   bug — perfect billability.

### 6.5 The 2-hour wall (transferable harness finding)

mini-swe-agent's `environment.container_timeout` defaults to `"2h"`. Episodes exceeding
it lose their container mid-flight and grind remaining steps against "No such container"
until the step cap (one episode logged 188 dead submission attempts). Container-death
counts, local runs, 2 rounds: **A 4 · B 7 · C2 10 · D 13 · D2 16** — monotone in episode
length; nearly every local "step-cap exhaustion" was actually a wall death. At ~15 tok/s
local decode the wall binds hard; at API speeds it doesn't (hosted, 8h: 0 deaths in 4
runs). Any slow-serving agent evaluation on mini-swe-agent defaults should audit this
before interpreting step-cap statistics. See `HARNESS-NOTES.md`.

## 7. The local/hosted gap dwarfs every intervention — and the hosted number is an
anomaly to validate, not a result to quote

Hosted A resolved 94% vs local A 72% on identical instances, prompts, and scaffold. The
wall fix explains ~2 slots. We could not separate the remaining suspects: (a) local
serving quality loss (DSpark speculative decoding, fp8 KV cache, sm121 triton kernels);
(b) hosted default **xhigh** reasoning vs local pinned medium (the effort knob does not
pass through OpenRouter — verified). A local xhigh run would separate them; at local
decode speeds with xhigh thinking volumes it is a multi-day run and was not prioritized.

**Why the 94% demands validation before belief**: it is ~5× the best published number
for this subset and scaffold (§8). The correct prior for a 5× gap is a measurement
difference, not a 5× better model. What we have checked: same eval harness and eval
scripts as the paper's repo, same images, same 300-step cap, same dataset revision;
golden-validation drops (our denominator excludes 4 instances that fail their own gold
patch on our network — the paper's infra would not drop these; on an all-29 denominator
hosted A is 90%, still ~5×). What we have not checked: the paper's sampling temperature
(unstated) and any provider-side serving differences.

**Contamination is the leading candidate.** All 29 instances' source commits (2025-02 →
2026-01) predate the model's 2026-08 release. Local resolved patches overlap gold
added-lines at mean 0.39 — mostly materially different solutions (one verbatim 1.00
reproduction: transformers-38332). **Hosted resolved patches overlap at mean 0.62,
median 0.61, with 18/47 above 0.75 — on refactors averaging 11 files, that is close to
reproduction, and "longer thinking retrieves memorized commits" fits it.** We note the
weaknesses of our own counter-evidence: the no-date-gradient observation has no power
(every instance is inside the contamination window — there is no control arm), and
"the paper's 2026 models saw the same commits" assumes cross-lab uniformity of training
data, which is exactly what can't be assumed. The decisive cheap test is **perturbation**:
rename symbols / move paths on 5–8 instances and re-run hosted A (≈$35–50 at xhigh; the
current OpenRouter balance (~$16) doesn't cover it — flagged as the first follow-up).

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
2. Type-check gating in test-scored, dynamically-typed settings: no variant helped, all
   trended negative, and the measured orthogonality of type errors to test outcomes
   (35% false-block rate on passing patches) explains why. Reserve gates for
   settings where type cleanliness is part of the acceptance criterion.
3. Audit hidden harness budgets (container lifetime vs decode speed) before trusting
   step-cap statistics; quantify your serving stack against a reference endpoint before
   attributing differences to interventions (`HARNESS-NOTES.md`).
4. Treat the hosted 94% as unvalidated pending the perturbation test.

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
