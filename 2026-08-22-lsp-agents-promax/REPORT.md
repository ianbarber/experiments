# Do LSP tools help a local coding agent refactor? (Qwen3.8-27B × SWE-Bench ProMax)

**Dates:** 2026-08-22 → 2026-08-28 · **Fleet:** DGX Spark (GB10, aarch64) serving + 2× x86
workers + NAS · **Benchmark:** SWE-Bench ProMax (arXiv 2608.09802), python subset (29
instances) · **Scaffold:** mini-swe-agent 2.4.6 · **Model:** Qwen3.8-27B (FP8 local /
OpenRouter fp8 hosted)

## TL;DR

Across seven arm-configurations and 406 evaluated episodes: **no way of providing LSP
capability to this model improved resolve rate, and every acceptance-gate variant made it
worse.** The model never adopts navigation tools (1 call in ~180 tool-equipped episodes);
imperative instructions buy only `lsp diag` compliance; a submission-time type-check gate
— hard or silent-soft, local or hosted — always trailed its baseline. Trace analysis
shows why the navigation offer is useless here (localization is already instant; failures
are incomplete reference propagation) and why the gate backfires (on dynamically-typed
repos, new type errors frequently don't fail the test suite, so the gate spends budget
fixing unbilled problems). Two side-findings matter beyond the LSP question: a **2-hour
container-lifetime wall** in mini-swe-agent silently killed long episodes and confounded
early gate results, and the same model scored **72% locally vs 94% hosted** on identical
instances — serving stack and reasoning-effort defaults move this benchmark more than any
intervention we tested.

## 1. Setup

- **Benchmark**: SWE-Bench ProMax — 170 expert-curated multilingual refactoring
  instances; we ran the 29-instance python subset. Gold patches avg 11.4 source files.
  Resolve = full test suite passes in the instance's Docker image; the harness re-validates
  the golden patch per run and drops golden-invalid instances from the denominator
  (2–5 per run on our LAN; several images bake in an unreachable ByteDance proxy —
  `sys-proxy-rd-relay.byted.org` — which breaks their network-dependent tests anywhere
  outside the authors' infra).
- **Model**: Qwen3.8-27B (dense hybrid Gated-DeltaNet, released 2026-08-13; HF arch
  `qwen3_5`). Local: official FP8 checkpoint on the GB10 via `lmsysorg/sglang:qwen38-27b`
  + DSpark speculative decoding (see `code/serving/SERVING.md`; pip vLLM/SGLang are dead
  ends on sm_121). Hosted: OpenRouter pinned to CoreWeave/Parasail (fp8).
- **Protocol**: mini-swe-agent (tool-calling mode), 300-step cap, temp 1.0 / top_p 0.95 /
  top_k 20 (model-card thinking defaults; the paper doesn't state temperature),
  `reasoning_effort: medium` locally (passthrough to hosted providers does not work —
  hosted ran at default xhigh; see §7). Two rounds per arm, hosts swapped between rounds.
  Paired per-instance analysis on the intersection of golden-valid sets, exact
  McNemar/sign tests.
- **LSP tooling**: `code/lsp-tool/` — a bash CLI (`lsp def/refs/hover/sym/outline/diag/
  rename --dry-run`) over a warm daemon wrapping solidlsp (serena-agent 1.7.0), Pyrefly
  1.2.0 as the Python server, content-enriched output (±4 lines context per result).
  Baked into per-instance image variants; verified in-container (refs 2.6 s cold, ~50 ms
  warm).

## 2. Arms

| Arm | Intervention |
|-----|--------------|
| A | Baseline (stock template) |
| B | `lsp` CLI present + neutral tool docs in prompt |
| C (pilot) | B + polite "prefer lsp" paragraph — **abandoned after 5 episodes: 0 lsp calls** |
| C2 | B + imperative, workflow-integrated instruction (lsp-first analysis; refs before change; diag before submit) |
| D | A's prompt + `submit` wrapper running a **delta-scoped Pyrefly acceptance gate** (baseline snapshot at episode start; only new error fingerprints count; cap 8; reject blocks submission) |
| D2 | **Silent soft gate**: prompt byte-identical to A; a BASH_ENV `echo`-function intercepts the submission marker and runs the same gate; after 2 rejections the next attempt passes silently |
| A/D2 hosted | A and D2 re-run via OpenRouter with `container_timeout: 8h` (see §6) |

## 3. Results

**Local (GB10 serving), golden-valid python, 2 rounds each:**

| Arm | Resolved | Rate | Paired vs A |
|-----|----------|------|-------------|
| A | 36/50 | 72% | — |
| B | 35/49 | 71% | 3-vs-4, p=1.0 |
| C2 | 34/50 | 68% | 4-vs-2, p≈0.69 |
| D | 29/50 | 58% | **7-vs-0, p≈0.016** |
| D2 | 32/50 | 64% | 6-vs-3, p≈0.51 |

**Hosted (OpenRouter fp8, 8h wall), 2 rounds each:**

| Arm | Resolved | Rate | Paired |
|-----|----------|------|--------|
| A hosted | 47/50 | 94% | — |
| D2 hosted | 42/50 | 84% | A-better 6-vs-1, p≈0.125 |

Per-run detail in `results/summary.csv`; per-instance outcomes in `results/*.json`.

## 4. Finding: navigation tools are unadoptable by prompting

- **B (available + documented): 0 lsp calls in 58 episodes** (~5,000+ commands). Tool
  docs verified present in every prompt.
- **C pilot (polite preference): 0 calls in 5 episodes** — instruction verified present.
- **C2 (imperative): 61 calls across 58 episodes — 60 `lsp diag`, 1 `lsp refs`.** The
  model complies with the concrete, checkable clause ("run diag on edited files") and
  wholly ignores the navigation-habit replacement ("use refs before changing a symbol"),
  despite it being phrased as IMPORTANT/ALWAYS.

This extends arXiv 2608.13568's frontier-model finding (0–6% unprompted semantic-tool
use) to an open 27B, and sharpens it: **instruction strength moves compliance only for
verification-shaped asks, never navigation-shaped ones.** If you want this model using
refs, prompting is not the lever; retraining or forced integration is.

## 5. Finding: localization was never the problem — recall is

Trace analysis over round 1 (`code/analysis/localization.py`, output in
`results/localization-round1.txt`):

| Split | first gold-file touch (step) | edit precision | edit recall | gold files missed |
|-------|------------------------------|----------------|-------------|-------------------|
| A resolved | 1.5 | 0.97 | 0.76 | 1.9 |
| A unresolved | 0.7 | 1.00 | 0.32 | 6.2 |
| B/C2 (same pattern) | 0.8–2.1 | 0.94–1.00 | 0.74–0.75 / 0.35–0.43 | 2.4–2.7 / 3.0–4.9 |

The agent touches a gold-patch file within ~2 steps in **every** episode (0 never-found
cases, ~1 search before contact) and almost never edits a wrong file. Failures edit the
right files but cover only ~⅓ of the needed set. This independently confirms, at trace
level, the ProMax paper's §5.2 claim that the dominant failure is incomplete refactoring
and "not a matter of failing to find the right files" — and it means steering models
toward navigation tools attacks a non-problem on this benchmark.

## 6. Finding: acceptance gates always trail — and why

Every gate variant lost to its baseline: local hard gate −14 pts (the study's only
significant effect), local silent soft gate −8, hosted silent soft gate −10. Telemetry:

- Hard gate (D): rejections in 11/58 episodes; rejected episodes averaged 198 steps and
  mostly died at the cap; repair-pass 2/5 golden-valid.
- Silent soft gate (D2 local): fired 9/58; **repair-pass 6/9** — the self-explanatory
  rejection is enough for the model to fix and resubmit. Hosted D2: fired 17/58,
  repair-pass 12/17.
- One legitimate catch: a rejection with 43 real new type errors (cv2 enum misuse).

So the *mechanism* works — and still loses. Two causes:

1. **Wall/budget interaction** (partially artifactual — §6.5): gate episodes run longer
   and disproportionately crossed the 2-hour container wall in local runs.
2. **Type errors ≠ test failures** (fundamental, survives the wall fix): on
   dynamically-typed repos with outcome-focused suites, baseline arms regularly submit
   patches carrying new type diagnostics that *pass the tests anyway*. The gate converts
   those into repair detours (or, unrepaired, into certain empty-patch failures), paying
   real steps for problems the benchmark never charges. This matches the delivery-grid
   result in [ianbarber/lsps-for-llms](https://github.com/ianbarber/lsps-for-llms)
   (imperative diagnostic-fixing was the only significant arm, at −0.131) — the gate's
   value inverts between "seeded single-defect revision at 60 steps" and "300-step
   multi-file refactoring under wall-clock pressure".

### 6.5 The 2-hour wall confound

mini-swe-agent's `environment.container_timeout` defaults to `"2h"`. Episodes exceeding
it lose their container mid-flight and grind remaining steps against "No such container"
until the step cap (one episode logged 188 dead submission attempts). Container-death
counts, local runs, 2 rounds: **A 4 · B 7 · C2 10 · D 13 · D2 16** — monotone in episode
length, and nearly every "step-cap exhaustion" in the study is actually a wall death. At
~15 tok/s local decode this binds hard; at API speeds it doesn't (hosted reruns with an
8h setting: **0 deaths in 4 runs**). Any slow-serving agent evaluation using
mini-swe-agent defaults should check this before interpreting step-cap statistics.

## 7. Finding: the serving stack moves this benchmark more than any intervention

Hosted A resolved **94%** vs local A **72%** on identical instances, prompts, and
scaffold. The wall fix explains only ~2 slots. Remaining suspects, unresolved between:
(a) local quality loss (DSpark speculative decoding, fp8 KV cache, sm121 triton kernels);
(b) hosted running default **xhigh** reasoning (the `reasoning_effort`/
`chat_template_kwargs` knob does not pass through OpenRouter — verified) vs local pinned
medium. A local xhigh A-run would separate these; not run (see cost note).

Two caveats attach to the 94%:
- **Contamination**: all 29 python instances' source commits (2025-02..2026-01) predate
  the model's Aug-2026 release. Local resolved patches reproduce gold added-lines at mean
  0.39 (most successes are materially different solutions; one instance,
  transformers-38332, is a 1.00 verbatim gold reproduction). **Hosted resolved patches
  overlap gold at mean 0.62, with 18/47 above 0.75** — deeper thinking appears to partly
  be *recall*. No resolve-rate gradient by commit date exists in either setting, and the
  paper's own 2026 models saw most of the same commits yet scored ≤17.2% on this subset
  with this scaffold — so contamination cannot carry the whole effect, but the hosted
  headline should be quoted with this attached.
- **Cost incident**: because the effort knob doesn't pass through, hosted episodes ran
  xhigh; reasoning bills as output ($2.40–3.40/M) and the 4-run revalidation consumed
  roughly **$1.4–1.7k** of OpenRouter credit vs a ~$100 estimate made assuming medium.
  Lesson encoded in LABNOTES: probe 1–2 episodes and read actual billed usage before
  launching hosted fleets.

## 8. Benchmark notes (SWE-Bench ProMax, python subset)

- Paper Table 3, python column, mini-swe-agent: best published model = **17.2%**
  (GPT-5.2/Gemini-3-Pro/Kimi-K2.5/Qwen3.5 tied). Our Qwen3.8-27B: 72% local / 94% hosted
  (62% / 90% on an all-29 denominator) — a generational jump consistent with its
  vendor-reported SWE-bench-Pro 61.7, with the §7 contamination caveat.
- Golden-patch validity varies by host/network: dspy-9193, dspy-1801,
  transformers-38788, langextract-239 fail golden validation on a normal LAN (baked-in
  corporate proxy). transformers-38332 was golden-valid on one worker and not the other.
  The harness's per-run golden re-validation handles this correctly; cross-run analyses
  must intersect golden-valid sets.

## 9. Verdict

For a strong open-weight 27B on multi-file python refactoring:

1. **Don't bother shipping LSP navigation tools to the agent via prompt** — availability
   is inert, instructions don't transfer the habit, and the failure mode they'd address
   (localization) isn't the bottleneck.
2. **Don't gate submission on a type checker** in dynamically-typed, test-scored
   settings — repair loops work mechanically but the economics are negative. If coverage
   feedback is worth pursuing, it should target the actual failure (unpropagated
   references), not type cleanliness.
3. **Audit your harness's hidden budgets** (container lifetime vs decode speed) before
   trusting step-cap statistics.
4. **Serving configuration is a first-order variable** — quantify your stack against a
   reference endpoint before attributing differences to interventions.

## 10. Reproduction map

| Path | What |
|---|---|
| `PLAN.md` / `NOTES.md` | Original experiment plan; protocol amendments + run ledger |
| `LABNOTES.md` | Full chronological lab notebook, including failures, fixes, and incidents |
| `code/agent/` | All arm configs (`arm_*.yaml`), waved batch runner (disk-bounded load→rollout→eval→evict), image fetch/cache scripts |
| `code/lsp-tool/` | The `lsp` CLI + daemon (solidlsp/Pyrefly), acceptance-gate scripts (`gate/`), per-instance Docker layer build (`docker/`) |
| `code/analysis/` | Localization/thrash trace analysis |
| `code/serving/` | GB10 serving notes + the exact `.env` used |
| `results/` | Per-run per-instance outcomes (slim), summary.csv, localization output |
