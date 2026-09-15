# Lab notebook — LSP tools on SWE-Bench ProMax

Chronological record: what was done, what was observed, what was decided, in
order. Protocol reference lives in NOTES.md; the plan in PLAN.md. Times are
`dgx-spark` local (UTC-7); container/server logs print UTC (+7h).

---

## 2026-08-22 (Sat) — planning & investigation

- **~19:10** Project start. Serving box (`dgx-spark`): NVIDIA GB10, aarch64,
  20 ARM cores (10× Cortex-X925 + 10× A725), 119 GiB unified memory (~273 GB/s),
  CUDA 13.0, driver 580.159, Docker 29.2.1. GPU idle; nothing else to stop.
- Research (3 parallel agents):
  - **Benchmark**: SWE-Bench ProMax, arXiv 2608.09802, COLM 2026, released
    2026-08-10. 170 instances / 70 repos / 7 languages (py 29, ts 28, java 26,
    go 23, c++ 22, rust 22, c 20). Paper baselines: mini-swe-agent — Qwen3.5-MoE
    20.6%, Sonnet 4.6 30.6; OpenHands — GPT-5.2 41.2. Images
    `key4127/refactor-dockerhub:<id>`, **amd64-only**, ~317 GB compressed.
  - **Model**: Qwen3.8-27B is real (released 2026-08-13/14), dense hybrid
    48× Gated DeltaNet + 16× gated attention, 262k ctx, multimodal, MTP head,
    Apache 2.0. HF `model_type: qwen3_5`. Official FP8 checkpoint ~29 GB.
    Default reasoning_effort=xhigh known to overthink → we use medium.
  - **LSP landscape**: Serena/solidlsp (40+ languages, pyrefly pluggable),
    Pyrefly 1.2.0 GA. Key prior work arXiv 2608.13568: LSP tools usually hurt
    token budgets for localization, help for reference-completeness/rename, and
    only with content-enriched output; frontier models use LSP 0–6% unprompted.
    No published LSP ablation with a local open-weight model.
- PLAN.md written.
- **Lab topology**: `strix-halo` (32T/122G/409G free, no docker, other local
  serving left running), `worker-a` (16T/29G/147G, docker), `worker-b`
  (16T/125G/80G, docker), NAS `nas:/Public` → `/mnt/nas` everywhere, 6.8 TB
  free. Plan: spark serves; x86 hosts run containers natively; NAS = image tar
  cache + run archive. QEMU/cloud path dropped.

## 2026-08-22 — Phase 0/1 execution

- **21:16** Lab probes confirm. `worker-a` picked as primary worker
  (`strix-halo` busy, no docker). NAS cache flow validated: dspy image
  docker-save → 752 MB zst in 15 s (~50 MB/s).
- **21:27** `Qwen/Qwen3.8-27B-FP8` download → landed on NAS (`HF_HOME` points
  there), 6 min.
- Harness cloned; dataset: 170 instances parse; fields incl. `image_name` ✓.
- **Gold gate #1** (dspy-9193): golden patch FAILS — `test_enable_net_flag`,
  CodeInterpreterError, model==golden failure → environment, not patch. →
  env-flaky list.
- **Gold gate #2** (albumentations-2337): SUCCESS/SUCCESS in 42 s. **Eval
  pipeline valid.**
- mini-swe-agent 2.4.6 on `worker-a`; `mini-extra swebench --subset
  swe-bench-promax/SWE-Bench-ProMax --split test` loads directly, honors
  `image_name` — planned adapter unnecessary.
- **Serving saga on `dgx-spark` (aarch64/sm121)** — chronological:
  1. vLLM nightly pip: **no aarch64 wheels** — dead.
  2. SGLang 0.5.9 pip: installs; torch resolved to CPU wheel → reinstalled
     cu130 → sgl-kernel linked against libnvrtc.so.**12** → reinstalled cu129
     → imports OK.
  3. Launch #1: cuDNN guard (torch 2.9.1 Conv3d bug) → pin
     nvidia-cudnn-cu12==9.16.0.29.
  4. Launch #2: hybrid-GDN models on Blackwell demand explicit attention
     backend.
  5. Launch #3 (`--attention-backend triton`): triton's bundled ptxas predates
     **sm_121a** → `TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas`.
  6. Launch #4: CUDA graph capture: GemmaRMSNorm "no kernel image" —
     sgl-kernel's sm100 cubins genuinely can't run on sm_121. pip route DEAD.
  7. **MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark** (image
     `lmsysorg/sglang:qwen38-27b`, GB10-tuned). Vendored at
     `serving/spark-sglang/`, knobs: QUANT=fp8, MAX_CONCURRENT_REQUESTS=8.
     `./start-dspark.sh` (DSpark spec decode, flashinfer, fp8 KV, CPU pinning
     to X925).
  - Side lessons: rsync of HF snapshots needs `-aL` (symlinks dangle); disk
    hit 98% → deleted duplicate weight copies + dead venvs (41 GB back).
    GGUF Q8_0 + MTP draft downloaded to NAS as fallback.
- **~21:55** Server READY. Sanity: correct code + arithmetic,
  reasoning_content separated, **31.7 tok/s** single-stream short-context
  code; effort control via chat_template_kwargs ✓.
- **22:20** First smoke attempt: mini-swe-agent RuntimeError → cost tracking
  on unknown model → `MSWEA_COST_TRACKING=ignore_errors`.
- **22:24 → 23:42 Smoke episode** (albumentations-2337, arm A): Submitted, 80
  steps, 78 min, 26 KB patch → **RESOLVED by official harness** (golden also
  success). Full pipeline proven; near-0% baseline risk eliminated.

## 2026-08-23 (Sun, early) — dev-10 shakeout

- dev-10 (1/language + extras, arm A, 3 workers, `worker-a`): rollout ~2.2 h,
  all episodes ran. 8/10 Submitted (patches 2–71 KB), 2 LimitsExceeded @300
  steps with empty patches (2337 — which the smoke run had solved: temp-1.0
  variance; ruff-21445).
- Eval: **3/9 golden-valid resolved (33%)** — maven ✓ (Java), cli ✓ (Go),
  s2n-tls ✓ (C); 0 python/ts/rust/c++. transformers-38788 golden-invalid (#2
  on flaky list). Reference: paper mini-swe-agent frontier ~21–31% overall.
- LSP tool built: `lsp-tool/` CLI+daemon over solidlsp (serena-agent 1.7.0,
  pyrefly 1.2.0 / basedpyright 1.39.10), content-enriched output, 26/26 e2e
  on both servers, warm refs 52–81 ms.

## 2026-08-23 — stage 1 launches & infrastructure fixes

- **s1-arm-a-r1 launch #1 FAILED — disk**: flat pre-fetch of 29 python images
  blew `worker-a`'s 147 GB (verl image bundles CUDA/nvshmem). Evicted images
  (all NAS-cached), wrote **`agent/run_batch_waved.sh`**: per-wave load →
  rollout → eval → evict; eval always grades in ORIGINAL images. Relaunched
  (waves of 6, 3 workers).
- **LSP image layers** (`worker-b`): pilot failed (no py≥3.11, no uv in image)
  → uv bootstrap in Dockerfile → pilot OK (829 MB zst; in-container refs
  2.6 s cold). Batch of 29: 20 OK + failures in 3 classes:
  1. **Baked-in ByteDance proxy** (`sys-proxy-rd-relay.byted.org`) in several
     images kills all network → `env -u` proxy neutralization for build steps
     only (final env unchanged). Also the likely root cause of the
     golden-invalid instances.
  2. Failed `python -m venv` leaves partial dir uv won't reuse → rm -rf
     before fallback.
  3. Verify picked first tracked .py = empty `__init__.py` → pick largest
     file instead.
  Sweep of 9 failures after fixes: **9/9 OK → 29/29 LSP images on NAS.**
- **s1-arm-b-r1** (arm B, LSP available, `worker-b`, lsp waves of 4) launched
  concurrently with arm A against the same server (removes serving drift;
  host effect counterbalanced in round 2 by swapping hosts).
- Server under load: 6 concurrent episodes, ~14–15 tok/s per stream at 40k+
  ctx.

## 2026-08-23 — stage 1 round 1 results (running)

- **Arm A (baseline) COMPLETE: 19/25 resolved (76%) on golden-valid python**
  (4 golden-invalid: transformers-38788, dspy-9193, dspy-1801, langextract-239
  — all consistent with the proxy issue). Audited deepeval-2345 pass: genuine
  (20 KB multi-file patch, suite green, exit 0).
- **Arm B (LSP available): ZERO `lsp` calls** through 16+ episodes (~1500+
  commands); tool docs verified present in prompts. Replicates 2608.13568
  unprompted-usage finding on an open 27B. Resolve through wave 4: 8/12 (67%).
  transformers-38332 golden-valid on `worker-a` but golden-invalid on
  `worker-b` — golden validity varies by host; paired analysis must intersect
  golden-valid sets. (Correction: an early "daemon running in arm B"
  observation was a self-matching grep artifact.)
- **Arm C (LSP preferred) launched** on `worker-a` after arm A. First
  completed episode (transformers-38788, 56 cmds) used 0 — adoption is
  per-episode heterogeneous.

## 2026-08-23 — Arm C pilot result & protocol revision

- **Arm C wave-1 pilot (5 episodes, ~800 commands): ZERO `lsp` calls** —
  preference paragraph verified present. The polite "prefer the lsp tool"
  clause is behaviorally inert on this model; C-as-written would have
  replicated A.
- (Second live-check correction: "pyrefly processes running in C containers"
  was another self-matching pipeline artifact — grep's own cmdline contained
  the pattern. No LSP daemon had run in any arm's containers. Lesson: process
  checks must use patterns that cannot appear in the checking command, e.g.
  `[p]yrefly`.)
- **Decision**: stop s1-arm-c-r1 after wave 1 (archived as `s1-arm-c-r1-pilot`;
  2 of 5 episodes hit the step cap), and strengthen the manipulation →
  `agent/arm_c2.yaml`: imperative, workflow-integrated instruction (lsp-first
  analysis; `lsp refs` required before changing a symbol and again before
  patch creation; `lsp diag` on edited files). Still tests the original arm-3
  intent ("prompt instructing to prefer the LSP"), but strong enough to
  produce a manipulation at all. Relaunched as **s1-arm-c2-r1**.
- For a 27B agent, *availability* (B) and *polite preference* (C-pilot) both
  yield 0% adoption — instruction strength is a first-order variable.

## 2026-08-23 — Arm B round 1 complete; paired A↔B

- **Arm B FINAL: 18/24 (75%) golden-valid; 0 `lsp` calls across all 29
  episodes.**
- **Paired A↔B on the 24 co-golden-valid instances: A 18/24, B 18/24 — exact
  tie.** Discordant: only-A = {albumentations-2337, albumentations-2495};
  only-B = {lerobot-2808, verl-3915}; McNemar p=1.0.
- Never-solved by either arm: django-19643, langchain-32996, gallery-dl-7872,
  dspy-9047 — prime targets for judging C2's effect.
- **First C2 episode** (transformers-38788): 2 lsp calls (`diag` ×2 of 80
  cmds) — imperative prompt produces nonzero adoption; used for verification,
  not navigation.
- Round 2 counterbalance started: **s1-arm-a-r2 on `worker-b`**.

## 2026-08-24 — round 1 complete (all three arms)

**Resolve rates, golden-valid python (n≈25):**

| Arm | Resolved | Rate |
|-----|----------|------|
| A baseline | 19/25 | 76% |
| B lsp-available | 18/24 | 75% |
| C2 lsp-imperative | 16/25 | 64% |

**Paired (exact McNemar):** A↔B 2 vs 2 (p=1.0); A↔C2 4 vs 1 (p≈0.375); B↔C2
2 vs 0 (p≈0.5). No significant differences at this n; point estimates lean
against the imperative-LSP arm.

**Usage:** across 63 episodes with the tool present (B 29 + C2 29 + C-pilot
5): **navigation subcommands (refs/def/sym/outline/hover/rename) were used
exactly 0 times**. C2's 21 total calls (16/29 episodes) were 100% `lsp diag`.

Interim: for Qwen3.8-27B on ProMax python, (1) LSP availability is
behaviorally inert; (2) imperative prompting buys only verification-style
diag calls and a (non-significant) resolve-rate *decrease*. Round 2 doubles n.

## 2026-08-24 — localization/thrash analysis

Question: is the model going to the wrong file — would steering toward
navigation tools help? **Answer: no wrong-file thrash exists.**

| Split | steps | 1st gold-file touch (step) | searches pre-gold | edit precision | edit recall | gold files missed |
|-------|------|---------------------------|-------------------|----------------|-------------|-------------------|
| A resolved (19) | 47 | 1.5 | 1.3 | 0.97 | 0.76 | 1.9 |
| A unresolved (6) | 82 | 0.7 | 0.8 | 1.00 | 0.32 | 6.2 |
| B resolved (18) | 33 | 0.8 | 1.0 | 0.94 | 0.75 | 2.7 |
| B unresolved (6) | 104 | 1.8 | 2.2 | 0.97 | 0.43 | 3.0 |
| C2 resolved (16) | 43 | 1.1 | 1.2 | 0.94 | 0.74 | 2.4 |
| C2 unresolved (9) | 131 | 2.1 | 1.7 | 1.00 | 0.35 | 4.9 |

- The agent touches a gold file within ~1–2 steps in every episode.
- Edit precision ≈ 0.94–1.00: it essentially never edits a wrong file.
- **The failure mode is recall**: unresolved episodes edit the right files
  but only ~1/3 of the needed set. That is the paper's "incomplete
  refactoring" failure — and precisely what `lsp refs` addresses, and
  precisely the subcommand the model refuses to call.
- **Implication**: retraining toward *navigation* attacks a non-problem. The
  mechanical hook is reference-coverage feedback. → Arm D design. Converges
  with the earlier [lsps-for-llms](https://github.com/ianbarber/lsps-for-llms)
  finding that a submission-time type-check gate helped where forced
  integration is tricky.

## 2026-08-24 — prior LSP-gate report digested; Arm D built

- Digest of `ianbarber/lsps-for-llms`: submission-time **blocking gate**
  blocked 11/12 bad completions (vs 2/12 one-shot); the only FDR-significant
  delivery effect was the imperative "treat as squigglies, fix now" sentence
  at **−0.131** — independently explains our C2 trend. Their mini-swe-agent
  case study reproduces our zero-election exactly.
- **Arm D built** (`agent/arm_d.yaml` + `lsp-tool/gate/`): arm-A prompt with
  only the Submission step changed to a `submit` wrapper; wrapper runs a
  **delta-scoped Pyrefly acceptance gate** (baseline at episode start; only
  NEW fingerprints count; fail-closed). Matches the recall failure mode:
  stale references left by incomplete renames surface as new diagnostics in
  unmodified files.
- Bench test in promax-lsp:albumentations-2337: baseline 64 KB of
  pre-existing fingerprints (delta scoping essential); clean submit → marker
  passes; injected `bad-assignment` → precise rejection, rc=1.

## 2026-08-25 — rounds 1+2 complete for A/B/C2

**Two-round totals, golden-valid python (hosts counterbalanced):**

| Arm | Resolved | Rate |
|-----|----------|------|
| A baseline | 36/50 | 72% |
| B lsp-available | 35/49 | 71% |
| C2 lsp-imperative | 34/50 | 68% |

Per-instance 2-round sign tests: A↔C2 4-vs-2 (p≈0.69), A↔B 3-vs-4 (p=1.0),
B↔C2 3-vs-2 (p=1.0). **No effect of LSP provision or instruction on resolve
rate, ±~4 pts.** Navigation subcommands used **once** in ~120 tool-equipped
episodes (a single `lsp refs` in C2-r2); everything else `lsp diag`.

- **Arm D r1** running on `worker-a` (wave-1: 2 rejections in 1/5 episodes;
  the rejected episode hit the step cap — rejection-loop budget burn is the
  failure mode to watch). **D-r2 launched on `worker-b`**.

## 2026-08-25 — Arm D round 1 complete

- **D-r1: 13/25 (52%)** — 20 pts below A-r1 (76%) on the same host. Gate
  telemetry: 5/29 episodes drew ≥1 rejection (mean 198 steps, 3/5 hit the
  step cap, 40% resolved); never-rejected episodes ran 55% (11/20) with 5
  step-caps — total 8 LimitsExceeded vs A-r1's 2.
- The gate converts "submit a type-broken patch that might still pass tests"
  into "certain failure (empty patch)" whenever repair doesn't converge
  within budget. A soft gate would bound that cost. Also candidate: the gate
  note in the submission instructions may induce over-cautious
  polish-instead-of-submit (anticipatory effect, cousin of the C2 trend and
  of the −0.131 imperative-sentence result in lsps-for-llms).

## 2026-08-26 — STAGE 1 COMPLETE: final four-arm results

**Combined two rounds, golden-valid python, hosts counterbalanced:**

| Arm | Resolved | Rate | Paired vs A (sign test) |
|-----|----------|------|--------------------------|
| A baseline | 36/50 | 72% | — |
| B lsp-available | 35/49 | 71% | 3-vs-4, p=1.0 |
| C2 lsp-imperative | 34/50 | 68% | 4-vs-2, p≈0.69 |
| **D acceptance gate** | **29/50** | **58%** | **7-vs-0, p≈0.016** |

- **The only significant effect in the study is the gate hurting.** All
  prompting interventions are nulls.
- Regime note vs the earlier gate result: that gate won on 60-step revision
  tasks, temp 0, seeded single defects; this is 300-step full refactor
  episodes at temp 1.0 on 11-file gold patches. The gate's value inverts
  across regimes.
- Candidate D2: **silent gate** (no prompt mention; rejection message
  self-explanatory on first firing) ± soft-gate (allow-through after 2
  rejections). Separates mechanism (1) rejection-repair burn from (2)
  anticipatory caution.

## 2026-08-26 — paper python-subset comparison & contamination check

- Paper Table 3 python column (29 instances): **mini-swe-agent best = 17.2%**.
  Our Arm A = 72% golden-valid (62% on an all-29 denominator) — ~4× the
  published field on the same scaffold.
- Contamination checks (python commits span 2025-02..2026-01, all plausibly
  inside Qwen3.8's Aug-2026 training window):
  (1) **No date gradient**: A resolves 72% on pre-2025-08 commits vs 72%
  after.
  (2) **Gold-patch overlap** (added-line recall in resolved A-r1 patches):
  mean/median 0.39. **Exception: transformers-38332 at 1.00 (verbatim gold)
  — treat as likely memorized; deepeval-08844c8 0.81 and hummingbot 0.77
  borderline.**
- Contamination contributes on isolated instances but cannot explain the
  bulk; the paper's own 2026 models had exposure to most of the same commits
  and scored 17%. Most parsimonious story: Qwen3.8-27B is a genuine jump on
  python refactoring, and ProMax's published numbers predate its release by
  4 days. Headline rates stay internally paired; they are not claimed as a
  4× field result.

## 2026-08-26 — Arm D2 (silent soft gate) built, tested, launched

- Prompt **byte-identical to arm A** — no gate mention. A BASH_ENV
  `echo`-function intercepts the submission marker and runs the delta-scoped
  gate first; on new errors it prints the rejection and suppresses the
  marker; **soft gate: after 2 rejections the next attempt passes silently**.
- Bug found in soft-limit counter during container test: `open(ATTEMPTS,"w")`
  truncates before `rejections()` reads in the same expression → counter
  stuck at 1. Fixed (count-then-write), retested: reject → reject → silent
  pass ✓.
- `s1-arm-d2-r1` (`worker-a`) and `s1-arm-d2-r2` (`worker-b`) launched
  concurrently.

## 2026-08-27 — D2 complete + 2-hour container wall

**D2 (silent soft gate): 16/25 both rounds → 32/50 (64%).** Paired: A>D2
6-vs-3 (p≈0.51, n.s.), D2>D 6-vs-3 (p≈0.51). Gate fired in 9/58 episodes;
**6/9 gated episodes resolved** (repair loop works); soft-limit reached only
twice.

**The confound:** mini-swe-agent containers run `sleep 2h`
(`environment.container_timeout`, default "2h"). Episodes exceeding 2 hours
lose their container mid-flight and burn remaining steps against a dead
environment → LimitsExceeded with an empty patch. Container-death count per
arm (2 rounds): **A 4, B 7, C2 10, D 13, D2 16** — monotone in episode
length; nearly all "step-cap" deaths are wall deaths. At ~15 tok/s local
decode the wall binds hard; at API speeds it mostly wouldn't.

Interpretation: (i) treat the wall as legitimate wall-clock budget and keep
results as-is; (ii) treat mid-episode death as artifact and re-run A vs D2
with `container_timeout: "8h"`. The "gate hurts" headline (A>D p≈0.016) is
partially artifact under (ii). Chose (ii) for the hosted pair below; did not
repeat locally (multi-day at 15 tok/s; hosted is already a wall-free
within-regime comparison).

## 2026-08-27 — hosted revalidation launched

- OpenRouter auth via the local mini-swe-agent config (not in this tree).
  Smoke: structured tool calls ✓ (CoreWeave, Parasail), reasoning separated
  ✓, usage reporting ✓.
- `reasoning.effort` passthrough could not be verified. Local server shows
  the same flat reasoning volume on trivial prompts across
  low/medium/xhigh. Hosted arms only ever compared with each other.
- Provider strategy: pinned `CoreWeave, Parasail` (fp8). Chutes excluded
  (reports 0 reasoning tokens — suspicious template).
- Pre-launch cost estimate from A-r1 trajectories: ~$0.60/episode, ~$75–120
  for the revalidation. **The estimate was wrong** — hosted defaulted to
  xhigh (passthrough failed), reasoning billed as output, actual spend an
  order of magnitude above the probe. Lesson: hosted agentic runs need a
  per-run cost-cap probe (2 episodes, read billed usage, *then* extrapolate)
  before fleet launch.
- Launched A and D2, 2 rounds, `container_timeout: 8h`, hosts split.

## 2026-08-28 — hosted revalidation complete

**Hosted (OpenRouter, 8h wall, CoreWeave/Parasail fp8), 2 rounds each:**

| Arm | Resolved | Notes |
|-----|----------|-------|
| A hosted | 47/50 (94%) | r1 96%, r2 92% |
| D2 hosted | 42/50 (84%) | r1 92%, r2 76% |

- Paired A↔D2: A-better 6, D2-better 1 (sign p≈0.125, n.s. — same direction
  as local).
- **Wall deaths: 0 in all four runs.**
- Gate fired in 17/58 D2 episodes; 12/17 gated episodes still resolved. Yet
  D2 trails A in every comparison ever run. On dynamically-typed repos with
  outcome-focused suites, new-type-error patches often pass tests anyway, so
  type-gating delays more than it saves for this model.
- **Serving-stack finding**: hosted A 94% vs local A 72%. Wall fix explains
  little (local A lost only ~4 slots). Suspects: local quality loss (DSpark
  spec decode / fp8 KV / sm121 kernels) and/or hosted default xhigh thinking.
- **Contamination signal stronger hosted**: resolved-patch gold-overlap mean
  0.62 (median 0.61, 18/47 >0.75) vs local 0.39. The 94% headline is not a
  field comparison.

## 2026-08-28 — review round & retroactive gate measurement

- Review actioned: TL;DR → bullets; wall warning under the results table;
  non-adoption framed with two named untested confounds (grep familiarity;
  trigger density); stats: Holm note, Wilson CIs, instance-level pairing.
  Transferable harness findings extracted to HARNESS-NOTES.md.
- Local 8h rerun declined — the hosted pair is already a wall-free A-vs-D2
  comparison.
- **Retroactive gate**: apply the delta-scoped gate to every resolved
  baseline-A patch (local + hosted, both rounds; 83 patches) in fresh
  containers. v1 filled `worker-a`'s disk (no image eviction); v2
  (instance-major, per-instance evict, resume) completed all 83.
  **Result: the gate would have rejected 29/83 (35%) of test-passing
  baseline patches**, stable across runs (29–39%); median 4 new type errors
  per false-blocked patch (max 43); 11/24 distinct instances affected.

## 2026-08-29 — review round 2: the 2×2 completed

- 35% was one cell. Gate over all 14 golden-valid *unresolved* baseline-A
  patches: **flag rate 2/14 (14%) vs 29/83 (35%) on resolved** (Fisher
  p≈0.21). Point estimate inverted — the gate flags passing patches more
  than failing ones. Failures are under-edits, and unwritten code is
  type-silent.
- Other round-2 fixes: gate threshold explicit (reject ≥1; cap-8 is
  display-only); CI caveat; local 72% quarantined alongside hosted 94%;
  contamination reframed to lead with the 0.39→0.62 shift.

No further arms. The untested confounds (grep-off, reference-model harness
control, path perturbation) are retracted, not queued.
