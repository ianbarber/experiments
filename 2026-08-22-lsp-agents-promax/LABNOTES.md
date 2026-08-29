# Lab notebook — RefactorBench LSP experiment

Chronological record: what was done, what was observed, what was decided, in order.
Protocol reference lives in NOTES.md; the plan in PLAN.md. Times are spark local (UTC-7);
container/server logs print UTC (+7h).

---

## 2026-08-22 (Sat) — planning & investigation

- **~19:10** Project start. Machine inventory (spark): NVIDIA GB10 DGX Spark, aarch64,
  20 ARM cores (10× Cortex-X925 + 10× A725), 119 GiB unified memory (~273 GB/s), CUDA 13.0,
  driver 580.159, Docker 29.2.1, disk 916 GB @ 88% full (111 GB free). DSv4 NOT running
  (it's `~/Projects/dsv4/ds4/ds4-server`, custom C/CUDA engine + 87 GB GGUF; nothing to stop).
- Research (3 parallel agents):
  - **Benchmark identified**: SWE-Bench ProMax, arXiv 2608.09802, COLM 2026, released
    2026-08-10. 170 instances / 70 repos / 7 languages (py 29, ts 28, java 26, go 23,
    c++ 22, rust 22, c 20). Paper baselines: mini-swe-agent — Qwen3.5-MoE 20.6%,
    Sonnet 4.6 30.6; OpenHands — GPT-5.2 41.2. Images `key4127/refactor-dockerhub:<id>`,
    **amd64-only**, ~317 GB compressed total. Paper PDF → `paper/`.
  - **Model verified**: Qwen3.8-27B is real (released 2026-08-13/14), dense hybrid
    48× Gated DeltaNet + 16× gated attention, 262k ctx, multimodal, MTP head, Apache 2.0.
    HF `model_type: qwen3_5` (same family as Qwen3.5). Official FP8 checkpoint ~29 GB.
    Default reasoning_effort=xhigh known to overthink → we use medium.
  - **LSP landscape**: Serena/solidlsp (40+ languages, pyrefly pluggable), Pyrefly 1.2.0
    GA. Key prior work arXiv 2608.13568: LSP tools usually hurt token budgets for
    localization, help for reference-completeness/rename, and only with content-enriched
    output; frontier models use LSP 0–6% unprompted. No published LSP ablation with a
    local open-weight model — experiment is novel.
- PLAN.md written; artifact: https://claude.ai/code/artifact/71117c82-8564-4341-946f-5aa471979eed
- **Lab topology** (from Ian): cortical (32T/122G/409G free, no docker, ⚠ running
  muse-glimmer llama-server + opencode — left untouched, sudo needs password),
  chunklebox (16T/29G/147G, docker), leejr (16T/125G/80G, docker), NAS
  `192.168.1.37:/Public` → `/mnt/nas` everywhere, 6.8 TB free. Plan revised: spark serves;
  x86 lab hosts run containers natively; NAS = image tar cache + run archive. QEMU/cloud
  path dropped.

## 2026-08-22 — Phase 0/1 execution ("lets go for it")

- **21:16** Lab probes confirm. chunklebox picked as primary worker (cortical busy+no
  docker). NAS cache flow validated: dspy image docker-save → 752 MB zst in 15 s (~50 MB/s).
- **21:27** `Qwen/Qwen3.8-27B-FP8` download → landed on NAS (HF_HOME points there), 6 min.
- Harness cloned (`harness/`); dataset: 170 instances parse; fields incl. image_name ✓.
- **Gold gate #1** (dspy-9193): golden patch FAILS — `test_enable_net_flag`,
  CodeInterpreterError, model==golden failure → environment, not patch. → env-flaky list.
- **Gold gate #2** (albumentations-2337): SUCCESS/SUCCESS in 42 s. **Eval pipeline valid.**
- mini-swe-agent 2.4.6 on chunklebox; `mini-extra swebench --subset
  swe-bench-promax/SWE-Bench-ProMax --split test` loads directly, honors image_name —
  planned adapter unnecessary.
- **Serving saga on spark (aarch64/sm121)** — chronological:
  1. vLLM nightly pip: **no aarch64 wheels** — dead.
  2. SGLang 0.5.9 pip: installs; torch resolved to CPU wheel → reinstalled cu130 →
     sgl-kernel linked against libnvrtc.so.**12** → reinstalled cu129 → imports OK.
  3. Launch #1: cuDNN guard (torch 2.9.1 Conv3d bug) → pin nvidia-cudnn-cu12==9.16.0.29.
  4. Launch #2: hybrid-GDN models on Blackwell demand explicit attention backend.
  5. Launch #3 (--attention-backend triton): triton's bundled ptxas predates **sm_121a** →
     TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas.
  6. Launch #4: CUDA graph capture: GemmaRMSNorm "no kernel image" — sgl-kernel's sm100
     cubins genuinely can't run on sm_121. pip route DEAD.
  7. Web search → **MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark** (image
     `lmsysorg/sglang:qwen38-27b`, GB10-tuned, measured numbers). Vendored at
     `serving/spark-sglang/`, `.env`: QUANT=fp8, MAX_CONCURRENT_REQUESTS=8.
     `./start-dspark.sh` (DSpark spec decode, flashinfer, fp8 KV, CPU pinning to X925).
  - Side lessons: rsync of HF snapshots needs `-aL` (symlinks dangle); disk hit 98% →
    deleted my own duplicate weight copies + dead venvs (41 GB back). GGUF Q8_0 + MTP
    draft downloaded to NAS as fallback.
- **~21:55** Server READY. Sanity: correct code + arithmetic, reasoning_content separated,
  **31.7 tok/s** single-stream short-context code; effort control via chat_template_kwargs ✓.
- **22:20** First smoke attempt: mini-swe-agent RuntimeError → cost tracking on unknown
  model → `MSWEA_COST_TRACKING=ignore_errors`.
- **22:24 → 23:42 Smoke episode** (albumentations-2337, arm A config): Submitted, 80
  steps, 78 min, 26 KB patch → **RESOLVED by official harness** (golden also success).
  Full pipeline proven; near-0% baseline risk eliminated.

## 2026-08-23 (Sun, early) — dev-10 shakeout

- dev-10 (1/language + extras, arm A, 3 workers, chunklebox): rollout ~2.2 h, all
  episodes ran. 8/10 Submitted (patches 2–71 KB), 2 LimitsExceeded @300 steps with empty
  patches (2337 — which the smoke run had solved: temp-1.0 variance; ruff-21445).
- Eval: **3/9 golden-valid resolved (33%)** — maven ✓ (Java), cli ✓ (Go), s2n-tls ✓ (C);
  0 python/ts/rust/c++. transformers-38788 golden-invalid (#2 on flaky list).
  Reference: paper mini-swe-agent frontier ~21–31% overall.
- LSP tool built (subagent): `lsp-tool/` CLI+daemon over solidlsp (serena-agent 1.7.0,
  pyrefly 1.2.0 / basedpyright 1.39.10), content-enriched output, 26/26 e2e on both
  servers, warm refs 52–81 ms.

## 2026-08-23 — stage 1 launches & infrastructure fixes

- **s1-arm-a-r1 launch #1 FAILED — disk**: flat pre-fetch of 29 python images blew
  chunklebox's 147 GB (verl image bundles CUDA/nvshmem). Evicted images (all NAS-cached),
  wrote **`agent/run_batch_waved.sh`**: per-wave load → rollout → eval → evict; eval
  always grades in ORIGINAL images. Relaunched (waves of 6, 3 workers).
- **LSP image layers** (leejr): pilot failed (no py≥3.11, no uv in image) → uv bootstrap
  in Dockerfile → pilot OK (829 MB zst; in-container refs 2.6 s cold). Batch of 29:
  20 OK + failures in 3 classes:
  1. **Baked-in ByteDance proxy** (`sys-proxy-rd-relay.byted.org`) in several images kills
     all network → `env -u` proxy neutralization for build steps only (final env
     unchanged). Also the likely ROOT CAUSE of the golden-invalid instances.
  2. Failed `python -m venv` leaves partial dir uv won't reuse → rm -rf before fallback.
  3. Verify picked first tracked .py = empty `__init__.py` → pick largest file instead.
  Sweep of 9 failures after fixes: **9/9 OK → 29/29 LSP images on NAS.**
- **s1-arm-b-r1** (arm B, LSP available, leejr, lsp waves of 4) launched — runs
  CONCURRENTLY with arm A against the same server (removes serving drift; host effect to
  be counterbalanced in round 2 by swapping hosts).
- Server under load: 6 concurrent episodes, ~14–15 tok/s per stream at 40k+ ctx.

## 2026-08-23 — stage 1 round 1 results (running)

- **Arm A (baseline) COMPLETE: 19/25 resolved (76%) on golden-valid python** (4
  golden-invalid: transformers-38788, dspy-9193, dspy-1801, langextract-239 — all
  consistent with the proxy issue). By wave: 4/4, 3/5, 4/6, 5/5, +w5. Audited
  deepeval-2345 pass: genuine (20 KB multi-file patch, suite green, exit 0).
  Fails: lerobot-2808, django-19643, langchain-32996, gallery-dl-7872, +w5 TBD.
- **Arm B (LSP available): ZERO `lsp` calls** through 16+ episodes (~1500+ commands);
  tool docs verified present in prompts. Replicates 2608.13568 unprompted-usage finding
  on an open 27B. Resolve through wave 4: 8/12 (67%). Note transformers-38332
  golden-valid on chunklebox but golden-invalid on leejr — golden validity varies by
  host; paired analysis must intersect golden-valid sets.
  (Correction logged: an early "daemon running in arm B" observation was a self-matching
  grep artifact.)
- **Arm C (LSP preferred) launched** on chunklebox after arm A. Live containers show
  pyrefly/basedpyright server processes — the preference instruction DOES induce usage;
  first completed episode (transformers-38788, 56 cmds) used 0 — adoption is
  per-episode heterogeneous. Quantify at completion.

## 2026-08-23 — Arm C pilot result & protocol revision

- **Arm C wave-1 pilot (5 episodes, ~800 commands): ZERO `lsp` calls** — preference
  paragraph verified present in prompts. The polite "prefer the lsp tool" clause is
  behaviorally inert on this model; C-as-written would have replicated A.
- (Second live-check correction: the "pyrefly processes running in C containers"
  observation was ANOTHER self-matching pipeline artifact — grep's own cmdline contained
  the pattern. No LSP daemon has run in any arm's containers. Lesson recorded: process
  checks must use patterns that cannot appear in the checking command, e.g. `[p]yrefly`.)
- **Decision**: stop s1-arm-c-r1 after wave 1 (archived as `s1-arm-c-r1-pilot`; 2 of its
  5 episodes hit the step cap), and strengthen the manipulation → `agent/arm_c2.yaml`:
  imperative, workflow-integrated instruction (lsp-first analysis step in Recommended
  Workflow; `<IMPORTANT>` block: lsp as PRIMARY navigation, `lsp refs` REQUIRED before
  changing any symbol and again before patch creation, `lsp diag` on edited files).
  Rationale: still tests Ian's arm-3 intent ("prompt instructing to prefer the LSP"),
  but strong enough to produce a manipulation at all; the pilot documents that weak
  phrasing does nothing. Relaunched as **s1-arm-c2-r1** (chunklebox, lsp waves of 5).
- Interpretation note for the writeup: for a 27B agent, *availability* (B) and *polite
  preference* (C-pilot) both yield 0% adoption — instruction strength is a first-order
  variable, consistent with (and stronger than) the frontier-model finding in 2608.13568.

## 2026-08-23 — Arm B round 1 complete; paired A↔B

- **Arm B FINAL: 18/24 (75%) golden-valid; 0 `lsp` calls across all 29 episodes.**
- **Paired A↔B on the 24 co-golden-valid instances: A 18/24, B 18/24 — exact tie.**
  Discordant: only-A = {albumentations-2337, albumentations-2495}; only-B =
  {lerobot-2808, verl-3915}; McNemar p=1.0. Clean negative control; the +1.3 kB prompt
  addition also costs nothing.
- Never-solved by either arm: django-19643, langchain-32996, gallery-dl-7872, dspy-9047 —
  prime targets for judging C2's effect.
- **First C2 episode** (transformers-38788): 2 lsp calls (`diag` ×2 of 80 cmds) —
  imperative prompt produces nonzero adoption; used for verification, not navigation.
- Round 2 counterbalance started: **s1-arm-a-r2 on leejr** (A ran on chunklebox in r1).

## 2026-08-24 — round 1 complete (all three arms)

**Resolve rates, golden-valid python (n≈25):**
| Arm | Resolved | Rate |
|-----|----------|------|
| A baseline | 19/25 | 76% |
| B lsp-available | 18/24 | 75% |
| C2 lsp-imperative | 16/25 | 64% |

**Paired (exact McNemar):**
- A↔B: 18=18, discordant 2 vs 2 (p=1.0) — perfect negative control.
- A↔C2: only-A {albumentations-2337, albumentations-2495, adk-19315fe, transformers-38332},
  only-C2 {lerobot-2808} — 4 vs 1, p≈0.375.
- B↔C2: 2 vs 0, p≈0.5.
No significant differences at this n; point estimates lean AGAINST the imperative-LSP arm.

**Usage (the cleanest result):** across 63 episodes with the tool present
(B 29 + C2 29 + C-pilot 5): **navigation subcommands (refs/def/sym/outline/hover/rename)
were used exactly 0 times**. C2's 21 total calls (16/29 episodes) were 100% `lsp diag` —
the model partially complies with the concrete "run diag on edited files" clause but never
substitutes lsp for grep/read navigation, no matter the instruction strength. C2 mean 90
steps/episode (max 300).

**Interim interpretation:** for Qwen3.8-27B on ProMax python, (1) LSP availability is
behaviorally inert; (2) imperative prompting buys only verification-style diag calls and
a (non-significant) resolve-rate *decrease* — consistent with 2608.13568's "conditional
and usually negative", with the twist that the model refuses navigation usage entirely, so
the "LSP value" question collapses into an instruction-following question at this scale.
(3) Baseline observation of independent interest: 76% on ProMax python vs paper frontier
mini-swe-agent numbers (Sonnet 4.6 ~30.6% overall) — Qwen3.8-27B is exceptionally strong
on python refactoring; per-language comparison to the paper's python column pending.
Round 2 (A on leejr running; B-r2 on chunklebox launched; C2-r2 queued) doubles n.

## 2026-08-24 — localization/thrash analysis (analysis/localization.py, round-1 trajectories)

Question (Ian): is the model going to the WRONG FILE much — would steering toward
navigation tools help? **Answer: no wrong-file thrash exists.**

| Split | steps | 1st gold-file touch (step) | searches pre-gold | edit precision | edit recall | gold files missed |
|-------|------|---------------------------|-------------------|----------------|-------------|-------------------|
| A resolved (19) | 47 | 1.5 | 1.3 | 0.97 | 0.76 | 1.9 |
| A unresolved (6) | 82 | 0.7 | 0.8 | 1.00 | 0.32 | 6.2 |
| B resolved (18) | 33 | 0.8 | 1.0 | 0.94 | 0.75 | 2.7 |
| B unresolved (6) | 104 | 1.8 | 2.2 | 0.97 | 0.43 | 3.0 |
| C2 resolved (16) | 43 | 1.1 | 1.2 | 0.94 | 0.74 | 2.4 |
| C2 unresolved (9) | 131 | 2.1 | 1.7 | 1.00 | 0.35 | 4.9 |

- The agent touches a gold file within ~1–2 steps in EVERY episode (0 "never" cases,
  ~1 search before first gold touch) — ProMax problem statements localize the entry file.
- Edit precision ≈ 0.94–1.00 across all splits: it essentially never edits a wrong file.
- **The failure mode is RECALL**: unresolved episodes edit the right files but only ~1/3
  of the needed set (missing 5–6, up to 13, gold files). This is precisely the paper's
  "incomplete refactoring" failure — and precisely what `lsp refs` addresses, and
  precisely the subcommand the model refuses to call in every arm.
- Unresolved episodes are 2–3× longer with more searching — churn is downstream
  (test-fixing), not initial file-finding.
- **Implication**: retraining/steering toward *navigation* attacks a non-problem.
  The mechanical hook is reference-coverage feedback: end-of-turn injection — after
  edits, auto-run `lsp refs` on changed symbols and report unmodified referencing files
  — converging with Ian's lsps-for-llms REPORT.md finding that end-of-turn type-check
  injection helps where forced integration is tricky. → Arm D design.

## 2026-08-24 — Ian's lsps-for-llms report digested; Arm D built & bench-tested

- Digest of github.com/ianbarber/lsps-for-llms REPORT.md (subagent, full repo read):
  submission-time **blocking gate** blocked 11/12 bad completions (vs 2/12 one-shot,
  ~zero cost on clean work); per-turn end-of-turn injection +0.06 n.s.; the ONLY
  FDR-significant delivery effect was the imperative "treat as squigglies, fix now"
  sentence at **−0.131** — independently explains our C2 trend. Their mini-swe-agent
  case study reproduces our zero-election exactly (Sonnet: 0 codenav calls; defn calls
  additive, not substitutive). Also: refs/impls ≈ grep for override enumeration; LSP's
  unique win is single-target goto; "navigation never beats a readable type".
- **Arm D built** (`agent/arm_d.yaml` + `lsp-tool/gate/`): arm-A prompt with ONLY the
  Submission step changed to a `submit` wrapper; wrapper runs a **delta-scoped Pyrefly
  acceptance gate** (baseline recorded at episode start via env_startup_command base64
  payload into the promax-lsp images; only NEW fingerprints count; syntax-cascade
  demotion; cap 8; directive text at gate time only; gate never blocks on checker infra
  failure). Matches the recall failure mode: stale references left by incomplete
  renames surface as new diagnostics in unmodified files.
- Bench test in promax-lsp:albumentations-2337: baseline 64 KB of pre-existing
  fingerprints (delta scoping essential); clean submit → marker passes; injected
  `bad-assignment` → precise rejection, rc=1. Ready to run; queued behind round 2.

## 2026-08-25 — rounds 1+2 complete for A/B/C2; combined analysis

**Two-round totals, golden-valid python (each instance ×2 rounds, hosts counterbalanced):**
| Arm | Resolved | Rate |
|-----|----------|------|
| A baseline | 36/50 | 72% |
| B lsp-available | 35/49 | 71% |
| C2 lsp-imperative | 34/50 | 68% |
Per-instance 2-round sign tests: A↔C2 4-vs-2 (p≈0.69), A↔B 3-vs-4 (p=1.0), B↔C2 3-vs-2
(p=1.0). **No effect of LSP provision or instruction on resolve rate, ±~4 pts.**
- C2-r1 vs r2: 64% → 72% — the r1 deficit was mostly noise, as suspected.
- Usage across ALL tool-equipped episodes (~120): navigation subcommands used **once**
  (a single `lsp refs` in C2-r2); everything else `lsp diag` under the imperative prompt
  (C2 totals: 61 calls / 39+21 diag). B: 0 calls in 58 episodes across both rounds.
- Instances stably discordant: transformers-38332 (A 2/2, C2 0/2), lerobot-2808
  (A 0/2, C2 2/2, B 1/2) — no pattern implicating the intervention.
- **Arm D (acceptance gate) r1 running on chunklebox** (wave 2; wave-1 gate stats:
  2 rejections in 1/5 episodes; the rejected episode failed to converge and hit the
  step cap — rejection-loop budget burn is the failure mode to watch). **D-r2 launched
  on leejr** for host counterbalance.

## 2026-08-25 — Arm D round 1 complete

- **D-r1: 13/25 (52%)** — 20 pts below A-r1 (76%) on the same host. Gate telemetry:
  5/29 episodes drew ≥1 rejection (mean 198 steps, 3/5 hit the step cap, 40% resolved);
  but even never-rejected episodes ran 55% (11/20) with 5 step-caps — total 8
  LimitsExceeded vs A-r1's 2.
- Capped-episode anatomy: 5 of 8 never created patch.txt or ran `submit` at all (pure
  grinding, a pattern that exists in all arms but is elevated here); 3 were
  rejected-and-never-recovered.
- Asymmetry worth noting for any D+ redesign: the gate converts "submit a type-broken
  patch that might still pass tests" into "certain failure (empty patch)" whenever repair
  doesn't converge within budget. A soft gate (allow-through with warning after N
  rejections) would bound that cost. Also candidate explanation for the clean-episode
  deficit: the gate note in the submission instructions may induce over-cautious
  polish-instead-of-submit behavior (anticipatory effect, cousin of the C2 trend and of
  Ian's −0.131 imperative-sentence result).
- D-r2 running on leejr (counterbalance) — judgment reserved until it lands.

## 2026-08-26 — STAGE 1 COMPLETE: final four-arm results

**Combined two rounds, golden-valid python, hosts counterbalanced:**
| Arm | Resolved | Rate | Paired vs A (sign test) |
|-----|----------|------|--------------------------|
| A baseline | 36/50 | 72% | — |
| B lsp-available | 35/49 | 71% | 3-vs-4, p=1.0 |
| C2 lsp-imperative | 34/50 | 68% | 4-vs-2, p≈0.69 |
| **D acceptance gate** | **29/50** | **58%** | **7-vs-0, p≈0.016** |

- **The only significant effect in the study is the gate hurting** (A>D on 7 paired
  instances, D>A on 0; B>D 8-vs-1, p≈0.039). All prompting interventions are nulls.
- D combined telemetry: rejections in 11/58 episodes; 14 step-caps (A: ~2/run). Failure
  decomposition (r1): rejected episodes 40% resolve w/ mean 198 steps; never-rejected
  episodes ALSO depressed (55%) with elevated no-submit-attempt caps → two mechanisms:
  (1) rejection-repair non-convergence converts maybe-pass patches into certain
  empty-patch failures; (2) anticipatory caution from the gate note in the prompt.
- Regime note (honest read vs Ian's lsps-for-llms gate result): his gate won on 60-step
  revision tasks, temp 0, seeded single defects; ours is 300-step full refactor episodes
  at temp 1.0 on 11-file gold patches. The gate's value inverts across regimes.
- Candidate D2 if we continue: **silent gate** (no prompt mention — standard submission
  intercepted; rejection message self-explanatory on first firing) ± soft-gate
  (allow-through with warning after 2 rejections). Cleanly separates mechanism (1) from (2).

## 2026-08-26 — paper python-subset comparison & contamination check

- Paper Table 3 per-language python column (29 instances): **mini-swe-agent best = 17.2%**
  (GPT-5.2/Gemini/Kimi/Qwen3.5 tied; Sonnet 4.6 & GLM-5 13.8); OpenHands best = GPT-5.2
  48.3, Qwen3.5 37.9. Our Arm A = 72% golden-valid (62% on an all-29 denominator) — ~4×
  the published field on the same scaffold, above even OpenHands-best.
- Contamination checks (python commits span 2025-02..2026-01, all plausibly inside
  Qwen3.8's Aug-2026 training window):
  (1) **No date gradient**: A resolves 72% on pre-2025-08 commits vs 72% after.
  (2) **Gold-patch overlap** (added-line recall in resolved A-r1 patches): mean/median
  0.39 — most successes are materially different solutions that still pass the suite
  (several at 0.0–0.1). **Exception: transformers-38332 at 1.00 (verbatim gold
  reproduction) — treat as likely memorized; deepeval-08844c8 0.81 and hummingbot 0.77
  borderline.** Writeup should include a sensitivity row excluding overlap>0.75
  instances.
- Read: contamination contributes on isolated instances but cannot explain the bulk;
  the paper's own 2026 models had exposure to most of the same commits and scored 17%.
  Most parsimonious story: Qwen3.8-27B is a genuine generational jump on python
  refactoring (consistent with its vendor SWE-bench-Pro 61.7), and ProMax's published
  numbers predate its release by 4 days.
- Paper §5.2 independently matches our trace analysis: dominant failure = incomplete
  refactoring, explicitly "not a matter of failing to find the right files".

## 2026-08-26 — Arm D2 (silent soft gate) built, tested, launched

- Design (Ian approved): prompt **byte-identical to arm A** — no gate mention. A BASH_ENV
  `echo`-function in /root/.bashrc intercepts the exact submission marker and runs the
  delta-scoped gate first; on new errors it prints the self-explanatory rejection and
  suppresses the marker (exit 1); **soft gate: after 2 rejections the next attempt passes
  silently**. Separates the two D-harm mechanisms: anticipatory caution (prompt-induced;
  absent by construction here) vs rejection-repair burn (bounded at 2).
- Bug found in soft-limit counter during container test: `open(ATTEMPTS,"w")` truncates
  before `rejections()` reads in the same expression → counter stuck at 1. Fixed
  (count-then-write), re-embedded, retested: reject → reject → silent pass ✓. Plain-echo
  passthrough and clean-submit paths verified ✓.
- `s1-arm-d2-r1` (chunklebox) and `s1-arm-d2-r2` (leejr) launched concurrently.
- Prediction to check: if D's deficit was anticipatory, D2 ≈ A with a few gated saves;
  if it was repair-burn, D2 lands between D and A with step-caps replaced by soft passes.

## 2026-08-27 — D2 complete + MAJOR CONFOUND DISCOVERED (2-hour container wall)

**D2 (silent soft gate): 16/25 both rounds → 32/50 (64%).** Paired: A>D2 6-vs-3 (p≈0.51,
n.s.), D2>D 6-vs-3 (p≈0.51). Gate telemetry (2 rounds): fired in 9/58 episodes; **6/9
gated episodes resolved** (repair loop works: model reads rejection, fixes, resubmits —
vs hard-gate D's 2/5); soft-limit reached only twice. One rejection was a genuinely
broken patch (43 new type errors, cv2 enum misuse).

**The confound:** while investigating a 188-submission-attempt episode, found that
mini-swe-agent containers run `sleep 2h` (`environment.container_timeout`, default "2h").
Episodes exceeding 2 hours LOSE THEIR CONTAINER mid-flight ("No such container" on every
later command) and burn remaining steps against a dead environment → LimitsExceeded with
an empty patch. Container-death count per arm (2 rounds): **A 4, B 7, C2 10, D 13, D2 16**
— a monotone gradient tracking episode length; nearly ALL "step-cap" deaths are actually
wall deaths. At our local decode speeds (~15 tok/s/stream) the wall binds hard; at the
paper's API speeds it mostly wouldn't.
- Conditional on surviving the wall: A ≈78%, B ≈83%, C2 ≈85%, D ≈78%, **D2 ≈94%**
  (biased — long/hard episodes die more — but directionally: D2 survivors do very well;
  gate-as-free-insurance may be real once the artifact is removed).
- Interpretation fork: (i) treat the wall as legitimate wall-clock budget (gate arms pay a
  real time cost) and keep results as-is, documented; (ii) treat mid-episode death as
  artifact (the pathological dead-container grinding is not "budget") and re-run the key
  comparison (A vs D2, 2 rounds each) with `environment.container_timeout: "8h"` (~2 days
  compute). The "gate hurts" headline (A>D p≈0.016) is partially artifact under (ii).

## 2026-08-27 — hosted revalidation launched (OpenRouter)

- Key verified on spark (`~/.config/mini-swe-agent/.env`, copied to both workers' mini
  config). Smoke tests: auth ✓, structured tool calls ✓ (AkashML, CoreWeave, Parasail),
  reasoning separated ✓, usage/cost reporting ✓ (~$0.0003/call).
- `chat_template_kwargs`/`reasoning.effort` passthrough could NOT be verified — but the
  discriminating test was invalid: our LOCAL server shows the same flat reasoning volume
  on trivial prompts across low/medium/xhigh (147/449/273 chars). Effort control is
  low-stakes; kept in extra_body; hosted arms only ever compared with each other.
- Provider strategy: pinned `CoreWeave, Parasail` (fp8 = matches local quant; both emit
  normal reasoning accounting). Chutes excluded (reports 0 reasoning tokens — suspicious
  template). 12 providers total; input $0.29–0.50/M, output $2.40–3.40/M.
- Cost basis (measured from A-r1 trajectories): mean 1.69M input + ~40k output tokens per
  episode → ~$0.60/episode; revalidation ≈ $75–120 total.
- **Launched**: `s1h-arm-a-r1`→`s1h-arm-d2-r1` chained on chunklebox (6/5 workers),
  `s1h-arm-a-r2`→`s1h-arm-d2-r2` on leejr (4 workers). Both configs:
  `container_timeout: 8h` (the fix for the 2h-wall confound), same prompts/gate as the
  local A/D2 arms. Hosted results form a self-contained comparison; never mixed with
  local numbers.

## 2026-08-27/28 — hosted revalidation interim

- **s1h-arm-a-r1 (hosted, 8h wall): 24/25 (96%)** vs local A 76%/68%. Δ far exceeds the
  wall fix (local A-r1 had just 1 wall death). Candidate causes, to adjudicate when all
  four runs land: (a) local serving quality loss (DSpark spec-decode acceptance, fp8 KV,
  sm121 triton kernels), (b) hosted runs at default reasoning effort (likely xhigh) vs
  local pinned medium, (c) provider BF16/fp8 standard stacks. Follow-ups if the gap
  holds: patch-gold overlap on hosted outputs (contamination re-check), wall-death count
  (expect 0), and possibly a local run at xhigh to separate (a) from (b).

## 2026-08-28 — hosted revalidation COMPLETE; final results + cost incident

**Hosted (OpenRouter, 8h wall, CoreWeave/Parasail fp8), 2 rounds each:**
| Arm | Resolved | Notes |
|-----|----------|-------|
| A hosted | 47/50 (94%) | r1 96%, r2 92% |
| D2 hosted | 42/50 (84%) | r1 92%, r2 76% |
- Paired A↔D2: A-better 6, D2-better 1 (sign p≈0.125, n.s. — but same direction as local).
- **Wall deaths: 0 in all four runs** — artifact fully eliminated.
- Gate fired in 17/58 D2 episodes; 12/17 gated episodes still resolved (repair loop
  works). Yet D2 trails A in every comparison ever run (local hard, local silent, hosted
  silent). Emerging conclusion: on dynamically-typed repos with outcome-focused suites,
  new-type-error patches often pass tests anyway, so type-gating delays/derails more
  than it saves for this model.
- **Serving-stack finding**: hosted A 94% vs local A 72% (same instances, same scaffold).
  Wall fix explains little (local A lost only ~4 slots). Suspects: local quality loss
  (DSpark spec decode / fp8 KV / sm121 kernels) and/or hosted default xhigh thinking.
- **Contamination signal STRONGER hosted**: resolved-patch gold-overlap mean 0.62
  (median 0.61, 18/47 >0.75) vs local 0.39 — deeper thinking may partially be
  *recalling* these real 2025–26 commits. The 94% headline needs this caveat prominently.

**COST INCIDENT**: OpenRouter account shows $1,733 of $1,750 lifetime credits used
(~$16 left). If most of that is ours, actual run cost was ~15–20× the $75–120 estimate.
Probable mechanism: `reasoning_effort` passthrough does NOT work (as suspected), so all
hosted episodes ran at default **xhigh** — reasoning tokens are billed as output
($2.40–3.40/M) and xhigh plausibly emits ~10–20k reasoning tokens/step × ~60 steps ≈
0.5–1M output tokens/episode ≈ $2–4/episode output + input ≈ $5–7/episode × 232 episodes.
My "passthrough is low-stakes" judgment was wrong: it was cost-stakes. Lesson: hosted
agentic runs need a per-run cost cap probe (run 2 episodes, read actual billed usage via
/api/v1/generation, THEN extrapolate) before fleet launch. Ian to confirm how much of
total_usage predates this project (activity dashboard).

## Open items

- Round 2 (hosts swapped: A→leejr, B→chunklebox, C→leejr) after round 1.
- Analysis: paired matrix on intersection of golden-valid sets, McNemar A↔B, B↔C
  (order-interleaving already partial via concurrency); lsp-usage vs outcome within C.
- cortical still out of the pool (needs sudo for docker; busy with muse-glimmer).
- Report ProMax issues upstream when done: proxy-baked images, golden flakiness by host.

## 2026-08-28 — review round & retroactive gate measurement

- External review received (via Ian). Actioned: TL;DR → bullets; wall warning moved under
  the §3 results table; §4 reframed to non-adoption with two named untested confounds
  (grep familiarity; trigger density); §5 split into "entry-point localization =
  non-problem" vs "refs-for-completeness = untested because unadopted"; §6 decomposed
  honestly ("didn't help, probably hurt" — one clean, non-significant comparison) with C2
  elevated as a third gate-free type-feedback data point; §7 retitled and reframed as an
  anomaly requiring validation, contamination language hardened, weak rebuttals
  acknowledged as weak, perturbation test named as the decisive follow-up; stats: Holm
  note, Wilson CIs, instance-level pairing made explicit, denominators defined.
  Transferable harness findings extracted to HARNESS-NOTES.md.
- Reviewer's local-8h-rerun request: declined (Ian's call) — the hosted pair is already a
  wall-free within-regime A-vs-D2 comparison; a local repeat would only probe
  regime-generalization of an already-consistent direction. Now argued explicitly in §6.
- **Retroactive gate false-block analysis** (Ian requested): apply the delta-scoped gate
  to every resolved baseline-A patch (local + hosted, both rounds; 83 patches) in fresh
  containers. Early rows already show test-passing patches with 8–18 new type errors.
  Results → REPORT.md §6 and `results/retro_gate.jsonl`.
- Retro-gate v1 filled chunklebox's disk (no image eviction — the waved runner's lesson,
  re-learned); v2 (instance-major, per-instance evict, resume) completed all 83.
  **Result: the gate would have rejected 29/83 (35%) of test-passing baseline patches**,
  stable across runs (29–39%); median 4 new type errors per false-blocked patch (max 43);
  11/24 distinct instances affected. Type cleanliness ⊥ test success, now quantified.
