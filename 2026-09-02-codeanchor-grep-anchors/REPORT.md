# Do CodeAnchor-style anchors on grep output help a local coding agent refactor? (Arm E)

**Dates:** 2026-09-02 → 2026-09-06 · **Benchmark:** SWE-Bench ProMax python subset (29
instances, 25 golden-valid on our LAN) · **Scaffold:** mini-swe-agent 2.4.6 · **Model:**
Qwen3.8-27B-FP8, local SGLang/DSpark on the GB10 · **Design:** arm E (anchors) vs arm A8
(stock baseline), 2 rounds each, hosts swapped between rounds, 8 h container wall.

## TL;DR

- **Null on outcome, and not cheaper.** Arm E resolved 32/50 (64%) vs the control's 34/50
  (68%); paired on 25 instances: E better on 2, control better on 3, 20 tied (sign
  p = 1.0). Raw totals show E using 13% fewer input tokens and 5% less wall-clock, but
  that is two outlier control episodes: paired per instance the ratios are 0.95 (tokens,
  p = 0.76) and 1.03 (wall, p = 0.40). The only consistent trend is ~7% fewer steps
  (p = 0.07), cancelled by ~8% higher per-step latency from the longer context. Edit recall
  against gold files is identical (0.667 vs 0.666). See §3.1.
- **Variance win: plausible, unproven.** Round-to-round dispersion on the full set is equal
  (§3.2). A 7-episode rerun on the five largest-gap instances (§3.4) shows the control
  producing the runaway episodes (three 300-step-cap hits vs none; token p90 17.7M vs 9.3M)
  at equal typical cost, but the pooled spread test is flat (p = 0.71; bootstrap CI on the
  spread ratio 0.42–1.57) and the wall-clock tail is not shorter on the local server.
- **E3 (definition-site tags on file views, no caps — the paper's own placement) changes
  nothing on outcome and costs more:** 32/50 again (paired 1-vs-4, p = 0.375), steps −4.5%
  (p = 0.043) but wall-clock +35% (p = 0.015) and tokens +36% at the median; of the 64 gold
  files both arms miss, only 9 were ever named even by uncapped definition-site anchors (§3.3).
- **The mechanism works and is cheap.** Half of the agent's grep commands received an
  addendum (8.4 per episode, ≈9.4k chars ≈ 4% of cumulative prompt tokens, 5.6 s of
  language-server work per episode, zero failures in 1,001 grep events). The agent acts
  on it: of 99 flagged files that were gold-patch files, it opened 99 and patched 92.
- **It does not reach the failure.** The two arms cover the same gold files (both 123,
  E-only 3, control-only 2) and miss the same 64. Of the 69 round-1 misses: 41% are
  non-Python files a Python reference graph cannot reach, 39% are Python files never
  referenced by any symbol the agent chose to search, 4% were hidden by my display cap
  (measured by an uncapped replay), 6% were shown and ignored. Passive injection is
  bounded by the questions the agent asks.
- **Deviation from the paper, measured.** CodeAnchor caps nothing; I capped the addendum
  (4 symbols, 6 "not shown" files, 2,800 chars). An uncapped replay of every anchored grep
  shows the cap changed what the agent saw substantially (3.6× more files would have been
  named) but explains ≤4% of the coverage gap and at most 1 of the 3 gold files the
  control patched and E did not. The cap is documented as a protocol deviation, not the
  cause of the null.

## 1. Question and prior

The stage-1 study ([2026-08-22-lsp-agents-promax](../2026-08-22-lsp-agents-promax/REPORT.md)) found the model never adopts
`lsp refs` at any prompt strength, while its dominant failure is incomplete refactoring
(right files found, too few of them edited). CodeAnchor (arXiv 2606.26979) reports that
*passively* injecting static "used by" facts as source comments — prompt unchanged —
improves localization (+1.2 to +2.2 pp Func@5, −1.5 rounds, +9–10% input tokens) on
SWE-bench Lite/Verified with a grep-first agent, motivated by the same observation that an
optional call-graph tool goes unused. Arm E tests that idea here with two changes forced
by our setting: facts come from the warm language server (Pyrefly via solidlsp) rather
than an offline PyCG pass, and they are appended to grep *observations* rather than
written into source files, because our patches are `git diff`s and injected comments would
contaminate them.

## 2. Mechanism

Prompt byte-identical to arm A. A mini-swe-agent environment subclass
(`code/agent/anchor_env.py`, selected through the launcher shim `code/agent/anchor_runner.py`)
post-processes every observation whose command invoked a grep-family tool and whose output
contains hits in `.py` files. Hits are attributed to files three ways: `path:line:` prefixes;
`LINE:text` output of a single-file grep (grep omits the filename, ~half of the model's greps);
and, for chains such as `grep A f1; grep B f2` or `head f | grep -n`, text-matching against
every `.py` path named in the command. The hits and the pattern's identifier tokens go to a
new batch op in the daemon (`code/lsp-tool/src/lsp_tool/anchor.py`, CLI `lsp anchor`), which resolves each hit
to its definition, groups hits by symbol (disambiguating same-named symbols), and for each
symbol renders:

```
--- code anchors (language server): 2 symbols ---
apply_to_bboxes  method DualTransform.apply_to_bboxes  defined albumentations/core/transforms_interface.py:508  (1 hit above)
  used by 28 sites in 8 files:
    albumentations/augmentations/geometric/transforms.py: BaseDistortion.apply_to_bboxes [override], Perspective.apply_to_bboxes [override], +8 more
    ...
  referenced but NOT in this grep output: albumentations/augmentations/crops/transforms.py (6), ... +1 more
defined but never referenced elsewhere: Lambda.apply_to_bboxes (albumentations/augmentations/transforms.py:3222), ...
```

Source files sort before tests; uses are tagged `[import]`, `[subclass]`, `[override]`. Caps:
40 hits, 8 candidate symbols looked up, 4 rendered, 5 files per symbol, 6 "not shown" files,
2,800 chars, 12 s budget; the addendum is trimmed or skipped so an observation never crosses
the harness's 10,000-char elision threshold (which would otherwise hide grep output the
control would see). Images are the stage-1 `promax-lsp:<id>` variants; the updated package is
copied in at container start. Every grep event is logged (`anchor_log.jsonl`).

Arm E was launched four times before any counted episode: an image-injection bug (the runner
only injects the instance image for the literal environment name `docker`), the elision guard,
single-file attribution, and candidate-file attribution. The partial outputs
(`s1-arm-e-r1-aborted-v{1,2,3}`) are not results. A8 is arm A re-run in the same period with
the 8 h wall, so the comparison shares serving stack, hosts and dates.

## 3. Results

| Arm | Round 1 | Round 2 | Total | Rate |
|---|---|---|---|---|
| E anchors | 16/25 | 16/25 | 32/50 | 64% |
| A8 control | 18/25 | 16/25 | 34/50 | 68% |

Paired on the 25 shared golden-valid instances (instance score 0–2 summed over rounds):
E > A8 on `google__adk-python-c_19315fe`, `huggingface__transformers-38332`; A8 > E on
`albumentations-team__albumentations-2337`, `optuna__optuna-6166`, `pandas-dev__pandas-61244`;
20 tied. Exact two-sided sign test p = 1.0. Stage-1 arm A (2 h wall, Aug 23–25) was 36/50
(72%); A8 reproduces it, so the 8 h wall changed nothing for the baseline.

Secondary metrics (per episode; Δ = E − A8 paired per instance, mean over rounds):

| Metric | E | A8 | Δ mean | Δ median | E lower on | sign p |
|---|---|---|---|---|---|---|
| Steps | 52.3 | 56.0 | −4.3 | −2 | 16/24 | 0.15 |
| Cumulative input tokens | 2.75M | 2.99M | −0.29M | −0.008M | 13/25 | 1.0 |
| Output tokens (reasoning) | 29.2k (17.1k) | 29.6k (17.0k) | — | — | — | — |
| Wall-clock (h) | 0.758 | 0.777 | −0.03 | +0.006 | 11/25 | 0.69 |
| Grep-family commands | 17.9 | 20.1 | −1.9 | −1 | 14/23 | 0.41 |
| Edit recall vs gold files | 0.667 | 0.666 | −0.003 | 0 | 4/9 | 1.0 |
| Edit precision | 0.926 | 0.931 | | | | |

Per-step latency: E 43.7 s mean / 22.0 s median vs A8 40.5 s / 20.0 s (output tokens per
step 471 vs 458). Anchor computation itself is 5.6 s per episode. All 116 episodes
submitted; zero container-wall deaths in any run.

### 3.1 Was arm E cheaper? (tokens and wall-clock)

Totals over the 50 paired episodes per arm (25 shared golden-valid instances × 2 rounds):

| Metric | E | A8 | E/A8 |
|---|---|---|---|
| Cumulative input tokens | 97.0M | 111.5M | 0.87 |
| Output tokens | 1.21M | 1.22M | 0.99 |
| Wall-clock | 32.0 h | 33.6 h | 0.95 |
| Steps | 2,266 | 2,482 | 0.91 |
| Grep commands | 812 | 905 | 0.90 |

Paired per instance (ratio of E to A8, each the mean over rounds), the picture is flat:
geometric-mean ratio 0.95 for input tokens (E cheaper on 13 of 25, Wilcoxon signed-rank
p = 0.76), 1.03 for wall-clock (cheaper on 11 of 25, p = 0.40), 0.93 for steps (cheaper on
16 of 25, p = 0.07), 0.94 for greps (p = 0.24). The 13% token gap in the totals comes almost
entirely from two long control episodes (albumentations-2495 −6.4M, transformers-38332
−5.0M tokens for E relative to A8), while three instances go the other way by +1.4M to
+2.4M each — temperature-1.0 episode-length noise, not a systematic saving.

The one consistent trend is steps: ~7% fewer, with ~6% fewer greps, consistent with an
addendum occasionally replacing a follow-up grep. It does not reach wall-clock because
per-step latency is ~8% higher (43.7 s vs 40.5 s mean; medians 22.0 vs 20.0 s): each step
carries a longer context — the addenda add ≈4% to cumulative prompt tokens and are re-read
on every later step — so the tokens the anchors cost roughly pay for the steps they save.
Anchor computation itself is negligible (5.6 s per episode, 12 s budget never binding).

**Answer to the cost question: not measurably cheaper in tokens or wall-clock for a
typical instance; a modest, near-significant reduction in steps; cheaper in raw totals only
because of two outlier control episodes.**

Where the higher per-step latency comes from (step-level decomposition, ~3,000 steps per
arm): steps taken right after an anchored observation are the *fastest* steps in either arm
(median 18.0 s, 205 output tokens, 74 reasoning tokens) versus 26.0 s / 296 / 96 for E's
other steps and 23.0 s / 279 / 90 for A8; decode throughput is identical across arms (12.0
vs 11.9 output tokens per second of step time) and observation size does not explain it
(A8 steps after 3–10k-char observations take 24 s; E's anchored ones 16 s). The anchors are
therefore latency-neutral at the step where they appear; E's higher average is
compositional — the steps anchors eliminate are cheap ones (a follow-up grep plus a short
model turn), so the remaining mix is heavier. Implication for a fast API endpoint (our
hosted reference decoded ~2.5× faster, 11 s median per step): all model time shrinks and
tool time dominates short steps, so the achievable wall saving is the *time share* of the
eliminated steps — roughly 3–5% — not their 7% count, and only if the step reduction holds.

### 3.2 Consistency (the "variance win" hypothesis)

CodeAnchor's stated motivation is that deterministic anchors "make exploration more
disciplined under stochastic LLM control", so a positive result could take three forms:
a correctness win (§3: none), an efficiency win (§3.1: none beyond a modest step trend), or
a **variance win** — the same tasks done more consistently. Round-to-round dispersion per
instance, |log(round 1 / round 2)| (0 = identical rounds), 25 instances:

| Metric | E geo-mean spread | A8 geo-mean spread | E more consistent on | Wilcoxon p |
|---|---|---|---|---|
| Cumulative input tokens | 1.68× | 1.71× | 10/25 | 0.58 |
| Steps | 1.28× | 1.32× | 12/25 | 0.86 |
| Wall-clock | 1.64× | 1.50× | 9/25 | 0.29 |
| Grep commands | 1.42× | 1.64× | 19/25 | 0.14 |
| Output tokens | 1.50× | 1.39× | 7/25 | 0.09 |

Outcome flips between rounds: 4/25 in each arm. Typical dispersion is the same; the only
hint of discipline is in the search phase (grep counts more consistent with anchors). The
extreme tail is shorter with anchors, though: over the 50 episodes per arm, cumulative input
tokens p90 5.9M / max 8.6M (E) vs 4.8M / 17.6M (A8), steps p90 87 / max 109 vs 94 / 167,
wall p90 1.5 h / max 3.7 h vs 2.0 h / 4.4 h. Both of E's albumentations-2495 episodes
(8.6M, 7.0M tokens) were shorter than both of the control's (17.6M, 10.8M), and the
control's transformers-38332 round 2 was a 12.8M-token failing excursion. Two rounds cannot
separate "anchors truncate long excursions" from luck; the **variance rerun** (5 extra
rounds of both arms on the five largest-gap instances, queued after E3) is designed to.

## 3.3 Arm E3: definition-site placement, uncapped (the paper's own trigger surface)

E3 (`code/agent/arm_e3.yaml`, launched 2026-09-04) removed both deviations from the paper
that §5 lists: no caps on symbols, files or users, and anchors also on **file views** —
every `def`/`class` line visible in a `cat` / `sed -n` / `head` / `tail` / `nl` view gets its
"used by" list in definition order, i.e. tags colocated with definitions. The only
truncation is the harness's 10,000-char observation limit (applied to the addendum, never to
the command output; it bound on 15% of addenda). Two rounds, hosts swapped, against the same
A8 control (round 2 is a merge of three pieces after a worker reboot and an image prune
during the run; see `LABNOTES.md`).

| Arm | Round 1 | Round 2 | Total | Rate |
|---|---|---|---|---|
| E3 | 17/25 | 15/25 | 32/50 | 64% |
| A8 control | 18/25 | 16/25 | 34/50 | 68% |

Paired on the 25 shared instances: E3 better on 1 (`google__adk-python-c_19315fe`, both
rounds), control better on 4, 20 tied (sign p = 0.375).

| Metric (per episode) | E3 | A8 | Δ mean | Δ median | E3 lower on | sign p |
|---|---|---|---|---|---|---|
| Steps | 53.2 | 56.0 | −2.5 | −2.5 | 18/25 | **0.043** |
| Cumulative input tokens | 3.16M | 2.99M | +0.17M | +0.03M | 11/25 | 0.69 |
| Wall-clock (h) | 1.05 | 0.78 | +0.22 | +0.12 | 6/25 | **0.015** |
| Grep commands | 19.7 | 20.1 | −0.2 | 0 | 11/23 | 1.0 |
| Edit recall vs gold files | 0.651 | 0.666 | −0.025 | 0 | 5/8 | 0.73 |

Exposure doubled: 17.1 anchored observations per episode (482 grep + 518 view addenda over
58 episodes; median 1.2–1.3k chars, p90 4.9–7.6k), ≈38k addendum chars per episode
(≈9.4k tokens, re-read on every later step), 11.9 s of language-server work per episode,
zero failures. Uptake is unchanged: 1,123 flagged source files, 23% later opened; 188
flagged gold files → 178 opened, 176 patched. **Coverage is unchanged too**: gold files
covered by both arms 121, E3-only 3, A8-only 4, neither 64 — and of those 64 missed files
only 9 were ever named in any E3 addendum, uncapped and with the wider trigger surface.

Cost went the wrong way. Steps fell significantly (−4.5%, p = 0.043) but cumulative input
tokens rose (+36% at the median: the larger observations raise the context carried by every
later step) and wall-clock rose 35% on the mean, +0.12 h at the median, higher on 19 of 25
instances (p = 0.015): per-step latency went from 50 s to 71 s as contexts grew (local
prefill and cache pressure on the GB10), and the longest episodes got longer (max 14.7M
tokens / 5.4 h vs 17.6M / 4.4 h for the control; p90 wall 1.96 h vs 1.98 h). Consistency
did not improve either (round-to-round token spread 1.49× vs 1.71×, E3 more consistent on
12/25; outcome flips 4 vs 4).

**Read:** the paper's placement and the absence of caps buy nothing here. Both variants
say the same thing from opposite ends of the exposure range: the agent reads and acts on
anchors, and the files it misses are the ones no anchor ever names, because they are
non-Python or are referenced only by symbols the agent never searches or opens. On a local
server, more exposure is a net cost.

### 3.4 Variance rerun: 7 episodes per instance on the five largest-gap instances

To test whether arm E's shorter tail (§3.2) was real, both arms were rerun five more times on
the five instances with the largest per-instance token gaps in either direction
(albumentations-2495 and transformers-38332 favoured E; langchain-32996, django-19643 and
gallery-dl-7872 favoured the control), giving 7 episodes per instance per arm, 35 per arm.

| Instance | Arm | Resolved | Tokens min / median / max (M) | Steps median / max | Wall median / max (h) |
|---|---|---|---|---|---|
| albumentations-2495 | E | 0/7 | 4.6 / 7.0 / 9.3 | 101 / 127 | 1.8 / 2.5 |
| | A8 | 0/7 | 1.9 / 7.8 / 19.6 | 105 / 300 | 2.1 / 3.3 |
| transformers-38332 | E | 3/7 | 0.8 / 2.5 / 28.8 | 88 / 209 | 1.1 / 5.5 |
| | A8 | 2/7 | 0.3 / 12.8 / 36.4 | 151 / 300 | 1.9 / 7.2 |
| langchain-32996 | E | 1/7 | 2.0 / 3.0 / 5.1 | 63 / 87 | 1.0 / 1.9 |
| | A8 | 1/7 | 1.2 / 2.3 / 3.0 | 59 / 71 | 1.2 / 1.6 |
| django-19643 | E | 1/7 | 3.8 / 5.3 / 7.9 | 92 / 114 | 3.7 / 6.1 |
| | A8 | 1/7 | 3.9 / 4.8 / 17.8 | 91 / 300 | 2.8 / 4.2 |
| gallery-dl-7872 | E | 1/7 | 0.8 / 1.6 / 6.0 | 59 / 119 | 0.9 / 1.8 |
| | A8 | 1/7 | 1.5 / 3.2 / 4.3 | 77 / 103 | 0.9 / 1.4 |

- **Level: no difference.** Pooled geometric-mean tokens 4.02M (E) vs 4.19M (A8); wall 1.55 h
  vs 1.42 h; resolved 6/35 vs 5/35; one context-limit blow-up (262k tokens) in each arm, both on
  transformers-38332. Per instance, only langchain-32996 separates (E 1.28× the tokens,
  rank-sum p = 0.04); the two "E was cheaper" instances from §3.1 are not cheaper at the
  median with 7 episodes (0.90× and 0.20×, p = 0.75 and 0.85).
- **Spread: the direction Ian predicted, not the significance.** The control produced the
  runaway episodes — three hit the 300-step cap (albumentations, transformers, django) against
  none for E, whose longest episode was 209 steps — so its token tail is fatter: p90 17.7M vs
  9.3M, max 36.4M vs 28.8M, sd of log tokens about the per-instance median 1.03 vs 0.74. But the
  pooled dispersion test is flat (median |log x − median| 0.28 vs 0.26, rank-sum p = 0.71) and a
  bootstrap on the mean-absolute-deviation ratio gives 0.82 with 95% CI [0.42, 1.57]. Per
  instance, albumentations-2495 is the one clear tightening (spread 0.27 vs 0.58, p = 0.13);
  transformers-38332 and gallery-dl-7872 are *more* dispersed with anchors.
- **Wall time does not inherit the token tail**: p90 4.9 h (E) vs 3.7 h (A8), max 6.1 h vs 7.2 h —
  arm E's long django episodes were slow per step (large contexts), so fewer runaway steps did not
  mean less wall.

**Read:** consistent with CodeAnchor's own framing that its saving is mostly variance, the
anchors trim the worst excursions (no step-cap hits vs three) without changing typical cost; but
35 episodes per arm cannot establish the spread reduction (CI spans 1), and on this local server
the wall-clock tail is not shorter. A variance win here is plausible and unproven.

## 4. Why nothing moved

**Uptake is real.** Across 58 E episodes the addenda flagged 578 distinct source files as
"referenced but NOT in this grep output"; the agent later opened 24% of them. Of the 99
flagged files that were gold-patch files, it opened 99 and patched 92. When the anchor points
at the right file, the agent follows it.

**Coverage does not change.** Union over rounds, gold files covered: both arms 123, E-only 3,
A8-only 2, neither 64. The incomplete-refactor gap is the same gap. Classifying the 69 gold
files both arms missed in round 1:

| Missed gold files (69) | n | % |
|---|---|---|
| Non-Python (docs, yaml, config) | 28 | 41% |
| Python source never referenced by any symbol the agent searched | 27 | 39% |
| Created or deleted by the gold patch | 7 | 10% |
| Python source shown in a capped addendum and ignored | 4 | 6% |
| Python source hidden by the cap (uncapped replay only) | 3 | 4% |

The 39% row is the technique's ceiling in this form. Example: lerobot-2808's gold patch
touches ten `examples/*.py` scripts that use the camera *config* classes; the agent grepped
camera internals (`latest_frame`, `RealSenseCamera`) whose references genuinely stay inside
`src/lerobot/cameras/`. No fact attached to those greps could mention `examples/`. Anchors
attached to grep results answer the agent's questions more completely; they do not change
which questions it asks. CodeAnchor's own placement (tags at definition sites the agent
*reads*) has the same property, with a wider trigger surface.

**The cap counterfactual** (`code/analysis/anchor_replay.py`, `code/analysis/anchor_cap_counterfactual.py`):
241 of 256 anchored greps from round 1 were re-executed in pristine containers with every cap
removed. Uncapped addenda would have named 2,389 files vs 657 (3.6×) and 937 vs 705 symbols,
yet only 3 of the 69 missed gold files (optuna-c_058e8bc ×2, lerobot ×1). Of the 3 Python gold
files the control patched and E did not, 1 was cap-hidden and 2 were shown.

## 5. Deviations from CodeAnchor, and what they cost

1. **Caps** (theirs: none). Measured above: ≤4% of the coverage gap. Kept token overhead at
   ~4% vs their ~10%.
2. **Placement on grep output, not definition sites.** Forced by patch grading. Narrows the
   trigger surface to symbols the agent greps (≈18 commands/episode) rather than every
   function it reads.
3. **Language-server references instead of a PyCG call graph**, live rather than offline.
   More precise (aliases, methods disambiguated), no inheritance edges beyond what the
   `[subclass]`/`[override]` line-text tags recover.
4. **Test files** dominate many "used by" lists (the agent may not edit tests). Source files
   sort first and are never hidden behind tests, so this is token cost, not information loss.

## 6. Follow-ups


- **E3 (definition-site placement, uncapped) was run — §3.3: null on outcome, worse on cost.**
  The remaining lever within this technique family is the agent's own navigation, not the
  facts it is shown.
- **Variance rerun done (§3.4)**: tails trimmed (0 vs 3 step-cap episodes), typical cost equal,
  spread reduction not significant with 35 episodes per arm; wall tail not shorter locally.
- Non-Python gold files (41% of misses) need a different signal entirely (docs/config search
  hints), out of scope for LSP anchors.

## 7. Reproduction map

| Path | What |
|---|---|
| `code/agent/anchor_env.py`, `code/agent/anchor_runner.py`, `code/agent/arm_e.yaml`, `code/agent/arm_a8.yaml` | Environment subclass, launcher shim, arm configs |
| `code/agent/run_batch_waved.sh` (mode `anchor`) | Waved runner |
| `code/lsp-tool/src/lsp_tool/anchor.py` (+ `daemon.py`, `cli.py`) | Batch anchor op, `lsp anchor`, raw/uncapped mode |
| `code/analysis/anchor_compare.py` | Paired comparison + telemetry |
| `code/analysis/anchor_replay.py`, `code/analysis/anchor_cap_counterfactual.py`, `code/analysis/anchor_variance.py` | Uncapped replay and missed-file classification |
| `results/s1-arm-{e,a8}-r{1,2}.json`, `results/*.anchor_log.jsonl` | Per-instance outcomes; per-grep anchor telemetry (trajectories on the lab NAS) |
| `results/summary_*.txt`, `results/episodes_*.csv`, `results/replay_uncapped_e_r1.jsonl`, `results/cap_counterfactual_r1.txt`, `results/cost_e_vs_a8.txt`, `results/variance_rerun.txt`, `results/s1-arm-{e,a8}-var-r*.json` | Summaries, episode CSVs, uncapped replay + counterfactual, cost/consistency tables, variance rerun |
| `LABNOTES.md`, `NOTES.md` | Chronology incl. the four launches; protocol + run ledger |
