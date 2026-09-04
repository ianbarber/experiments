# Do CodeAnchor-style anchors on grep output help a local coding agent refactor? (Arm E)

**Dates:** 2026-09-02 → 2026-09-04 · **Benchmark:** SWE-Bench ProMax python subset (29
instances, 25 golden-valid on our LAN) · **Scaffold:** mini-swe-agent 2.4.6 · **Model:**
Qwen3.8-27B-FP8, local SGLang/DSpark on the GB10 · **Design:** arm E (anchors) vs arm A8
(stock baseline), 2 rounds each, hosts swapped between rounds, 8 h container wall.

## TL;DR

- **Null.** Arm E resolved 32/50 (64%) vs the control's 34/50 (68%). Paired on 25
  instances with scores summed over rounds: E better on 2, control better on 3, 20 tied
  (sign p = 1.0). Steps −4.3/episode (E lower on 16 of 24, p = 0.15, n.s.), cumulative
  input tokens −0.29M mean but −0.008M median (13 vs 12), wall −0.03 h mean / +0.006 h
  median, edit recall against gold files identical (0.667 vs 0.666).
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

Reading the token/wall question directly: the weak trend to fewer steps and fewer greps is
consistent with the agent using an addendum in place of a follow-up grep, but the addendum's
own context cost (≈4% of prompt tokens, re-read on every later step) and slightly longer
prefill cancel it. Net tokens and wall time are unchanged within noise.

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

**E3 is running** (launched 2026-09-04, both rounds concurrently; config `code/agent/arm_e3.yaml`):
definition-site placement (tags on every def/class visible in `cat`/`sed -n`/`head`/`tail`
views, in addition to grep hits) with no caps, per the paper. Results will be appended here.

- **E3 — definition-site placement, uncapped.** Annotate `cat`/`sed -n` views of a file with
  each function's users (full compact file list, source-first, no enclosing-scope detail),
  in addition to grep hits. This is the paper's own trigger surface and the only variant the
  diagnostics suggest could reach the 39% row. Cost ≈ 2 rounds × 2 arms ≈ 2 days.
- Uncapped E2 alone is faithful but predicted to change little.
- Non-Python gold files (41% of misses) need a different signal entirely (docs/config search
  hints), out of scope for LSP anchors.

## 7. Reproduction map

| Path | What |
|---|---|
| `code/agent/anchor_env.py`, `code/agent/anchor_runner.py`, `code/agent/arm_e.yaml`, `code/agent/arm_a8.yaml` | Environment subclass, launcher shim, arm configs |
| `code/agent/run_batch_waved.sh` (mode `anchor`) | Waved runner |
| `code/lsp-tool/src/lsp_tool/anchor.py` (+ `daemon.py`, `cli.py`) | Batch anchor op, `lsp anchor`, raw/uncapped mode |
| `code/analysis/anchor_compare.py` | Paired comparison + telemetry |
| `code/analysis/anchor_replay.py`, `code/analysis/anchor_cap_counterfactual.py` | Uncapped replay and missed-file classification |
| `results/s1-arm-{e,a8}-r{1,2}.json`, `results/*.anchor_log.jsonl` | Per-instance outcomes; per-grep anchor telemetry (trajectories on the lab NAS) |
| `results/summary_e_vs_a8.txt`, `results/episodes_e_vs_a8.csv`, `results/replay_uncapped_e_r1.jsonl`, `results/cap_counterfactual_r1.txt` | Summary, episode CSV, uncapped replay data, counterfactual |
| `LABNOTES.md`, `NOTES.md` | Chronology incl. the four launches; protocol + run ledger |
