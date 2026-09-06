# Lab notebook — CodeAnchor-style anchors (arm E / E3 / variance rerun)

Chronological, unedited except for this header and hostname scrubbing. Continues the notebook of the
[2026-08-22 LSP study](../2026-08-22-lsp-agents-promax/LABNOTES.md).

## 2026-09-02 — Arm E: CodeAnchor-style passive anchors on grep output (design + build)

Ian's ask: test the CodeAnchor idea (arXiv 2606.26979, "How Much Static Structure Do Code
Agents Need? A Study of Deterministic Anchoring") — inject semantic/code-linkage facts as
an *addendum to grep/rg results*, powered by the LSP, and look at tokens/wall-time as well
as resolve rate. CodeAnchor itself writes static-analysis tags (`# used by: fetch,
lang_stats`) into source files as comments, purely passively (prompt unchanged), and
reports on SWE-bench Lite/Verified: localization Func@5 +2.2/+1.2 pp (McNemar p=0.04/0.02),
−1.5 to −1.6 tool rounds, +9–10% input tokens; their pilot found an optional call-graph
tool went unused ("the agent typically relied on plain grep") — exactly our arm B/C2
non-adoption finding, which is why passive injection is the right next lever here.

**Why grep-addendum instead of in-file comments**: our patches are `git diff`s and the eval
grades the diff — comments injected into source would leak into patches. Appending to the
grep *observation* keeps the environment byte-identical and the prompt byte-identical to A.

**Baseline substrate (arm A trajectories, r1/r2, 58 episodes)**: 18.4/17.1 grep-family
commands per episode (median 15/17), ~20k chars of grep output per episode, **0 `rg`
uses**, 0 `git grep` worth noting; 88% of greps sit in `cd /testbed && ...` chains, 24%
are piped into head/tail/wc etc., 2% via `xargs grep`, 3% redirected to a file. Mean
episode: 60/72 steps, 2.9M/5.0M cumulative input tokens, 25k output tokens (14k
reasoning), 0.64/0.87 h wall (from response timestamps).

**Mechanism (built today)**:
- `lsp_tool/anchor.py` + daemon op `anchor` + CLI `lsp anchor` (JSON on stdin): for each
  grep hit (path:line, or path:text located in the file) pick the identifier matching the
  grep pattern's tokens (fallback: def/class name) → textDocument/definition → group hits
  by defining position (semantic disambiguation of same-named symbols) → for up to 8
  candidates textDocument/references → render up to 4 symbols: `name kind Qual.name
  defined path:line (k hits above)`, `used by N sites in M files:` with per-file lines
  listing the enclosing scope of each use (document symbols), tagged `[import]`,
  `[subclass]`, `[override]`; then `referenced but NOT in this grep output: file (n), …`
  — the reference-completeness signal aimed at the measured failure mode. Zero-reference
  definitions (typically overrides) collapse into one trailing line. Caps: 40 hits, 4
  symbols, 5 files/symbol, 3 scopes/file, 2800 chars, 12 s wall budget → partial output.
- `agent/anchor_env.py`: `AnchorDockerEnvironment(DockerEnvironment)` for mini-swe-agent
  (`environment_class:` dotted path works in 2.4.6). At container start it `docker cp`s
  the updated package into the image's /opt/lsp-tool venv and warms the Pyrefly daemon;
  after every command that mentions a grep-family tool it parses the *visible* output
  (head/tail elision respected), and appends the addendum. Post-processing the observation
  (rather than shadowing the grep binary) means `grep | head`, `xargs grep`, `cd x &&
  grep` are all covered and nested scripts/command substitutions can never be corrupted.
  Telemetry per grep to `anchor_log.jsonl` (status anchored/empty/no_hits/not_ready/
  timeout, hits, symbols, chars, seconds).
- Unit tests of the command parser on the 533 real grep commands of A-r1: 30 yield no
  identifier token, the rest 1–5 (cap 12). In-container tests (worker-a):
  albumentations 40 hits → 5 symbols in 1.3 s; transformers (3,556 py files; Pyrefly
  ready in 2 s) 8 hits → 3 symbols 4.2 s cold / 1.0 s warm, hub symbol `PreTrainedModel`
  (946 refs, 393 files) 4.3 s. Output ~1.2–3.4k chars before the 2800 cap.

**Arms**: E = A's prompt + anchors (promax-lsp images, 8 h wall); **A8** = A re-run with
the 8 h wall as the same-period paired control (serving stack identical, hosts swapped
between rounds). Existing A runs (2 h wall, 2026-08-23/25) kept as a secondary reference.

**Ops note**: SGLang had been stopped 08-29 and the local HF weight cache had been
emptied (blobs gone, snapshot symlinks dangling); spark's cached `alpine` image is amd64
(exec format error). Restored the 29 GB FP8 weights from `/mnt/nas/hf-cache` with an
arm64 alpine container, then `start-dspark.sh` (DSpark draft re-downloads from HF).

## 2026-09-02 (evening) — round 1 launched: s1-arm-e-r1 (worker-a) + s1-arm-a8-r1 (worker-b)

- Harness-level smoke of `AnchorDockerEnvironment` in a live promax-lsp container: anchored
  greps cost 0.2–2.5 s including the docker exec; `-l`/`-c`/pipe-from-python greps
  correctly untouched; telemetry rows written.
- **Launch #1 of E failed (5 ValidationError, 0 rollouts)**: mini-swe-agent's SWE-bench
  runner injects the instance `image` only when `environment_class` is literally
  `"docker"` (or `swerex_modal`), so a dotted custom class receives no image. Killed
  (careful: `pkill -f <run name>` also matches the ssh shell issuing it — self-kill).
  Fix: `agent/anchor_runner.py` shim rebinds the `docker` name to
  `anchor_env.AnchorDockerEnvironment` before handing over to the stock `mini-extra`
  main; `run_batch_waved.sh` gained mode `anchor` (lsp images/dataset + shim). Verified
  mapping + `swebench --help` through the shim.
- Serving restored: DSpark stack up at 25.6 tok/s single-stream (short prompt), reasoning
  separated at `reasoning_effort: medium`.
- **Launched** (~20:40 PT): `s1-arm-e-r1` worker-a, mode anchor, waves of 5, 3 rollout
  workers, 2 eval workers; `s1-arm-a8-r1` worker-b, mode base, waves of 4, 3/2. Same 29-id
  regex and data order as stage 1. Analysis script: `analysis/anchor_compare.py` (paired
  resolve + Δsteps/Δtokens/Δwall/Δgreps/Δrecall, anchor telemetry).
- **E-r1 restarted once more (~20:55 PT)** after its first 5 minutes had already produced
  anchored observations in the live loop (3 anchored greps, 4 symbols / 1.7k chars each,
  2.5–3 s): added a fidelity guard — mini-swe-agent elides observations ≥10,000 chars
  (head 5k + tail 5k + "too long" warning), so an addendum appended to a long grep output
  could trigger elision that arm A would never see. The addendum is now trimmed at a line
  boundary to the room left under 10,000 chars, or skipped (`no_room`, logged) when <500
  chars remain. Partial output of the first attempt kept as `s1-arm-e-r1-aborted-v1`.
- Plan: when a host's round-1 run prints `[waved] done`, launch round 2 there with arms
  swapped (`s1-arm-e-r2` on worker-b in mode anchor, waves of 4; `s1-arm-a8-r2` on
  worker-a in mode base, waves of 6). Then `analysis/anchor_compare.py --e
  s1-arm-e-r1,s1-arm-e-r2 --a s1-arm-a8-r1,s1-arm-a8-r2`.
- **E-r1 restarted a third time (~21:35 PT) — parser coverage bug.** The first 30 min of
  telemetry showed 27 `no_hits` / 2 anchored: `grep -n PAT file.py` (single file operand)
  prints `LINE:text` with no filename, and the hit parser required a `path.py:` prefix —
  roughly half of the model's greps. Fix: `extract_search` now records each grep
  invocation's file operands and recursion flags; when the invocations agree on exactly
  one non-recursive file, line-only hits are attributed to it (`single_file`); line-only
  output that cannot be attributed is logged as `unattributed`. Unit-tested on the exact
  commands from the aborted run (10/10). Partial output kept as `s1-arm-e-r1-aborted-v2`.
  Lesson: always read the first 20–30 telemetry rows of a new arm before letting it run.

## 2026-09-03 — E-r1 fourth launch (~01:50 PT): candidate-file attribution; A8-r1 progressing

- After ~4 h of E-r1 (wave 1 nearly complete: 4/5 submitted) the telemetry read 116 grep
  events: 39 anchored, 43 no_hits (ls/find/-l lists — correct), 6 empty (hits on string
  constants / not-yet-defined symbols — correct), 1 no_room (output already >10k),
  **27 unattributed** (23%). Three recoverable shapes: `grep -r PAT single_file.py`
  (grep omits the filename for a single file even with -r), chains of single-file greps
  on different files (`grep -n A f1; echo ===; grep -n B f2`), and `head/sed FILE |
  grep -n` (line numbers relative to the piped chunk). Fix: `-r` + single file operand
  with an extension → single_file; otherwise every `.py` path named anywhere in the
  command becomes a candidate and the **daemon matches the hit text** (same line first,
  else anywhere in the candidate) to choose the file. Verified on the scratch repo.
  Restarted E-r1 a fourth time so all counted episodes share one mechanism; partial
  outputs kept as `s1-arm-e-r1-aborted-v{1,2,3}` (not used). Cost ≈ 3 h on the critical
  path (E-r1 now ends ~13:00; A8-r1 on worker-b unaffected, wave 2 of 8 in progress).
- A8-r1 wave 1: albumentations-2337 resolved, albumentations-2495 not; dspy-9193 and
  transformers-38788 golden-invalid as always.
- **E-r1 wave 1 rolled out (~04:00 PT), first qualitative look** (5 trajectories):
  anchored observations per episode 20 / 25 / 7 / 11 / 6; files flagged as "referenced
  but NOT in this grep output" 29 / 34 / 20 / 36 / 7, of which the agent later opened
  9 / 15 / 0 / 2 / 1 and ended up patching 1 / 2 / 0 / 1 / 0. So the signal is noticed
  and sometimes acted on. Caveat for the writeup: a large share of flagged references
  are test files (the agent may not edit tests); source files are always listed first
  and are never hidden behind tests, so this is a token-cost issue, not an information
  loss. Mechanism stays frozen; a "tests aggregated" variant is a follow-up if E shows
  promise. Telemetry after the final relaunch: 0 unattributed, median anchor 0.37 s.
- **~15:30 PT status**: A8-r1 on its last wave (8/8); E-r1 at wave 3/6 (14 episodes
  submitted; langextract-239 — golden-invalid here — running >2 h). Interim on the 8
  shared golden-valid instances evaluated so far: E 5/8, A8 5/8. E telemetry: 330 grep
  events → 165 anchored (50%), 113 no_hits, 40 empty, 11 unattributed (3%), 1 no_room;
  median anchor 0.25 s, p90 1.6 s, 0 errors.
- Round-2 launches chained on the workers themselves (robust to my session's watchers
  being killed): `~/refactorbench-eval/chain_e_r2.sh` on worker-b waits for
  `[waved] done: s1-arm-a8-r1` then runs E-r2 (anchor, waves of 4, 3/2);
  `chain_a8_r2.sh` on worker-a waits for `[waved] done: s1-arm-e-r1` then runs A8-r2
  (base, waves of 6, 3/2). Logs: `runs/chain_*.log`, `runs/s1-arm-{e,a8}-r2.log`.

## 2026-09-03 (afternoon) — A8-r1 complete; E-r2 auto-launched

- **s1-arm-a8-r1 (worker-b, 8 h wall): 18/25 golden-valid (72%)**, 29/29 Submitted, **0 wall
  deaths**; mean 54 steps (median 42), 19.2 grep cmds/episode, 2.89M cumulative input
  tokens, 28.8k output (16.6k reasoning), 0.68 h wall/episode (median 0.32 h). Matches
  the stage-1 A average (72%) — the 8 h wall changed nothing for A, as expected (A had
  few wall deaths). Archived to NAS.
- The worker-b chain fired: `s1-arm-e-r2` started (anchor mode, waves of 4) right after.

## 2026-09-03/04 — ROUND 1 COMPLETE: E 16/25 (64%) vs A8 18/25 (72%)

- **s1-arm-e-r1 (worker-a): 16/25 golden-valid (64%)**, 29/29 Submitted, 0 wall deaths.
  Mean 52.6 steps (median 41), 18.3 greps/episode, 2.72M cumulative input tokens
  (median 1.30M), 28.9k output (17.0k reasoning), 0.71 h wall (median 0.42 h). Anchor
  telemetry: 509 grep-family commands → 256 anchored (50%), mean 8.8 anchored
  observations and ~10k addendum chars per episode, 4.9 s of anchor computation per
  episode, 0 not-ready/timeouts/errors.
- **s1-arm-a8-r1 (worker-b): 18/25 (72%)** — see above.
- **Paired (25 golden-valid instances): E better 1, A better 3, sign p=0.625.**
  Δsteps −1.3 (E lower on 14/24, p=0.54), Δinput-tokens −0.10M mean / +0.03M median
  (13 vs 12), Δwall +0.08 h mean (E higher on 14/25, p=0.69), Δgreps +0.8, Δedit-recall
  −0.015. **Round-1 read: null on resolve, steps, tokens and wall; anchors cost
  nothing measurable in latency (≈5 s/episode) but also bought nothing.** Round 2
  (hosts swapped) is running: E-r2 worker-b, A8-r2 worker-a (auto-chained).
- **Round-1 mechanism diagnostics (E-r1 trajectories, 29 episodes)**:
  1. *Uptake*: anchors flagged 316 distinct source files as "referenced but NOT in this
     grep output"; the agent later opened 21% of them. Of the 45 flagged source files
     that were gold-patch files, **45/45 were later opened and 40 patched** — the signal
     is noticed and acted on when it points at the right place. Of the 120 gold files E
     patched, 90 had appeared in some addendum.
  2. *Coverage*: on the 25 shared golden-valid instances, gold files covered by both arms
     118, E-only 3, A8-only 3, **neither 69** — the incomplete-refactor gap is untouched.
     Breakdown of the 69: 41% non-python (docs/yaml/config — unreachable by a Python
     reference graph), 42% existing .py source that **never appeared in any addendum and
     was never opened**, 6% files the gold patch creates, 4% deleted, 6% mentioned in an
     addendum but ignored, 1% opened but not patched.
  3. *Why never flagged*: two mechanisms. (a) The agent never searched the right symbol —
     lerobot-2808's 10 `examples/*.py` gold files use the camera *config* classes; the
     agent grepped camera internals (`latest_frame`, `RealSenseCamera`), whose
     references genuinely stay inside `src/lerobot/cameras/`. Passive injection is
     bounded by the questions the agent asks — same limit as CodeAnchor's tags (which
     attach to what the agent opens). (b) Display cap: 61/244 (25%) of "NOT shown" lines
     were truncated at 6 files ("+N more"), hiding 1,779 file mentions in total (django
     473, gallery-dl 331) — hub-symbol repos lose exactly the long tail a large refactor
     needs. A v2 should print the full compact file list (paths + counts, ~30) and drop
     the per-file enclosing-scope detail instead.
  4. *Cost*: per-step latency E 41.9 s mean / 20.0 s median vs A8 37.1 / 21.5 (output
     tokens/step 471 vs 458): the mean gap is tail-driven (one E episode at 3.4 h), not
     anchoring (≈5 s/episode total). Addenda ≈4% of cumulative prompt tokens.

## 2026-09-04 — cap counterfactual (Ian's confound concern): the caps are not the story

- Ian: "when investigating techniques, be very careful about introducing new constraints"
  — the addendum caps (4 symbols / 6 not-shown files / 2,800 chars) are mine, CodeAnchor
  caps nothing. Mitigation (approved): `analysis/anchor_replay.py` re-executed every
  anchored grep of E-r1 (29 episodes, 256 anchored observations, 241 replayed, 15 LS
  errors) in pristine promax-lsp containers with all caps removed (`raw` mode of the
  anchor op; separate package copy so E-r2/A8-r2 were untouched), then
  `analysis/anchor_cap_counterfactual.py` classified the 69 gold files missed by BOTH arms
  on the 25 shared instances. Result:
  | missed gold files (69) | n | % |
  |---|---|---|
  | non-python (docs/yaml/config) | 28 | 41% |
  | python source never reachable from the symbols the agent searched | 27 | 39% |
  | file created/deleted by the gold patch | 7 | 10% |
  | python source SHOWN in a capped addendum and ignored | 4 | 6% |
  | **python source HIDDEN BY THE CAP** (only in the uncapped replay) | **3** | **4%** |
  Uncapped addenda would have named 3.6× more files (2,389 vs 657) and 937 vs 705
  symbols — but only 3 of the 69 missed gold files (optuna-c_058e8bc ×2, lerobot ×1).
  Of the 3 python gold files A8 patched and E didn't, 1 was cap-hidden, 2 were shown.
  **Read: the cap is a real deviation from the paper but explains at most ~4% of the
  coverage gap; the binding limits are (i) 41% non-python targets a Python reference
  graph cannot reach and (ii) 39% source files never referenced by anything the agent
  chose to grep.** Replay data: `runs/s1-arm-e-r1/replay_uncapped.jsonl` on NAS.
- Implication for follow-ups: an uncapped E2 is the faithful thing to run and is cheap
  to build, but the counterfactual predicts little change. A variant that would extend
  reach is CodeAnchor's own placement — tags on *definition sites the agent reads* (i.e.
  annotate `cat`/`sed -n` views of a file with each function's users), not only grep hits.

## 2026-09-04 — STAGE E COMPLETE: two rounds, E 32/50 (64%) vs A8 34/50 (68%) — null

- **s1-arm-e-r2 (worker-b): 16/25**; **s1-arm-a8-r2 (worker-a): 16/25**; both 29/29
  Submitted, 0 wall deaths (8 h wall). Two-round totals: **E 32/50 (64%), A8 34/50 (68%)**.
- **Paired, 25 shared golden-valid instances, instance scores summed over rounds: E better
  2 (adk-19315fe, transformers-38332), A8 better 3 (albumentations-2337, optuna-6166,
  pandas-61244), 20 tied → sign p=1.0.**
- Secondary (paired Δ = E − A8 per instance, mean over rounds):
  | metric | E mean | A8 mean | Δ mean | Δ median | E lower on | sign p |
  |---|---|---|---|---|---|---|
  | steps/episode | 52.3 | 56.0 | −4.3 | −2 | 16/24 | 0.15 |
  | cumulative input tokens | 2.75M | 2.99M | −0.29M | −0.008M | 13/25 | 1.0 |
  | output tokens | 29.2k | 29.6k | — | — | — | — |
  | wall h/episode | 0.758 | 0.777 | −0.03 | +0.006 | 11/25 | 0.69 |
  | grep cmds/episode | 17.9 | 20.1 | −1.9 | −1 | 14/23 | 0.41 |
  | edit recall vs gold | 0.667 | 0.666 | −0.003 | 0 | 4/9 | 1.0 |
  Per-step latency E 43.7 s vs A8 40.5 s mean (medians 22.0 vs 20.0 s); anchor compute
  5.6 s/episode; addenda 8.4/episode, ~9.4k chars/episode ≈ 4% of cumulative prompt tokens.
  **Read: no resolve effect; a weak, non-significant trend to fewer steps and fewer greps
  (E used the anchors instead of some follow-up greps) that does not translate into fewer
  tokens or less wall time — the addendum's own context cost roughly cancels the saved
  steps, and per-step latency is slightly higher.**
- Uptake (58 E episodes): 578 source files flagged as "referenced but NOT in this grep
  output", 24% later opened; **99 flagged gold files → 99 opened, 92 patched**. Gold-file
  coverage (union over rounds): both arms 123, E-only 3, A8-only 2, **neither 64** — the
  incomplete-refactor gap is untouched. Cap counterfactual (see 2026-09-04 above): the
  caps hid 3 of the 69 round-1 missed files; 41% non-python, 39% never referenced by any
  symbol the agent searched.
- Telemetry health across both E rounds: 1,001 grep-family commands, 487 anchored, 0
  not-ready, 0 timeouts, 0 errors. Analysis outputs on NAS:
  `/mnt/nas/refactorbench/analysis/anchor_*`. Write-up: `ANCHORS-REPORT.md`.

## 2026-09-04 — Arm E3 launched: definition-site anchors, uncapped (paper-faithful)

- Ian: port the E write-up to the experiments repo (separate entry, referencing the
  2026-08-22 study) and kick off E3 meanwhile.
- **E3 design** (`agent/arm_e3.yaml`; env flags `anchor_views`, `anchor_uncapped`):
  prompt byte-identical to A; anchors on grep hits as in E **plus file views** — every
  `def`/`class` line visible in a `cat` / `sed -n` / `head` / `tail` / `nl` view (numeric
  prefixes stripped; lines located in the file by text) gets its "used by" list, in
  definition order, headed `for <file>` and with "referenced but NOT in this file" — i.e.
  CodeAnchor's own placement (tags colocated with definitions). **No caps** on symbols,
  files, users or missing-file lists (paper caps nothing); the only truncation is the
  harness's 10,000-char observation limit, applied to the addendum at a line boundary
  (never to the command output) and logged (`trimmed` / `no_room`). Budget 25 s, exec
  timeout 60 s, 60 grep hits / 80 view definitions per observation. Variables are not
  tagged (paper tags functions/classes).
- Smoke in the transformers container: `cat f | head -60` → 2 definition tags (2.2 s);
  `cat -n f | sed -n 1,80p` → 3 tags (0.6 s); `cat` of a 287k-char file → `no_room`
  (harness already elides it; untouched); uncapped grep on the hub `loss_function`
  (224 refs / 153 files) → 9.6k-char addendum trimmed to fit (4.1 s).
- Launched concurrently (both workers free): `s1-arm-e3-r1` worker-a (waves of 5,
  3/2), `s1-arm-e3-r2` worker-b (waves of 4, 3/2). Control = the existing A8 rounds (same
  serving stack, 2 days apart — documented).
- Report: added §3.1 "Was arm E cheaper?" (totals E/A8 0.87 tokens, 0.95 wall, 0.91
  steps; paired geo-mean ratios 0.95 / 1.03 / 0.93, Wilcoxon p 0.76 / 0.40 / 0.07; the token
  total is two outlier control episodes). Pushed (`aa51c12`); remote had Ian's hostname
  scrub commit (worker-a→worker-a, worker-b→worker-b, strix-halo→strix-halo, IPs→dgx-spark/nas)
  — rebased on it; rule saved for future pushes.
- **Ian: the paper signals the saving is largely variance; our long episodes might be a
  positive example — rerun both arms on those tests a few times.** Queued the variance
  rerun: 5 extra rounds of E and of A8 on the 5 instances with the largest |Δ tokens|
  (albumentations-2495 −6.4M, transformers-38332 −5.0M; langchain-32996 +2.4M,
  django-19643 +2.1M, gallery-dl-7872 +1.4M — both directions for symmetry). Runs
  `s1-arm-e-var-r{1..5}` (worker-a, after E3-r1) and `s1-arm-a8-var-r{1..5}` (worker-b,
  after E3-r2), chained via `~/refactorbench-eval/chain_var_*.sh`; filter `var5.re`.
  Question: on these instances, is E's episode-length distribution genuinely shorter
  (positive example) or is the r1/r2 gap within the per-instance variance?
- **Ian's framing of a positive result: (1) correctness (unlikely), (2) efficiency, (3)
  variance/consistency.** Added to the report: §3.1 step-level latency decomposition —
  steps right after an anchored observation are the fastest in either arm (median 18 s,
  205 out tokens) vs 26 s for E's other steps and 23 s for A8; decode proxy identical
  (12.0 vs 11.9 tok/s) → anchors are latency-neutral where they appear, E's higher mean is
  compositional (cheap steps removed); on a fast API the achievable wall saving is the
  time share of the removed steps (~3–5%). §3.2 consistency — round-to-round |log(r1/r2)|
  per instance: tokens 1.68× (E) vs 1.71× (A8), steps 1.28× vs 1.32×, outcome flips 4 vs 4
  → no variance win on typical dispersion; greps more consistent with anchors (19/25,
  p=0.14); extreme tail shorter (max tokens 8.6M vs 17.6M, p90 wall 1.5 h vs 2.0 h). The
  variance rerun (queued) is the test of the tail claim. Pushed.
- **Incident (2026-09-04 13:12 PT, worker-b)**: a Docker image prune during Ian's disk
  cleanup (h3 models, anaconda envs, HF cache, ~/models removed; 52 → 692 GB free)
  deleted `promax-lsp:django__django-19643` after E3-r2's wave 2 had loaded it but before
  its container started (3 workers, 4-instance wave) → `docker run` exit 125 →
  `CalledProcessError`, empty patch. Other wave-2 instances unaffected (their containers
  were running); later waves reload images from the NAS. Fix: separate run
  `s1-arm-e3-r2-fix` (django only, anchor mode) launched on worker-b; to be merged into the
  E3-r2 results at analysis time (the waved runner would otherwise overwrite
  `pass_rate_wave1.json`). Rule: no docker prunes on a worker while a run is active.
- **Chain-script bug (caught in time)**: the variance-rerun chains written via an
  unquoted remote heredoc lost the round suffix (`$r` expanded at write time) → all five
  rounds would have shared one run name and each round's `rm -rf` would have deleted the
  previous round. Arm E round 1 was already running as `s1-arm-e-var-r`; detached it from
  the chain and installed `chain_var_e_fix.sh` (renames it to `-r1` on completion, then
  runs rounds 2–5 with correct names); replaced worker-b's not-yet-fired chain with
  `chain_var_a8.sh`. Second lesson (repeated): `pgrep -f`/`pkill -f` patterns that appear
  literally in the invoking ssh command match the ssh shell itself — use `[c]hain…`
  bracket patterns and verify from a separate session.
- **E3-r1 (worker-a): 17/25 (68%)** — done; E3-r2 on its last waves.
- **Incident 2 (2026-09-04 21:17 PT): worker-b rebooted** (uptime reset, login at 21:17,
  dockerd restarted 21:17:36) while E3-r2 was rolling out wave 6 → runner and containers
  gone; waves 1–5 (20 episodes, `pass_rate_wave1..5.json`) intact. Recovery: continuation
  run `s1-arm-e3-r2b` (9 remaining instances: pandas, pipenv, supervision, dspy-1801/8105/
  9047, verl-3915/4185, ragas; anchor mode, waves of 4) launched 22:55; A8 variance chain
  replaced by `chain_var_a8_v2.sh` waiting for `[waved] done: s1-arm-e3-r2b`. E3-r2 =
  merge of `s1-arm-e3-r2` (waves 1–5), `s1-arm-e3-r2-fix` (django), `s1-arm-e3-r2b`.
  Ask Ian for a heads-up before rebooting a worker with a run active.

## 2026-09-05 — E3 COMPLETE: 32/50 (64%) vs A8 34/50 — null on outcome, worse on cost

- E3-r2 merged (waves 1–5 + django fix + 9-instance continuation): 15/25. Two-round E3
  32/50; paired vs A8: E3 better 1 (adk-19315fe both rounds), A8 better 4, 20 tied
  (p=0.375). Steps −2.5 (E3 lower on 18/25, **p=0.043**); input tokens +0.17M mean /
  +0.03M median (median per-episode 1.76M vs 1.29M); **wall +0.22 h mean / +0.12 h median,
  E3 higher on 19/25, p=0.015** (per-step 71 s vs 50 s as contexts grow; longest episode
  5.4 h). Exposure 17.1 anchored observations/episode (482 grep + 518 view), ≈38k addendum
  chars/episode, 15% trimmed by the 10k harness limit, 11.9 s LS work/episode, 0 errors.
  Uptake 23% of flagged source files opened; 188 flagged gold files → 178 opened, 176
  patched. Coverage: both 121, E3-only 3, A8-only 4, **neither 64; only 9 of those 64 ever
  named in any E3 addendum**. Consistency: token spread 1.49× vs 1.71× (12/25), flips 4 vs 4.
  Report §3.3 written; results + telemetry added to the experiments entry; pushed.
- Variance reruns: E-var round 1 running on worker-a (rounds 2–5 chained); A8-var
  rounds 1–5 chained on worker-b (started after E3-r2b).
- Variance rerun progress: E-var r1 1/5, r2 1/5 (worker-a); A8-var r1 2/5 (worker-b).
  A8-var-r2 transformers-38332 ended in `BadRequestError` — context reached 247k input
  tokens + 16k completion > 262,144 limit (the model's native context; DSpark disables
  YaRN). Legitimate long-excursion failure, kept as a failed episode with its full
  token/step counts in the variance analysis (same instance as A8-r2's 12.8M-token failing
  excursion). Prior occurrences: C2-r1 ×1, hosted A-r1 ×1.
- Variance rerun: E-var r3 0/5, r4 1/5 (transformers-38332 hit the 262k context limit →
  `BadRequestError`, as A8-var-r2 did on the same instance — both arms produce runaway
  excursions there); A8-var r3 0/5. Round 5 of each in progress.

## 2026-09-06 — VARIANCE RERUN COMPLETE; experiment closed

- E-var r1..5: 1,1,0,1,1 / 5; A8-var r1..5: 2,0,0,0,2 / 5 (hard instances by design).
  7 episodes per instance per arm. **Level equal** (geo-mean tokens 4.02M vs 4.19M; wall
  1.55 vs 1.42 h; resolved 6/35 vs 5/35). **Tails favour E**: control hit the 300-step cap
  in 3 episodes (albumentations, transformers, django) vs 0 for E; token p90 9.3M vs 17.7M,
  max 28.8M vs 36.4M; sd of log tokens 0.74 vs 1.03. **But not significant**: pooled
  |log x − median| rank-sum p=0.71; bootstrap CI on the spread ratio [0.42, 1.57] (point
  0.82); per instance only albumentations-2495 tightens (p=0.13), transformers and
  gallery-dl are more dispersed with anchors; langchain is consistently costlier with
  anchors (1.28×, p=0.04). Wall tail NOT shorter (p90 4.9 h vs 3.7 h). Verdict on Ian's
  three outcomes: correctness no; efficiency no; variance plausible-unproven.
- Report §3.4 written; results (10 var runs, slim JSON) + `anchor_variance.py` +
  `variance_rerun.txt` added to the experiments entry; pushed. Monitors stopped. Both
  workers idle; SGLang still serving on spark.
