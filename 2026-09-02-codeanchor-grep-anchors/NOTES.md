# Protocol amendments & run ledger — arm E / E3 / variance rerun

Base protocol (model, sampling, harness, images, eval) is that of the
[2026-08-22 study](../2026-08-22-lsp-agents-promax/NOTES.md).

## 2026-09-02 — Arm E (CodeAnchor-style anchors) + A8 control: protocol

- **Arm E** (`agent/arm_e.yaml`): prompt byte-identical to arm A. Environment class
  `anchor_env.AnchorDockerEnvironment` (`agent/anchor_env.py`, on PYTHONPATH via
  `run_batch_waved.sh`) appends a language-server addendum to every observation of a
  grep-family command that shows `path:line:` hits in `.py` files: per matched symbol,
  kind + qualified name + definition site, references grouped by file with the enclosing
  scope of each use (tags `[import]`/`[subclass]`/`[override]`), and "referenced but NOT
  in this grep output" files. Rollout images `promax-lsp:<id>` (Pyrefly daemon warmed at
  container start; `lsp_tool` package hot-patched in via `docker cp` — no image rebuild);
  eval in the original images as always. Caps: 40 hits, 4 symbols, 2800 chars, 12 s
  budget per grep; LS not ready → addendum skipped (logged), never blocks.
- **Arm A8** (`agent/arm_a8.yaml`): arm A with `container_timeout: 8h` — the same-period
  paired control (both arms use the 8 h wall, removing the 2 h-wall artifact of stage 1).
- Telemetry: `runs/<run>/anchor_log.jsonl` (one row per grep-family command: status,
  hits, symbols, addendum chars, seconds). Primary comparison E vs A8, paired by instance
  across 2 rounds, hosts swapped; secondary metrics: steps, cumulative input tokens,
  output/reasoning tokens, wall per episode, grep calls, edit recall vs gold files.

| Run | Arm | Host | Status |
|-----|-----|------|--------|
| s1-arm-e-r1 | E anchors | worker-a | **DONE: 16/25 (64%)**; launched 4× before any counted episode (see LABBOOK 09-02/03); aborted partials `s1-arm-e-r1-aborted-v{1,2,3}` are not results |
| s1-arm-a8-r1 | A8 control | worker-b | **DONE: 18/25 (72%)** |
| s1-arm-e-r2 | E round 2 | worker-b | **DONE: 16/25 (64%)** |
| s1-arm-a8-r2 | A8 round 2 | worker-a | **DONE: 16/25 (64%)** |

**STAGE E FINAL: E 32/50 (64%) vs A8 34/50 (68%); paired 2-vs-3, p=1.0; Δsteps −4.3 (n.s.), Δtokens ≈0, Δwall ≈0. Protocol deviation from CodeAnchor: addendum caps (4 symbols / 6 not-shown files / 2,800 chars) — measured by uncapped replay to hide 3/69 missed gold files (4%).**

## 2026-09-04 — Arm E3 (definition-site anchors, uncapped): protocol

- `agent/arm_e3.yaml`: as E, plus `anchor_views: true` (def/class lines visible in file
  views get their users — CodeAnchor's definition-site placement) and `anchor_uncapped:
  true` (no symbol/file/user caps; harness 10k observation limit is the only truncation,
  applied to the addendum, logged). Control: A8 r1/r2 (reused).

| Run | Arm | Host | Status |
|-----|-----|------|--------|
| s1-arm-e3-r1 | E3 | worker-a | **DONE: 17/25 (68%)** |
| s1-arm-e3-r2 | E3 | worker-b | **DONE: 15/25 (60%)** (merged: waves 1–5 + `-fix` + `-r2b`) |

**E3 FINAL: 32/50 (64%) vs A8 34/50 (68%); paired 1-vs-4 p=0.375; steps −2.5 (p=0.043), wall +0.22 h (p=0.015), tokens +0.17M mean.**

## 2026-09-05/06 — Variance rerun ledger

| Run | Arm | Host | Result (5 instances) |
|-----|-----|------|----------------------|
| s1-arm-e-var-r1..r5 | E | worker-a | 1, 1, 0, 1, 1 of 5 |
| s1-arm-a8-var-r1..r5 | A8 | worker-b | 2, 0, 0, 0, 2 of 5 |

Pooled 35 episodes/arm: level equal (geo-mean tokens 4.02M vs 4.19M); control has 3
step-cap episodes vs 0; spread reduction n.s. (p=0.71, bootstrap CI 0.42–1.57). §3.4.

## 2026-09-14 — Post-hoc correction: reference-server bug and missed-file necessity audit (no new runs)

- **Bug.** The base images for `huggingface__lerobot-2808` and `huggingface__transformers-38332`
  ship `PYTHONPATH=/testbed:`; both repos use a `src/` layout. Pyrefly 1.2.0 puts PYTHONPATH
  entries on its site-package path ahead of the inferred import root, so an opened file is
  named `src.<pkg>...` while everything else imports `<pkg>...`; find-references walks reverse
  imports of the definition handle and returns only the relative-import neighbourhood of the
  opened file. Seven base images set `PYTHONPATH=/testbed` (the other five are flat layout,
  unaffected); two set `/testbed/src` (unaffected). `results/base_image_env.txt`.
- **Verification.** (i) `code/analysis/anchor_replay_local.py`: every anchored grep of E-r1
  replayed against a clean local server on the repos at base commit — identical to the in-image
  replay on 27/29 instances, lerobot 44 vs 16 files, transformers-38332 390 vs 328
  (`results/replay_uncapped_local_e_r1.jsonl`). (ii) In the actual lerobot image on worker-a:
  `lsp refs` on `OpenCVCameraConfig` = 2 files with the shipped env, full set with
  `env -u PYTHONPATH lsp daemon start`. (iii) Locally, `PYTHONPATH=<repo>:` alone reproduces the
  2-file answer; `PYTHONPATH=<repo>/src:` does not; a flat-layout repo is unaffected either way.
  Image versions identical to local (pyrefly 1.2.0, serena-agent 1.7.0, lsp-tool 0.1.0).
- **Scope.** All E, E3 and variance-rerun episodes of those two instances used truncated
  "used by" lists; the two instances are one paired tie and one E win, so a rerun could only
  move the paired count from 2-vs-3 to 3-vs-3. Not rerun. Any future use of the promax-lsp
  images must start the daemon with `env -u PYTHONPATH` (or `PYTHONPATH=/testbed/src`).
- **Corrected counterfactual** (`code/analysis/anchor_missed_audit.py`,
  `results/missed_audit_2026-09-14.txt`): of the 69 gold files missed by both arms in round 1,
  28 non-Python (41%), 22 never named by any anchor (32%, was 27/39%), 8 named only by an
  uncapped working server (12%, was 3/4%), 4 shown and ignored, 7 created/deleted.
- **Necessity audit.** Over every run on the NAS (all arms/rounds), 65/69 missed files were
  omitted by at least one patch that resolved the instance → not required by the graded tests
  (all 28 non-Python; 21/22 never-named; all 8 reachable; all 7 created/deleted; 1/4 shown).
  Never omitted: albumentations-2495 `augmentations/mixing/{functional,transforms}.py` +
  `core/composition.py` (shown, ignored) and dspy-9047 `dspy/evaluate/evaluate.py` (gold
  introduces `toDict()` calls; nothing to reference). Edit recall vs gold files, the metric
  behind the "incomplete refactor" reading, counts docs, squashed unrelated changes
  (optuna-c_058e8bc, ragas-2333) and untested refactor intent (lerobot examples) alike.
- **Failure modes against test-needed files** (`code/analysis/anchor_failure_modes.py`,
  `results/failure_modes_2026-09-14.txt`; needed = intersection of every resolving patch's
  files, median 4 per instance vs 6 gold files). Recall vs needed files E 0.960 / A8 0.939
  (vs gold files 0.675 / 0.678); episodes that never edited a needed file E 4/50, A8 8/50.
  The 34 failing episodes, labelled by reading the task statement against the failing
  assertion (evidence quotes in the script): localization miss E 4 / A8 5; unmet stated
  requirement (the statement says it, the patch does not do it) E 8 / A8 5; requirement not
  derivable from the statement (name/signature/internal/reading only the tests pin) E 4 /
  A8 4; external knowledge (live third-party JSON) E 2 / A8 2. Same tests fail in both arms
  on the same instances. Naming: the dominant mode is partial compliance with a long,
  explicit specification, not incomplete refactoring.
