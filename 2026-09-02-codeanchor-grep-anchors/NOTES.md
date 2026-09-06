# Protocol amendments & run ledger — arm E / E3

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
| s1-arm-e3-r1 | E3 | worker-a | running (launched 2026-09-04) |
| s1-arm-e3-r2 | E3 | worker-b | running (launched 2026-09-04) |

## 2026-09-05/06 — Variance rerun (Ian: "the paper signals the saving is largely variance")

5 extra rounds of E and of A8 on the five largest-|Δ tokens| instances (`var5.re`):
`s1-arm-e-var-r1..5` (worker-a), `s1-arm-a8-var-r1..5` (worker-b). All 50 episodes ran;
one context-limit (262k) `BadRequestError` per arm on transformers-38332, kept as failed
episodes with their full counts. Analysis: `code/analysis/anchor_variance.py` →
`results/variance_rerun.txt`; report §3.4.
