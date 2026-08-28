# SWE-Bench ProMax × Qwen3.8-27B × LSP tools — experiment plan

**Question:** Does giving a local open-weight coding agent (Qwen3.8-27B on this GB10) LSP tools —
Pyrefly for Python, state-of-the-art servers for the other languages — improve resolve rate on a
refactoring benchmark, and does *instructing* the agent to prefer those tools help further?

As of 2026-08-22 no published ablation of LSP tools with a local open-weight model on a
SWE-bench-style benchmark exists — this experiment is novel. Closest prior work
(arXiv:2608.13568, frontier models only) found LSP tools are *usually negative* for
localization but help on reference-completeness and multi-file rename tasks — i.e. exactly
the task family of this benchmark — and only when tools return **code content, not bare
file:line locations**. That finding drives the tool design below.

---

## 1. Ground truth

### Benchmark: SWE-Bench ProMax
- Paper: *SWE-Bench ProMax: Benchmarking Agents on Large-Scale Multilingual Code Refactoring*,
  arXiv [2608.09802](https://arxiv.org/abs/2608.09802) (COLM 2026, released 2026-08-10).
  Local copy: `paper/swe-bench-promax-2608.09802.pdf`.
- Not from the Princeton SWE-bench team (SJTU + ByteDance et al.). Two other candidates were
  ruled out: SWE-Refactor (arXiv 2602.03712, Java-only, non-agentic) and the older
  RefactorBench (ICLR 2025, Python-only).
- **170 instances, 70 repos, 7 languages**: Python 29, TypeScript 28, Java 26, Go 23, C++ 22,
  Rust 22, C 20. Gold patches avg 11.4 files / 262 LOC (max 182 files). Big, multi-file
  refactors — the dominant failure mode in the paper is *incomplete refactoring* (too few
  files modified), which is precisely what find-references should help with.
- Harness: [github.com/key4127/SWE-Bench-ProMax](https://github.com/key4127/SWE-Bench-ProMax);
  dataset `swe-bench-promax/SWE-Bench-ProMax` on HF. Bring-your-own rollout → feed
  `[{"instance_id", "model_patch"}]` to `src/evaluation/test_run.py` (stdlib-only, needs
  Python ≥3.10; we have 3.12 ✓). Test-based resolve, 40-min eval timeout, 2 flaky retries,
  `--workers N`, `--cleanup` deletes images after eval.
- Docker images: one per instance, `key4127/refactor-dockerhub:<instance_id>`,
  **linux/amd64 only**, 170 tags ≈ **317 GB compressed** (avg 1.9 GB, max 10.6 GB).
- Paper protocol: mini-swe-agent and OpenHands scaffolds, **300 steps / $10 per instance**,
  temperature unspecified (flag — we document our own). Metric: resolve rate (pass@1).
- Reference numbers (overall resolve %): OpenHands — GPT-5.2 41.2, Claude Sonnet 4.6 38.8,
  GLM-5 36.5, **Qwen3.5 (MoE flagship) 36.5**, Kimi-K2.5 32.9. mini-swe-agent — Sonnet 4.6
  30.6, **Qwen3.5 20.6**, GPT-5.2 21.8. No mid-size open models were evaluated; a dense 27B
  on mini-swe-agent plausibly lands ~8–18%.

### Model: Qwen3.8-27B (confirmed real — released 2026-08-13/14)
- `Qwen/Qwen3.8-27B` + official `Qwen/Qwen3.8-27B-FP8` (~29 GB). Apache 2.0. Dense 27B,
  hybrid 48× Gated DeltaNet + 16× gated full-attention (GQA 4 KV heads × 256 dim), native
  multimodal, 262k context (1M via YaRN), MTP draft head for speculative decoding.
- **KV cache is tiny**: only the 16 full-attention layers accumulate KV → 64 KiB/token BF16
  (32 KiB fp8) → a 128k-token episode costs ~4–8 GB. Excellent fit for many concurrent
  agent episodes on unified memory.
- Hybrid thinking, **on by default at `reasoning_effort: xhigh`, which badly overthinks**
  (documented 22k reasoning tokens on trivial tasks). We run `medium` and document it.
- Sampling (model card, thinking mode): temp 1.0, top_p 0.95, top_k 20, min_p 0.
- Tool-calling supported (`--tool-call-parser qwen3_coder --reasoning-parser qwen3`) — but
  our chosen scaffold is text-only, so parser fidelity is not load-bearing (see D1).
- Serving: needs very recent stacks — vLLM nightly (verified on 0.26.1rc1.dev nightlies),
  SGLang ≥ ~0.5.4 (day-0 support, LMSYS blog 2026-08-12), llama.cpp
  (`ggml-org/Qwen3.8-27B-GGUF`, `--spec-type draft-mtp` ≈ +72% decode).
- **No official SWE-bench Verified score exists for the 27B** (vendor reports SWE-bench Pro
  61.7). All numbers vendor-reported; set expectations accordingly.
- The `Qwen3.5-27B` in the local HF cache (52 GB) is the *previous generation* — not this
  model. Fresh ~29 GB download required.

### Machines
- **spark (this box)** — NVIDIA **GB10 (DGX Spark class)**: aarch64, 20 ARM cores (10× X925 +
  10× A725), **119 GiB unified memory**, ~273 GB/s bandwidth, CUDA 13.0, driver 580.159,
  Docker 29.2.1. Disk: **111 GB free** of 916 GB (88% full). Role: model server only.
- **Lab x86 hosts** (passwordless SSH from spark, all x86_64, all mount the NAS):

  | Host | CPU / threads | RAM | Free disk | Docker |
  |------|---------------|-----|-----------|--------|
  | `cortical` | Ryzen AI MAX+ 395 / 32 | 122 GiB | 409 GB | **none — install in Phase 0** (⚠ ~34 GB RAM in use by something; check before saturating) |
  | `chunklebox` | Ryzen 7 8845HS / 16 | 29 GiB | 147 GB | 29.1.3 |
  | `leejr` | Ryzen 7 3700X / 16 | 125 GiB | 80 GB (91% full) | 29.6.1 |

- **NAS**: `192.168.1.37:/Public` mounted at `/mnt/nas` on spark and all three hosts —
  **6.8 TB free**. Use for: docker image tar cache (`docker save`/`load`, pulled once,
  shared across hosts and batches), run archives/trajectories, dataset artifacts, model
  checkpoint backups. Not for `/var/lib/docker` itself — overlay2 on NFS is unreliable;
  docker data-root stays on each host's local NVMe.
- **DSv4 is not currently running** (no process, no service, GPU idle). It's the custom
  engine at `~/Projects/dsv4/ds4` (`ds4-server` binary + 87 GB GGUF). Pre-flight before
  serving: `pgrep -af 'ds4'` → if found, `kill -INT <pid>` (it handles SIGINT cleanly),
  verify with `free -h` that memory is released.

---

## 2. Requirements mismatches (flagged)

| # | Severity | Mismatch | Mitigation |
|---|----------|----------|------------|
| M1 | ~~Critical~~ **solved by lab** | All 170 instance images are **amd64-only**; the GB10 is **aarch64**. Rollout *and* eval run inside these containers. | Run containers natively on the lab x86 hosts (`cortical` primary) with the GB10 serving the model over the LAN — see D3. No emulation, no cloud. Residual task: install Docker on cortical. |
| M2 | ~~High~~ **mostly solved** | 317 GB compressed images (likely 600–900 GB uncompressed) vs 111 GB free on spark; plus 29 GB model + ~15 GB serving stack. | Spark only needs the model + serving stack now (fits easily). Images live on the x86 hosts: cortical's 409 GB holds large batches; NAS tar cache (`/mnt/nas`, 6.8 TB free) means each image is pulled from Docker Hub once ever. Still batch + `docker rmi` on chunklebox/leejr (147/80 GB free). Spark cleanup candidates (*for Ian to move to NAS, not doing this unilaterally*): `~/models/ttblt_v3` 285 GB, dsv4 GGUFs 87 GB. |
| M3 | **High** | Wall-clock: ~273 GB/s bandwidth → est. 15–25 tok/s single-stream decode (FP8). Thinking tokens dominate. | fp8 weights + fp8 KV, `reasoning_effort: medium`, 4–8 concurrent episodes (batched aggregate est. 60–120 tok/s), MTP speculative decoding if the stack supports it. Estimates in §6; measured in Phase 1 before committing to full runs. |
| M4 | Medium | Model is 8 days old; NVIDIA's DGX-Spark vLLM/SGLang container images may not support the hybrid GDN architecture yet on aarch64/CUDA 13. | Try in order: (1) SGLang recent build (day-0 Qwen3.8 support), (2) vLLM nightly, (3) llama.cpp GGUF (proven on Spark; sufficient because the scaffold is text-only). Phase 0 task with a hard timebox. |
| M5 | Medium | Paper baselines exist only for frontier/large-MoE models; and paper omits sampling temperature. | Compare against Qwen3.5-MoE mini-swe-agent 20.6% as an upper anchor; document our sampling (model-card defaults) and reasoning effort as protocol deviations. |
| M6 | Medium | Statistical power: Python-only is 29 instances; a 3–5 pt arm difference won't clear noise. | Paired per-instance design + McNemar's test; 2 seeds per arm on the Python stage; treat stage 1 as directional signal, full-170 for headline numbers. |
| M7 | Low | Dataset/harness license unclear (README badge "Research", no LICENSE file, none on HF). | Fine for internal experimentation; resolve with authors (xiaodong.gu@sjtu.edu.cn) before publishing results. |
| M8 | Low | LSP quality is uneven across the 7 languages: clangd needs `compile_commands.json` (C/C++ ≈ 42 of 170 instances); jdtls needs project import + slow first index. Pyrefly has open issues on find-references in very large repos (#2039) and memory spikes (#2970). | Per-language expectations in the analysis; pre-warm indexes at episode start; generate `compile_commands.json` via `bear`/CMake export where the repo allows; basedpyright as the Python fallback/ablation if Pyrefly misbehaves. |

---

## 3. Design decisions

**D1 — Scaffold: mini-swe-agent for all arms.** The paper reports it (so the baseline is
anchorable), it is bash-only text (no function-calling parser fragility with a week-old
model), and it makes the LSP intervention clean: the tool arrives as a **CLI inside the
container**, and the arms differ only in environment + prompt. OpenHands scored higher in
the paper but has *no* LSP runtime support (SDK issue #1745, closed not-planned) and a much
heavier integration surface — keep as a possible extension (Arm D).

**D2 — Serving.** *(Settled in Phase 0 — the pip routes all fail on aarch64/sm121: vLLM has
no arm wheels; sgl-kernel's sm100 cubins lack kernels for sm_121; triton's bundled ptxas
predates sm_121a.)* We serve with the GB10-tuned container setup vendored at
`serving/spark-sglang/` (MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark; image
`lmsysorg/sglang:qwen38-27b`): `QUANT=fp8` (official `Qwen/Qwen3.8-27B-FP8` checkpoint),
`./start-dspark.sh` — DSpark speculative decoding (best for code), flashinfer attention,
fp8 KV (~33 KB/token), tuned GDN state pool, CPU pinning to the Cortex-X925 cores, endpoint
`http://192.168.1.92:8888/v1`, model `qwen3.8-27b-sglang`. Measured ~30 tok/s single-stream
on code. Thinking mode with `reasoning_effort: medium` via `chat_template_kwargs`;
model-card sampling (T=1.0, top_p 0.95, top_k 20). Fallback: llama.cpp + ggml-org GGUF
Q8_0 + MTP draft (downloaded to NAS). Note: DSpark caps context at the native 262144 (no
YaRN) — fine, episodes are capped below that anyway.

**D3 — Topology: GB10 serves, lab x86 hosts run the containers.**
- **spark (GB10)**: model server only — dedicated to inference, its best role. OpenAI-
  compatible endpoint on the LAN (bind LAN interface or SSH-tunnel from workers).
- **cortical** (32 threads, 122 GiB, 409 GB free): primary worker — runs mini-swe-agent
  rollouts inside the official amd64 instance images, and the eval harness. Needs Docker
  installed (Phase 0), and a check on what's currently using ~34 GB RAM there.
- **chunklebox / leejr**: optional extra eval workers to parallelize the CPU-bound eval
  phase (≤40 min/instance) and rollout env commands. chunklebox is RAM-light (29 GiB → cap
  concurrent C++/Rust evals); leejr is disk-tight (80 GB → small batches, aggressive `rmi`).
- **NAS** (`/mnt/nas`, shared by all): `docker save` tar cache so each of the 170 images is
  pulled from Docker Hub exactly once; run archives (trajectories, patches, logs) written
  here so any machine can analyze them.
- Generation throughput on spark is the global bottleneck (see §6), so one rollout host is
  enough; sharding across hosts mainly accelerates the eval phase.
- **All-local fallback** (if the lab hosts are ever unavailable): rebuild arm64 images for
  the Python subset via the repo's SWE-Factory pipeline. Rebuilt envs aren't paper-
  comparable, but the A/B/C ablation stays internally valid — the experiment needs
  consistency, not fidelity. Kept as a footnote, not the plan.

**D4 — LSP tool delivery: a single `lsp` CLI in the container.**
A small Python package (`lsp-tool/`) vendoring **solidlsp** (Serena's language-server layer —
already integrates Pyrefly, basedpyright, gopls, rust-analyzer, typescript-language-server,
clangd, jdtls, and 70+ others) with:
- a **daemon** per episode (warm index, Unix socket) so per-call latency is ms, not re-index;
  pre-warm during episode setup, before the agent's clock starts;
- subcommands: `lsp def <file:line:col | symbol>`, `lsp refs …`, `lsp hover …`,
  `lsp symbols <file|query>`, `lsp rename <target> <newname> --dry-run`, `lsp diag <file>`;
- **content-enriched output** (each result = location + the surrounding code block), the
  design arXiv:2608.13568 found necessary — bare file:line lists failed 75% of rename tasks;
- language auto-selected per instance from the dataset's `language` field; Python server
  configurable `pyrefly | basedpyright` (our headline server is **Pyrefly ≥1.2.0**,
  `pip install pyrefly`, spawned as `pyrefly lsp`).
Injection: an image layer (`FROM key4127/refactor-dockerhub:<id>` + copy wheel + amd64
server binaries), built once per instance on the worker host, cached to the NAS.
Serena's full MCP server is deliberately *not* used in-band (~24k tokens of tool schemas is
hostile to a 27B; MCP requires function calling); we reuse only its LS layer.

**D5 — Instance sets.**
- `dev-10`: 10 stratified instances (2 Py / 2 TS / 1 each Java, Go, C++, Rust, C + 1 extra Py)
  for debugging; excluded from headline aggregates.
- `stage1-python`: all 29 Python instances × 3 arms × 2 seeds.
- `stage2-full`: all 170 × 3 arms × 1 seed (paper protocol).

**D6 — Metrics.** Primary: resolve rate per arm (paired per-instance matrix, McNemar's test
between arms). Secondary: steps, output/thinking tokens, wall-clock per episode, `lsp` call
counts and mix (B vs C tells us whether the *instruction* or the *availability* moves usage),
files-modified vs gold-patch files (the incomplete-refactoring proxy), and per-language
breakdown.

---

## 4. Arms

| Arm | Environment | Prompt |
|-----|-------------|--------|
| **A — baseline** | Paper-faithful mini-swe-agent, 300-step cap | Stock template |
| **B — LSP available** | A + `lsp` CLI on PATH + daemon pre-warmed | Stock template + neutral tool documentation (what each subcommand does, no preference language) |
| **C — LSP preferred** | Identical to B | B + preference instruction (draft below) |

Draft Arm C instruction (to be tuned on `dev-10` only):

> When you need to understand the codebase — finding where a symbol is defined, every place
> it is used, or what will break if it changes — prefer the `lsp` tool over grep or reading
> files: `lsp refs` is exhaustive where grep misses dynamic references and aliased imports.
> Before finalizing a refactor, run `lsp refs` on each renamed/moved symbol and `lsp diag`
> on each edited file to confirm nothing was left behind.

Held constant across arms: model, sampling, reasoning effort, step cap, context cap, harness
version, container images, eval command. Only environment/prompt vary.

---

## 5. Phases

**Phase 0 — Infrastructure gates (~1–2 days)**
1. Pre-flight: confirm DSv4 down (`pgrep -af ds4`); clear ~10 GB (docker prune reclaims 4 GB).
2. Clone harness + pull dataset; verify 170 instances parse; commit a repo skeleton
   (`serving/`, `agent/`, `lsp-tool/`, `runs/`, `analysis/`).
3. **Serving gate:** stand up Qwen3.8-27B-FP8 (SGLang → vLLM nightly → llama.cpp, timebox
   ~half day each). Success = OpenAI-compatible chat completion with correct chat template,
   thinking content separated, and a measured tok/s (single + 8-way concurrent).
4. **Worker gate:** install Docker on cortical (and identify what's using ~34 GB RAM there);
   from cortical, hit the spark endpoint (latency + a 3k-token completion); pull one Python
   instance image, run its eval script with the **gold patch**, confirm it resolves well
   inside the 40-min timeout; `docker save` the image to `/mnt/nas` and `docker load` it on
   chunklebox to validate the NAS cache flow.
5. Write the batch runner: shard instance list → per-host pull/load → rollout (against spark)
   → eval → archive to NAS → `docker rmi`.

**Phase 1 — Harness bring-up (~1 day)**
1. mini-swe-agent (`uv tool install mini-swe-agent`) → point LiteLLM at the local endpoint;
   adapter script mapping ProMax JSON (`image_name`, `problem_statement`, `working_dir`) to
   mini-swe-agent's SWE-bench runner; 300-step cap; per-episode wall-clock cap (2 h).
2. Run `dev-10` end-to-end: rollout → patches JSON → `test_run.py` → report. Sanity checks:
   gold patches evaluate as resolved on the same instances; at least one agent patch applies.
3. Record measured tokens/episode and tok/s → recompute §6 budget, adjust concurrency/effort.

**Phase 2 — Arm A baseline**
`stage1-python` (×2 seeds), then `stage2-full` once the batch runner is proven. Anchor check: are we in a
plausible band relative to Qwen3.5-MoE's 20.6% mini-swe-agent number? If <3% overall,
investigate harness bugs before blaming the model (patch-extraction and context-truncation
bugs look identical to model failure).

**Phase 3 — LSP tool build (~2–3 days, parallel with Phase 2 runs)**
Build `lsp-tool` + daemon; per-language smoke tests inside real instance containers
(def/refs/hover/rename on a known symbol; assert content-enriched output); measure index
warm-up time per language (jdtls and rust-analyzer will be the slow ones); bake image layers.

**Phase 4 — Arms B and C**
Same instance sets and seeds as Arm A. Interleave arm order across instances to avoid
serving-drift confounds. Archive full trajectories (they're the analysis substrate).

**Phase 5 — Analysis & go/no-go**
Paired tables, McNemar A↔B and B↔C, per-language splits, tool-usage vs outcome, incomplete-
refactoring proxy, cost (tokens/steps/wall-clock) per resolved. Write up. "Promise" =
B or C beats A by a margin that survives pairing on stage 1 → proceed to stage 2 full run
and extensions.

**Extensions if promising (Arm D+):** Pyrefly ↔ basedpyright swap (server ablation, Python);
content-enriched ↔ location-only output (replicate 2608.13568 on an open model); automatic
`lsp diag` feedback after each edit (OpenCode-style, no agent choice involved); OpenHands +
function-calling tools variant; `xhigh` vs `medium` reasoning-effort interaction on a subset.

---

## 6. Runtime budget (verify in Phase 1)

Assumptions: medium effort ≈ 20–60k generated tokens/episode (40–120 steps), FP8 decode
15–25 tok/s single-stream, 60–120 tok/s aggregate at 6–8 concurrent, prefix caching on
(incremental prefill only).

| Run | Episodes | Est. generation wall-clock |
|-----|----------|---------------------------|
| dev-10 shakeout | 10 | 2–5 h |
| stage1-python, 3 arms × 2 seeds | 174 | ~1.5–3 days |
| stage2-full, 3 arms × 1 seed | 510 | ~4–10 days |

Plus eval: ≤40 min/instance, CPU-bound, parallelizable across cortical/chunklebox/leejr —
hours, not days. Env commands run natively on x86, so rollout overhead is negligible.
Practical reading: **the Python stage is a weekend; the full 3-arm × 170 study is roughly a
week, bottlenecked almost entirely by generation throughput on spark.** Budget levers, in
order: reasoning effort, concurrency, MTP speculative decoding, seeds.

---

## 7. Risks

- **Serving stack fails on aarch64/CUDA 13 for a week-old architecture** — llama.cpp is the
  proven-on-Spark fallback; text-only scaffold means we lose nothing but speed knobs.
- **Environment drift across worker hosts** (different kernel/docker versions could flip a
  flaky test) — validate gold patches per batch on the same host that evaluates the agent
  patches; keep each instance's rollout and eval on one host.
- **cortical has an unknown resident workload** (~34 GB RAM in use) — identify it before
  scheduling heavy eval batches; chunklebox's 29 GiB RAM caps concurrent C++/Rust evals.
- **27B baseline lands near 0%** on these large refactors → no headroom to measure an LSP
  effect. Detect early via dev-10/stage-1; response: relax to a easier slice (paper's
  per-language table shows C and Python are the most tractable), or measure graded proxies
  (files-correctly-modified overlap) rather than binary resolve only.
- **LSP daemons misbehave in minimal containers** (missing glibc bits, jdtls needing a JDK
  the image lacks) — smoke-test per language in Phase 3 before committing to runs.
- **Disk creep on the smaller workers** — every batch script ends with `docker rmi` +
  `docker system prune -f`; alert threshold at <30 GB free per host (leejr starts at 80 GB).

## 8. References

- ProMax: arXiv 2608.09802 · [harness](https://github.com/key4127/SWE-Bench-ProMax) ·
  [HF dataset](https://huggingface.co/datasets/swe-bench-promax/SWE-Bench-ProMax) ·
  images `key4127/refactor-dockerhub:<instance_id>`
- Model: [Qwen/Qwen3.8-27B-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) ·
  [vLLM recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-27B) ·
  [SGLang cookbook](https://lmsysorg.mintlify.app/cookbook/autoregressive/Qwen/Qwen3.8-27B) ·
  [ggml-org GGUF](https://huggingface.co/ggml-org/Qwen3.8-27B-GGUF)
- LSP: [Pyrefly](https://github.com/facebook/pyrefly) (≥1.2.0) ·
  [Serena/solidlsp](https://github.com/oraios/serena) ·
  [mcp-language-server](https://github.com/isaacphi/mcp-language-server) (minimal alternative)
- Evidence: arXiv 2608.13568 (LSP token-cost ablation) · 2306.10763 (Monitor-Guided
  Decoding) · RepoGraph (ICLR 2025) · 2503.09089 (LocAgent) · 2606.22417 (Code Isn't Memory)
- Scaffold: [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent)
