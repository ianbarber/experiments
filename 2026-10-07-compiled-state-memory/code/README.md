# code/

Everything needed to reproduce. Paths below are relative to `code/` unless stated; `..` is the entry folder.

## Environment

- Machine: `dgx-spark` (GB10, aarch64, sm_121, 128 GB unified memory). pip vLLM/SGLang do not
  work on this hardware; we build on the GB10-working SGLang image.
- Image: `docker build -t statelm:fla052 docker/` — `lmsysorg/sglang:qwen38-27b` (torch
  2.13.0+cu130, transformers 5.12.1, triton 3.7.1) + `fla-core`/`flash-linear-attention` 0.5.2
  (installed `--no-deps` so the base torch/triton are kept) + `rank_bm25`.
- `run.sh <cmd>` runs a command in the image with the entry folder at `/work` (working directory
  `/work/code`), the HF cache on `nas` (`/mnt/nas/hf-cache`) at `/nas-hf`, `$STATELM_WEIGHTS` at
  `/weights` and a persistent Triton cache. `DOCKER_EXTRA` adds `docker run` arguments.
- `serve_sglang.sh start | start-long | stop` serves the same bf16 27B weights on `127.0.0.1:30000`
  (`lmsysorg/sglang:qwen38-27b`, bf16 KV, fp32 GDN state). `start` is the generation profile
  (8,192-token context, radix cache off): self-study items and room-check digests. `start-long` is
  the 262,144-token profile with the radix cache on: room-check agents and judge. It refuses to
  start with under 100 GiB free.
- `STATELM_WEIGHTS`: the directory holding `Qwen3.8-27B/`, the bf16 snapshot
  `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`; mounted at `/weights` in both containers and read by
  the host-side scripts' tokenizer defaults. `STATELM_SEEDQ`: a local cache for Stage 1 seeds and
  query sets (mounted at `/seedq` by `scripts/run_dev*.sh`).
- `models.sh` gives the in-container model paths: Qwen3.5-4B from the HF cache on `nas`, and
  `Q38_27B=/weights/Qwen3.8-27B`. In committed results, `model` and `data` fields are container
  paths: `/work` is the entry folder, `/weights` is `$STATELM_WEIGHTS`, `/nas-hf` the HF cache on `nas`.
- Changes for publication:
  - I parameterised model and cache paths; the runs used a local cache. `models.sh` was
    `models.env`.
  - I annotated three code paths that no reported result uses: `--fit_seed` in
    `experiments/lme_seed.py` and `--ref` in `analysis/lme_seed_size.py` (never run), and
    `--role assistant` in `experiments/lme_ssf_gen.py` (run only by the dev-run-10 generation, which
    I stopped partway).
  - I left out the scripts of dev runs 9 and 10, which never ran, and of the dev-run-10 generation,
    which I stopped partway. No result uses them.
  - The files registered for G1 and H1 are byte-identical to their registered versions
    (`../PREREG.md`). So `experiments/lme_hint.py` keeps its original `--model` default; pass
    `--model`.
- Comments that cite `LABNOTES <date>` point to the dated headings of
  `../results/audit/LABNOTES_full.md`; `../LABNOTES.md` is a condensed version.

## Data

Not committed. LongMemEval (MIT licence): Hugging Face dataset `xiaowu0162/longmemeval-cleaned`,
revision `98d7416c24c778c2fee6e6f3006e7a073259d48f`, in `../data/raw/longmemeval/`:
`longmemeval_s_cleaned.json` (sha256 starts `d6f21ea9d60a0d56`), `longmemeval_m_cleaned.json`
(sha256 starts `9d79e5524794a2e6`; room-check stores) and `longmemeval_oracle.json` (no script reads
it). The answer-check prompt in `statelm/room.py` is copied from `src/evaluation/evaluate_qa.py` of
github.com/xiaowu0162/LongMemEval at `9e0b455f`. `../data/raw/text/qwen38_model_card.md` (filler
prose for the injection suite, pilots and diagnostics) is the 27B snapshot's `README.md`.

## Layout

| Path | What |
|---|---|
| `statelm/core.py` | `LM` wrapper: compile memory → `CompiledState` (fp32 GDN states); session caches for A/C (+ layer/head masks, scale, conv keep); K (KV-only) cache; scoring |
| `statelm/prompts.py` | compile text (system turn carrying the memory) and session text (user turn, non-thinking assistant prefix), with template checks |
| `statelm/synth.py`, `statelm/synth_prefs.py` | development items: 12 attributes, 12 preferences, DRM lures, capability-guard questions |
| `statelm/synth_v2.py` | sealed G1 items (48 facts, 48 preferences) and the v2 neutral filler used everywhere after G1 |
| `tests/check_injection.py` | injection suite T0–T10 (must pass per model); `tests/check_fla_continuation.py` kernel continuation check |
| `tests/debug_long_continuation*.py` | ad hoc checks behind the long-continuation finding (LABNOTES 2026-10-07); they save no result |
| `diagnostics/` | nominal half-life (CPU), measured retention, throughput, overhead |
| `experiments/pilot_position_sweep.py`, `pilot_p0b_mechanism.py` | Qwen3.5-4B pilots P0 / P0b |
| `experiments/gate_phase1.py` | v1 gate; superseded by `gate_v2.py` before any 27B run (checklist findings in LABNOTES) |
| `experiments/gate_v2.py` | pre-registered gate G1 (PREREG.md) |
| `experiments/dev_diagnostics.py` | 27B position / dilution / scale / band / conv diagnostics (dev items) |
| `experiments/gain_sweep.py`, `head_gain.py` | uniform and head-targeted gain (dev items) |
| `experiments/storage_probe.py`, `capacity_probe.py`, `interference_probe.py` | per-head Grams of compiled states + model readout for linear decoding |
| `experiments/retrieval_pilot.py`, `retrieval_gen.py` | hint-vs-access pilot: first-token likelihoods, greedy first action |
| `statelm/lme.py` | LongMemEval_S stores (present / absent / placebo), keyword index, BM25 ranks, H1 sample and audit |
| `experiments/lme_hint.py` | pre-registered H1 hint check (PREREG.md §H1); `--preflight` (CPU), `--resume` |
| `statelm/kvseed.py` | derived attention seed: k-means centroid keys, fitted log-mass bias, ridge values |
| `tests/check_kvseed.py`, `check_kvseed_fit.py` | seed plumbing S1–S12 (27B); synthetic fit check F1–F4 |
| `experiments/lme_where.py` | Stage 0: where the memory's effect lives (dev16) |
| `analysis/lme_where.py`, `lme_where_posthoc.py` | Stage 0 planned summary; post-hoc analyses |
| `experiments/lme_seed.py` | Stage 1: GDN state + a seed fitted on memory-derived reference queries |
| `experiments/lme_ssf_gen.py` | fact-level self-study items (SGLang, memory-free) |
| `analysis/lme_seed.py`, `lme_seed_keys.py`, `lme_seed_size.py` | Stage 1 summaries: dev runs 1–6; 7 (keys, drift); 8 (size, dose, leak check) |
| `analysis/lme_ssf_recall.py` | generator recall check R1 |
| `statelm/room.py` | room check: nested LongMemEval_M stores, memory tools, answer-check prompt |
| `experiments/room_prep.py` | CPU: calibration split, stores, keyword indexes, BM25 check |
| `experiments/room_digest.py` | arm D's one-line note per memory file |
| `experiments/room_agent.py` | arms N, O, A, T, D, M as agents (SGLang) |
| `experiments/room_judge.py` | LongMemEval's answer check, Qwen3.8-27B as judge |
| `analysis/room_check.py` | readouts and decision; second-judge export and merge |
| `analysis/room_figure.py` | Figure 5 |
| `diagnostics/sglang_long_smoke.py` | long-context profile smoke (~200k prefill, radix reuse) |
| `scripts/run_dev5.sh` … `run_dev8.sh` | Stage 1 dev runs 5–8 |
| `scripts/run_room_digest.sh`, `run_room_stage1.sh`, `run_room_stage2.sh` | room-check stages (each starts and stops the server) |
| `analysis/*.py` (others) | summaries (host-side, `uv run`); `analysis/figures.py` makes Figures 1–4 |

## Run

All HF-path GPU runs are batch 1 on one 27B instance; do not run two at once, or beside the SGLang
server.

```bash
source models.sh
./run.sh python tests/check_injection.py --model $Q38_27B --out /work/results/audit/check_injection_q38-27b.json
./run.sh python diagnostics/gdn_retention.py --model $Q35_4B --out /work/results/gdn_retention_q35-4b.json
./run.sh python experiments/pilot_position_sweep.py --model $Q35_4B --out /work/results/pilot/pos_sweep_q35-4b.jsonl
uv run --with numpy python analysis/pilot_position_sweep.py ../results/pilot/pos_sweep_q35-4b.jsonl
CKPT=<snapshot dir> uv run --with safetensors --with numpy --with torch python diagnostics/gdn_nominal_halflife.py

# G1 (pre-registered; analysis unmodified since registration)
./run.sh python experiments/gate_v2.py --model $Q38_27B --out /work/results/phase1/gate_v2_q38-27b.jsonl
uv run --with numpy --with scipy python analysis/gate_v2.py ../results/phase1/gate_v2_q38-27b.jsonl

# phase-1 exploratory (27B, dev items); each analysis prints the *_summary.txt in results/phase1/
./run.sh python experiments/gain_sweep.py --model $Q38_27B --out /work/results/phase1/gain_sweep_q38-27b.jsonl
./run.sh python experiments/storage_probe.py --model $Q38_27B --out /work/results/phase1/storage_probe_q38-27b.jsonl
uv run --with numpy --with scipy --with torch python analysis/storage_probe.py ../results/phase1/storage_probe_q38-27b.jsonl ../data/probe
uv run --with numpy --with scipy --with torch python analysis/binding_heads.py ../results/phase1/storage_probe_q38-27b.jsonl ../data/probe ../results/phase1/binding_heads_q38-27b.json
./run.sh python experiments/capacity_probe.py --model $Q38_27B --out /work/results/phase1/capacity_probe_q38-27b.jsonl
./run.sh python experiments/interference_probe.py --model $Q38_27B --out /work/results/phase1/interference_probe_q38-27b.jsonl
./run.sh python experiments/interference_probe.py --model $Q38_27B --cells user32,person32,user128,person128 \
    --out /work/results/phase1/interference_subject_q38-27b.jsonl
uv run --with numpy --with scipy --with torch python analysis/capacity_probe.py \
    ../results/phase1/capacity_probe_q38-27b.jsonl ../data/capacity ../results/phase1/storage_probe_q38-27b.jsonl ../data/probe
./run.sh python experiments/head_gain.py --model $Q38_27B --heads /work/results/phase1/binding_heads_q38-27b.json \
    --out /work/results/phase1/head_gain_q38-27b.jsonl
uv run --with numpy --with scipy python analysis/gain_sweep.py ../results/phase1/head_gain_q38-27b.jsonl
./run.sh python experiments/retrieval_pilot.py --model $Q38_27B --out /work/results/phase1/retrieval_pilot_q38-27b.jsonl
./run.sh python experiments/retrieval_gen.py --model $Q38_27B --out /work/results/phase1/retrieval_gen_q38-27b.jsonl
uv run --with numpy --with scipy python analysis/retrieval_pilot.py ../results/phase1/retrieval_pilot_q38-27b.jsonl
uv run --with numpy --with scipy python analysis/retrieval_gen.py ../results/phase1/retrieval_gen_q38-27b.jsonl
./run.sh python experiments/dev_diagnostics.py --model $Q38_27B --out /work/results/phase1/dev_diag_q38-27b.jsonl
uv run --with numpy --with scipy python analysis/dev_diagnostics.py ../results/phase1/dev_diag_q38-27b.jsonl
uv run --with numpy --with scipy --with torch python analysis/storage_heads.py ../results/phase1/storage_probe_q38-27b.jsonl ../data/probe
uv run --with numpy --with scipy python analysis/gate_v2_explore.py ../results/phase1/gate_v2_q38-27b.jsonl
uv run --with numpy --with scipy python analysis/gain_sweep.py ../results/phase1/gain_sweep_q38-27b.jsonl
./run.sh python experiments/pilot_p0b_mechanism.py --model $Q35_4B --out /work/results/pilot/p0b_q35-4b.jsonl
uv run --with numpy python analysis/pilot_p0b.py ../results/pilot/p0b_q35-4b.jsonl
./run.sh python diagnostics/gdn_retention.py --model $Q38_27B --out /work/results/gdn_retention_q38-27b.json
./run.sh python diagnostics/throughput.py --model $Q38_27B --out /work/results/audit/throughput_q38-27b-bf16.json
./run.sh python diagnostics/overhead.py --model $Q38_27B --out /work/results/audit/overhead_q38-27b.json

# H1 (pre-registered; run from the entry folder for the analyses)
./run.sh python experiments/lme_hint.py --preflight          # CPU: build every input, no model
./run.sh python -u experiments/lme_hint.py --model $Q38_27B --out /work/results/phase2/lme_hint_q38-27b.jsonl
cd ..
uv run --with numpy --with scipy python code/analysis/lme_hint.py results/phase2/lme_hint_q38-27b.jsonl \
    data/raw/longmemeval/longmemeval_s_cleaned.json
uv run --with numpy --with scipy python code/analysis/lme_hint_explore.py results/phase2/lme_hint_q38-27b.jsonl
uv run --with numpy --with matplotlib python code/analysis/figures.py
cd code
```

### Stage 0 (dev16; exploratory)

```bash
./run.sh python tests/check_kvseed.py --model $Q38_27B --out /work/results/audit/check_kvseed_q38-27b.json
./run.sh python tests/check_kvseed_fit.py
./run.sh python -u experiments/lme_where.py --model $Q38_27B --out /work/results/phase2/lme_where_q38-27b.jsonl
uv run --with numpy --with scipy --with torch python analysis/lme_where.py ../results/phase2/lme_where_q38-27b.jsonl
uv run --with numpy --with scipy --with torch python analysis/lme_where_posthoc.py ../results/phase2/lme_where_q38-27b.jsonl
```

The attribution parts (`analysis/lme_where.py` §4–4b, `lme_where_posthoc.py` P4) need the
uncommitted per-question dumps (`lme_where_q38-27b.jsonl.d/`) from a rerun.

### Stage 1 (dev; exploratory)

Dev runs 1–4 called `experiments/lme_seed.py` directly (settings in `../results/audit/LABNOTES_full.md`); 5–8 are in
`scripts/`. Self-study items (R1 scores them; runs 6–8 fit on them) cover nine dev16 present stores
and the absent stores of 0862e8bf and 6f9b354f.

```bash
./run.sh python -u experiments/lme_seed.py --model $Q38_27B --qids <dev ids> --out /work/results/phase2/lme_seed_dev<k>.jsonl  # runs 1-4

./serve_sglang.sh start      # then wait for http://127.0.0.1:30000/health
GEN='uv run --with transformers==5.12.1 --with jinja2 python experiments/lme_ssf_gen.py --scope all --out_dir ../results/phase2/ssf'
$GEN --stores p --qids 0862e8bf,0e4e4c46,37d43f65,6b168ec8,6f9b354f,75499fd8,a1eacc2a,c7dc5443,dad224aa
$GEN --stores a --qids 0862e8bf,6f9b354f
./serve_sglang.sh stop
uv run --with transformers==5.12.1 --with jinja2 python analysis/lme_ssf_recall.py ../results/phase2/ssf/*_p_all_*.json

scripts/run_dev5.sh && scripts/run_dev6.sh && scripts/run_dev7.sh && scripts/run_dev8.sh   # in order

uv run --with numpy python analysis/lme_seed.py ../results/phase2/lme_seed_dev6.jsonl   # any of dev runs 1-6
uv run --with numpy python analysis/lme_seed_keys.py ../results/phase2/lme_seed_dev7.jsonl ../results/phase2/lme_seed_dev6.jsonl
uv run --with numpy python analysis/lme_seed_size.py --base ../results/phase2/lme_seed_dev7.jsonl \
    --size ../results/phase2/lme_seed_dev8.jsonl --r16 ../results/phase2/lme_seed_dev8r16.jsonl \
    --dose ../results/phase2/lme_seed_dev8d.jsonl --absent ../results/phase2/lme_seed_dev8a.jsonl \
    --absent_base ../results/phase2/lme_seed_dev6a.jsonl --json ../results/phase2/lme_seed_size_summary.json
```

`lme_seed_size.py` prints `../results/phase2/lme_seed_dev8_analysis.txt`; its `--json` output is
Figure 4's input. `experiments/lme_seed.py` has a `--fit_seed` flag (default 42) that was added
after the last Stage 1 run; every run used 42.

### Room check (calibration questions)

```bash
uv run --with transformers==5.12.1 --with jinja2 python experiments/room_prep.py --out ../results/phase2/room/prep   # CPU
# optional: ./serve_sglang.sh start-long; uv run python diagnostics/sglang_long_smoke.py; ./serve_sglang.sh stop
scripts/run_room_digest.sh
scripts/run_room_stage1.sh smoke    # 2 questions
scripts/run_room_stage1.sh full     # all arms but M@mid/large; flip reruns; Qwen judge
scripts/run_room_stage2.sh          # M@mid/large; exits 3 if the 24 h stop rule fires
R=../results/phase2/room
uv run --with numpy python analysis/room_check.py --traj $R/traj.jsonl --judge $R/judge.jsonl --export_j2 $R/j2/items.json
uv run --with numpy python analysis/room_check.py --traj $R/traj.jsonl --judge $R/judge.jsonl \
    --judge2 $R/j2/labels_claude.json --j2_key $R/j2/key.json    # prints $R/room_check_final.txt
uv run --with numpy --with matplotlib python analysis/room_figure.py   # ../images/room_check.png
uv run --with numpy python analysis/room_hint_gain.py    # ../results/phase2/room/hint_gain.txt
```

The second judge was eight Claude Opus 5.5 agents (`claude-opus-5-5`), each given one file of 60
exported items (id, question type, question, correct answer, response; `items.json` is shuffled with
seed 43 and cut into consecutive files of 60) and LongMemEval's rubric, blind to the arm. Their
instructions are in `experiments/room_judge2_prompt.md`; there is no other script for it. `room_check.py` and `room_figure.py` read `../results/phase2/room/prep/`, so
run `room_prep.py` first; `../results/phase2/room/split.json` is a copy of `prep/split.json`, to
check a rebuild against.

## Results notes

- Larger `.jsonl` result files are committed gzipped (`*.jsonl.gz`): gunzip before running an
  analysis (`find ../results -name '*.jsonl.gz' -exec gunzip -k {} +`).
- Not committed: probe Grams (`../data/probe`, `capacity`, `interference`; ~1.4 GB, written by the
  experiments); Stage 1 seeds and query sets (`$STATELM_SEEDQ`) and Stage 0 dumps
  (`../results/phase2/*.jsonl.d/*.pt`); the raw records of Stage 1 dev runs 1–8 (tens of MB; their
  readouts are in `../results/audit/LABNOTES_full.md`, `lme_seed_dev8_analysis.txt` and `lme_seed_size_summary.json`);
  the room-check `prep/` folder (stores rebuilt by `room_prep.py`, digests by `run_room_digest.sh`); and
  fact-level self-study generations (`../results/phase2/ssf*`).
