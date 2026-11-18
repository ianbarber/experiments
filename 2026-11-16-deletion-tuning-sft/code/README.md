# Code

Paths are relative to this directory. Run on `worker-b` (one RTX 5090, 32 GB). Full
fine-tuning with fp32 AdamW peaks at 26.9 GiB, so it needs the card to itself. The whole
entry used about 15 GPU-hours.

## Environment

```bash
uv sync                      # versions are pinned in pyproject.toml; torch comes from the cu130 index
export PYTHONPATH=$PWD
uv run pytest -q             # 11 CPU tests: simulator, label mask, mask parameterisation
```

IFEval needs NLTK's `punkt_tab`: unpack it under `data/nltk_data/tokenizers/` (the code points
`NLTK_DATA` there). HumanEval executes generated code; `deltune/suite.py` sets
`HF_ALLOW_CODE_EVAL=1`.

Pinned inputs: model `LiquidAI/LFM3-1.7B` at revision `1da06bec`, dataset
`openenv/wordle-tool-calls-sft` at revision `f4b1006e`.

## Layout

| Path | What it is |
|---|---|
| `deltune/wordle.py` | Local Wordle simulator mirroring TextArena `Wordle-v0`, and the batched player |
| `deltune/data.py` | The 1,000 games as assistant-only SFT examples; fixed 950/50 split |
| `deltune/methods.py` | Deletion Tuning (mask layer, Wanda init, regulariser, packed mask) and LoRA setup |
| `deltune/train.py` | One training run; saves deployable checkpoints only |
| `deltune/evaluate.py` | Scores a checkpoint as a deployed bf16 model: Wordle pools and the suite |
| `deltune/suite.py` | The six-benchmark suite through lm-evaluation-harness |
| `wordlists.json` | Seen-list (156), held-out validation (300) and test (500) secrets, prompt, tool schema |
| `scripts/build_wordlists.py` | Rebuilds `wordlists.json` |
| `scripts/gates.py` | Implementation gates on the real model |
| `scripts/base_sensitivity.py` | Base model at Wordle under other decoding settings |
| `scripts/sweep.py` | Stages: base, tuning runs, final seeds |
| `scripts/diagnostic_lr.sh` | Seed 42 at neighbouring learning rates, the untrained-mask control, the HumanEval rerun |
| `scripts/followup.sh` | The paper's learning rates on three seeds, Deletion Tuning at the paper's 400-step recipe, and the base suite scored a second time |
| `scripts/analyze.py` | Selection, matched steps, intervals over seeds and over items; writes `../results/*.json` |
| `scripts/ifeval_behaviour.py`, `scripts/humaneval_reading.py` | What the generated answers look like; the first also writes the per-answer flags the analysis reads |
| `scripts/cost.py` | Time per step, peak memory and trained state per method, from the training logs |
| `scripts/mask_anatomy.py` | What the trained masks contain: inherited against learned, where and when, agreement between seeds; writes the ablation masks |
| `scripts/mask_ablation.sh` | Scores those parts of masks on seen-list Wordle |
| `scripts/figures.py` | Writes `../images/*.png` |
| `scripts/replay.py` | Prints full game transcripts for any checkpoint |
| `scripts/export_evidence.py` | Packs per-item scores and game outcomes into `../results/evidence.tar.gz` |

## Re-running the analysis without a GPU

```bash
mkdir -p runs && tar -xzf ../results/evidence.tar.gz -C runs
python scripts/analyze.py grid && python scripts/analyze.py final && python scripts/analyze.py lr
python scripts/figures.py all
```

`cost.json` and `mask_anatomy.json` are not rebuilt this way: they read the training logs and
the saved masks, which are not in the archive. `python scripts/mask_anatomy.py ablation`
refreshes the ablation scores from it.

## Running everything

```bash
python scripts/gates.py dt && python scripts/gates.py lora
python scripts/sweep.py base                         # base model: Wordle pools and the suite
python scripts/sweep.py grid                         # first-pass tuning runs on seed 42
python scripts/sweep.py one --method fft --lr 1e-6   # extensions; see LABNOTES for the list
python scripts/analyze.py grid                       # selection and matched steps
python scripts/sweep.py final --dt dt_lr1e-3_a100-1000_s42:400 \
    --lora lora_r4_lr3e-4_s42:1000 --fft fft_lr3e-5_s42:1000 --dt-extra-save 300
bash scripts/diagnostic_lr.sh
bash scripts/followup.sh base seed43 seed44 seed42 paper
python scripts/ifeval_behaviour.py && python scripts/humaneval_reading.py
python scripts/analyze.py final && python scripts/analyze.py lr
python scripts/cost.py                               # needs the training logs in runs/
python scripts/mask_anatomy.py --ablations           # needs the saved masks and the GPU, about 3 minutes
bash scripts/mask_ablation.sh
python scripts/figures.py all
python scripts/export_evidence.py
```

Run directories go to `runs/` (ignored by git). Full fine-tuning writes a 3.5 GB bf16
checkpoint per saved step; LoRA adapters are 16 MB and deletion masks about 6 MB.

Three things that are not in the scripts as written. The LoRA lr 1e-4 seed-42 run was the
pilot: it ran through an earlier in-process evaluation path and was rescored afterwards, so
it has checkpoints every 400 steps, not every 200. The two `diag_fft_*` runs are retrained
copies of tuning runs whose checkpoints had been deleted, stopped at step 1,200. And
`rerun_fft_lr3e-6_s42` is the same configuration trained once more to step 1,600. Training
is not bit-reproducible: the three copies of that run differ by a few points of Wordle
score.

One job at a time on the GPU. A second process during training or evaluation will run it out
of memory.
