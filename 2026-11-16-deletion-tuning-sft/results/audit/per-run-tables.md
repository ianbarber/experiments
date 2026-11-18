# Per-run tables

Tables that were in `LABNOTES.md` as the runs came in, moved here to keep the notes short.
The same numbers are in `results/grid.json`, `results/lr_diagnostic.json`,
`results/gates_dt.json` and `results/gates_lora.json`.

## Implementation gates (2026-11-16)

- Deletion Tuning with nothing deleted reproduces base logits bitwise. Pass.
- Initial mask: 0.184% of gate weights zero (paper: about 0.18%), 0.040% of the model,
  1.31 MB compressed. Three quarters of those initial deletions come from the init noise,
  not from the Wanda score.
- Gradients reach the mask logits only; the regulariser matches its closed form. Twelve
  steps at lr 5e-4 take training loss from 2.54 to 0.24; LoRA at 1e-4 is at 1.39 after the
  same batches.
- LoRA at init reproduces base. Merged versus unmerged: identical in fp32, 0.024 nats apart
  in bf16, because the update is about one bf16 rounding step. The gate as worded holds in
  fp32 and not in bf16, so LoRA is evaluated unmerged and the gate is recorded as
  failed-as-worded.
- Full fine-tuning with fp32 AdamW peaks at 26.9 GiB of the card's 31.4.
- `gates_lora.json` was written before the `g_proj` projections were added to the LoRA
  target list and recorded 142 modules. Every reported LoRA run has 153.

## Tuning runs on seed 42 (2026-11-16)

Seen-list win rate (156 games) at four checkpoints and held-out validation win rate (300
games) at step 1,600. Bold marks the configuration selected per method and its best
checkpoint.

| run | 400 | 800 | 1200 | 1600 | held-out |
|---|---|---|---|---|---|
| FFT lr 1e-6 | 38 | 49 | 72 | 69 | 0.7 |
| FFT lr 3e-6 | 67 | 69 | 79 | 75 | 1.3 |
| FFT lr 1e-5 | 69 | 67 | 77 | 76 | 0.3 |
| **FFT lr 3e-5** | 64 | 70 | **85** | 77 | 0.3 |
| FFT lr 1e-4 | 54 | 67 | 69 | 71 | 0.0 |
| LoRA r4 lr 1e-4 | 44 | 53 | 71 | 77 | 2.3 |
| **LoRA r4 lr 3e-4** | 55 | 62 | **79** | 75 | 2.3 |
| LoRA r4 lr 1e-3 | 62 | 62 | 61 | 61 | 0.3 |
| LoRA r1 lr 3e-4 | 55 | 63 | 76 | 78 | 1.0 |
| DT lr 3e-4 | 24 | 76 | 81 | 72 | 2.7 |
| DT lr 5e-4 | 80 | 85 | 79 | 82 | 2.7 |
| **DT lr 1e-3** | 87 | **88** | 83 | 88 | 2.0 |
| DT lr 3e-3 | 80 | 79 | 77 | 78 | 0.3 |
| DT lr 1e-3, alpha 1e3 to 1e4 | 10 | 14 | 10 | 10 | 0.7 |

## First forgetting numbers, seed 42 at the matched checkpoints (2026-11-16)

Change from base, points.

| | step | seen-list | HellaSwag | MMLU | TruthfulQA | WinoGrande | HumanEval | IFEval | mean |
|---|---|---|---|---|---|---|---|---|---|
| FFT lr 3e-5 | 1000 | 78.2% | -6.5 | -3.8 | -1.0 | +0.2 | -23.8 | -56.2 | -15.2 |
| LoRA r4 lr 3e-4 | 1000 | 77.6% | 0.0 | -1.2 | -0.7 | +0.2 | +4.9 | -3.1 | 0.0 |
| DT lr 1e-3 | 400 | 87.2% | +0.4 | +1.3 | -0.5 | +2.1 | +4.3 | -2.8 | +0.8 |

## Neighbouring learning rates, seed 42 (2026-11-17)

| | lr | step | seen-list | suite change | IFEval |
|---|---|---|---|---|---|
| FFT (retrained) | 3e-6 | 1200 | 83.3% | -0.7 | -7.0 |
| FFT (retrained) | 1e-5 | 1200 | 74.4% | -3.5 | -22.0 |
| FFT, selected | 3e-5 | 1000 | 78.2% | -15.2 | -56.2 |
| LoRA | 1e-4 | 1600 | 76.9% | +0.6 | -0.6 |
| LoRA, selected | 3e-4 | 1000 | 77.6% | 0.0 | -3.1 |
| DT | 5e-4 | 400 | 80.1% | -0.8 | -6.3 |
| DT, selected | 1e-3 | 400 | 87.2% | +0.8 | -2.8 |
| DT | 3e-3 | 400 | 80.1% | -11.8 | -53.8 |
