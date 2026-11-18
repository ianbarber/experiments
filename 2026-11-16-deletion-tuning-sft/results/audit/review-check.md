# Review check, 2026-11-17

A fresh Claude Code agent with no part in the run read the finished entry against the
paper's TeX source and the raw run files. It is a project agent: this is a check on
arithmetic and reading, not outside validation. It recomputed every headline number from the
run directories (all matched) and found no bug that changes a result. What it found wrong,
as I logged it at the time, before the follow-up runs. Items 1 to 6 I rechecked myself
against the raw files and the paper. Item 5 was later narrowed by the follow-up: the masks
trained at lr 5e-4 are smaller than the rank-1 LoRA.

1. **The Wordle interval ignored seeds.** "+13.7 points (8.8 to 18.6)" for DT over FFT
   resamples secrets with the three seeds held fixed. The per-seed gaps are 11.5, 27.6 and
   1.9; a t-interval over seeds runs from -19 to +46. Most of the gap is FFT seed 43 (55.8%)
   at lr 3e-5, the rate I already think is mis-tuned.
2. **The registered H2 result is a selection coin-flip.** Choosing by validation loss over
   the same seed-42 checkpoints takes FFT at 3e-6 (0.0673) and DT at 3e-3, step 400 (0.06108
   against 0.06113 for 5e-4 at step 800). Those two score -0.7 and -11.8: the registered
   result reversed.
3. **Below the cliff, the IFEval drop is a formatting habit.** Answers that do not open a
   thinking block pass at 75 to 81% for every non-collapsed checkpoint (base 76.9%). What
   changes is how many answers open one: 60 of 541 for base, 76 to 102 for DT, 121 to 208
   for LoRA, 183 for FFT at 3e-6.
4. **"These are the paper's SFT settings" was wrong.** The paper trains 400 steps with DT at
   5e-4. I ramp the mask over 1,600 steps and selected 1e-3.
5. **"The smallest adapter" was wrong.** My own rank-1 LoRA is 4.0 MB and wins 78% at step
   1,600. The mask is 5.4 to 5.8 MB.
6. **Three rows of my claims table were unfair to the paper.** It documents "under 0.05%
   deleted" for its RL runs only (the abstract says it of all three regimes; its SFT masks
   are 1.0 MB at 1B and 3.1 MB at 3B). It tunes per regime and does not claim one
   configuration for everything. Its retention result is for RL, and I ran no random-sparse
   baseline, so I cannot speak to "deletion itself".

Taken from the review without rechecking: on the four multiple-choice tasks DT is about 1.4
points above LoRA with the same sign in every seed; the 256-token cap truncates 7 to 18 DT
turns per seed and almost no baseline turns; LoRA at 3e-4 is overfit by step 1,600 on the
fresh seeds; `results/gates_lora.json` still records 142 LoRA modules from before the
`g_proj` fix (every reported run has 153).

Sizes in this list are in MiB (2^20 bytes). The report uses MB (10^6 bytes): rank-1 LoRA 4.2 MB, the
mask at lr 1e-3 5.7 to 6.1 MB.
