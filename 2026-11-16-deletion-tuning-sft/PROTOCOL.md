# Protocol

The pre-registration for this entry. Sections 1 to 5 are the kickoff text from `LABNOTES.md`,
moved here unchanged on the evening of 2026-11-16, while the run was in progress. Section 6
lists every amendment in the order it was made; each was written into the lab notes before
the runs it affects. The entry was not under version control during the run, so that order
rests on the lab notes and on file times.

## 1. Hypotheses

- H1 (efficacy). DT trained with SFT recovers at least 90% of FFT's gain over base on the
  new task's held-out test set. 90% is the bottom of the band the paper reports for small
  backbones.
- H2 (retention, primary). At matched new-task performance, DT's prior-task suite score
  drops less than FFT's and less than LoRA's.

## 2. What confirms and what rejects

Primary endpoint for H2: `d_prior` = suite score of the tuned model minus suite score of
base, in points, same items, mean over three seeds, at the matched checkpoint of each method.

- Confirm: `d_prior(DT) - d_prior(FFT) >= +1.0` and `d_prior(DT) - d_prior(LoRA) >= +1.0`,
  each with a paired-bootstrap 95% interval that excludes zero, and `d_prior(DT) >= -1.0`.
- Partial: the inequality holds against one baseline only. Reported as two findings.
- Reject: an interval includes zero, or DT forgets as much or more.
- Untestable: neither FFT nor LoRA loses a point. Then this vehicle induces no forgetting
  to prevent and I say so.

H1 confirm: recovered fraction `(DT - base) / (FFT - base) >= 0.90` on test, mean of three
seeds. Reject: below 0.90 after the declared grid.

## 3. Protocol

- Same data, batch size, step count and seeds for all three methods. AdamW, fp32 master
  weights, bf16 autocast, constant learning rate, loss on assistant tokens only. Seeds 42,
  43, 44 for the final comparison; tuning on seed 42 only.
- "Iterating to similar end results" is tuning on the new task's validation split and
  nothing else. Declared grids, at most eight configurations per method:
  - FFT: lr in {1e-6, 3e-6, 1e-5, 3e-5}.
  - LoRA on every linear projection in each block (attention q/k/v/o, conv in/out, MLP
    w1/w3/w2): rank in {1, 4}, lr in {3e-5, 1e-4, 3e-4, 1e-3}.
  - DT: eta = 1/2048, lambda = 1e-5, Wanda scores from 2048 training prompts (all paper
    defaults); lr in {1e-4, 3e-4, 5e-4, 1e-3}; alpha ramp in {(1e2, 1e3), (1e3, 1e4)}.
- Training length is set once, by a LoRA pilot, before the grids run, and then held fixed.
- Selection: per method, the configuration with the best validation score. Matched
  checkpoint: let L be the lowest of the three best validation scores; each method uses its
  earliest saved checkpoint that reaches L (within one standard error). Secondary analysis,
  the paper's own protocol: every method at its final step.
- The prior-task suite is never used to choose anything. Base is evaluated in the same
  session and on the same items as the tuned models.
- Also reported: fraction of all weights deleted, adapter bytes (zstd of the packed mask;
  LoRA factors; fp32 checkpoint), and the DT mask at step 0 (Wanda prior only) on the new
  task as the control for claim 4.

## 4. Implementation gates (must pass before any comparison is run)

- With all logits positive, the masked model reproduces base logits exactly.
- Hard-mask sparsity at initialisation is about 0.18% of gate weights (the paper's figure
  for eta = 1/2048 with its noisy init).
- At the end of training, hard-mask and soft-mask evaluation agree within one point.
- A merged LoRA model and the unmerged one give the same validation score.

## 5. Kill criteria for the programme

- K1. If FFT's best validation gain over base is under 10 points, the task is not a vehicle
  for "acquiring a capability". One switch to the declared fallback task is allowed; a third
  task is not.
- K2. If DT cannot reach half of FFT's gain inside its declared grid, H1 is rejected and H2
  is tested only at the level DT reaches. I do not add tricks beyond the paper's knobs.
- K3. If neither baseline loses a point on the suite at its final step, one escalation is
  allowed (4x the steps, all methods). If still nothing, H2 is untestable here and I stop.
- K4. Any failed implementation gate stops the comparison until fixed.
- Budget: about 12 GPU-hours in total.

## 6. Amendments, in order

Amendments 1 to 10 are dated 2026-11-16, amendments 11 and 12 2026-11-17. "Before" says what had and had not been run when the amendment was written.

| # | Amendment | Why | Before |
|---|---|---|---|
| 1 | New task is Wordle tool-calling (`openenv/wordle-tool-calls-sft`); fallback task Toki Pona translation. Matched step is fixed on seed 42 and applied to all seeds. | Kickoff decisions. | Any GPU time |
| 2 | Two Wordle pools: the 156 secrets the environment uses (all are answers in training) and 800 common words that appear nowhere in training (300 validation, 500 test). | The dataset's secrets all recur in training, so wins could be memorised. | Any GPU time |
| 3 | Metric is win rate within six guesses. Plain greedy decoding for every model. Suite details fixed (completion-style multiple choice and HumanEval, IFEval through the chat template with thinking off, prompt-level strict accuracy). LoRA alpha = 2 x rank. DT: per-matrix quantile, no weight decay, Wanda norms from the 950 training games. | Read 24 base-model games first; the paper leaves these open. | Any training |
| 4 | Primary new-task score is the held-out pool if the LoRA pilot gains 10 points on it, otherwise the seen-list. Training length T is the smallest of {400, 800, 1600} within 5 points of the pilot's best. | Rules for the pilot, written before it ran. | The pilot |
| 5 | Tuned-model turn budget 256 tokens (was 512); token-budgeted batches; a turn unfinished after 160 tokens continues in a second batch; suite batch sizes 16 and 96, base rerun with them. | Cost and memory only. Outcomes checked identical on 48 games. | Any method comparison |
| 6 | Seen-list win rate is primary (held-out gain was 2.3 points). The T rule is applied to the primary score, giving T = 1,600. | Rule 4 applied; the T rule had been worded for the pool that did not move. | Any method comparison |
| 7 | Training saves checkpoints every 200 steps and a separate evaluator scores them as deployed bf16 models. Final stage: suite at the matched step for seeds 42, 43, 44; suite at step 1,600 for seed 42 only; seen-list at step 1,600 for all seeds (H1). | Cost and crash safety. | Any method comparison |
| 8 | First-pass grids are a subset of section 3 (FFT 3e-6, 1e-5, 3e-5; LoRA r4 1e-4, 3e-4, 1e-3; DT 3e-4, 5e-4, 1e-3), extended when the best value sits on an edge; LoRA rank 1 and the DT 1e3-to-1e4 ramp run once at the best learning rate. | Cost. | Any method comparison |
| 9 | If DT at its matched step forgets less than a baseline the score mismatch is left alone; if it forgets as much or more, DT's step-300 checkpoint is scored as the closer match. | DT's saved checkpoints jump from 45% to 87% across the matching level. | Any suite score of a tuned model |
| 10 | Diagnostic, one seed: suite at neighbouring learning rates that reach the same Wordle level (FFT 3e-6 and 1e-5, LoRA 1e-4, DT 5e-4 and 3e-3). | FFT's selected rate won by less than the noise and is ten times the paper's. | Seeds 43 and 44; after seed 42's matched-step suite scores |
| 11 | Follow-up after a review of the finished entry (section 7). Budget raised from 12 to 16 GPU-hours. | The only regime where a method difference could show, every method short of its cliff, had one seed and an unconverged DT checkpoint; and DT had not been run at the paper's own schedule. | All results above and the review; seed 42 already seen at these learning rates |
| 12 | Exploratory description, no endpoint and no threshold: what each method costs to train (from the logs of runs already made), and what the trained masks contain (from the saved masks). Three parts of each paper-rates mask are scored on seen-list Wordle only: the learned deletions without the starting ones, and the learned deletions split by whether the logit started below or at and above 0.02. | The entry did not say what the methods cost or how much of a mask is inherited from where it starts. | All results above. The 0.02 split was fixed before the starting-logit table was computed |

## 7. Follow-up registered 2026-11-17 (amendment 11)

**A. The paper's SFT learning rates, same steps, three seeds.** FFT 3e-6, LoRA rank 4 at
1e-4, DT 5e-4 with the ramp 100 to 1,000 over 1,600 steps. Seeds 42, 43, 44. The suite is
scored at step 1,600 for every run: no checkpoint selection. Seen-list wins at steps 400,
800, 1,200 and 1,600. The base suite is re-scored in the same session; if any task moves
more than 0.2 points, changes are computed against the new base.

- Endpoints: suite change from base (six-task mean) and the four-task multiple-choice mean
  (HellaSwag, MMLU, TruthfulQA, WinoGrande), per method, mean of three seeds. IFEval is
  reported split into answers that open a thinking block and the rest.
- Intervals: over seeds (t with 2 degrees of freedom, paired by seed for differences) and
  over items (paired bootstrap on seed-averaged scores). A claim needs both.
- "Forgets": six-task change of -1.0 or lower with both intervals below zero.
- "Methods differ": DT minus a baseline on the multiple-choice mean has the same sign in
  all three seeds and both intervals exclude zero. Its size is reported either way.
- If no method forgets, H2 is untestable in this regime (the outcome named in section 2).

**B. DT at the paper's exact SFT recipe.** 400 steps with the ramp over those 400, lr 5e-4,
eta 1/2048, lambda 1e-5, seeds 42, 43, 44. Seen-list and held-out wins, mask density and
size, and the suite, all at step 400. Descriptive, no threshold.
