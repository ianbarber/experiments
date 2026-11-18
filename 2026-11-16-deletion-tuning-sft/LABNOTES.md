# Lab notes: Deletion Tuning under SFT, and what it forgets

Written as I went, in order, 2026-11-16 and 17. Orchestrated with Claude Code on `worker-b`
(one RTX 5090, 32 GB). I edited these notes for length only: the kickoff protocol text moved
unchanged to [`PROTOCOL.md`](PROTOCOL.md) during the run, the blow-by-blow of getting the
harness to work moved to [`results/audit/harness-notes.md`](results/audit/harness-notes.md),
and the larger per-run tables moved to
[`results/audit/per-run-tables.md`](results/audit/per-run-tables.md). Every failure and
every decision is still here, in the order it happened. The last entries are a review of
the finished entry and the follow-up runs it led to; `REPORT.md` states the conclusions
from all of it.

## 2026-11-16 — kickoff, before any GPU time

### What I am reproducing

Lu et al., "The Surprising Effectiveness of Deleting Weights in LLM Post-Training"
(arXiv 2611.10866). The method, Deletion Tuning (DT), freezes the model and learns one logit
per weight of each block's MLP gate projection. Training multiplies the weight by
`sigmoid(alpha * G)` with `alpha` ramped up; deployment keeps the weight if `G > 0` and zeroes
it otherwise. Nothing is ever written except zeros. The paper's claims that I can test on one
card:

1. Under SFT, DT matches full fine-tuning (FFT) on the new task (their Table 5, Llama 1B/3B).
2. DT deletes under 0.05% of all weights and the mask compresses to about a megabyte.
3. DT forgets less: prior-task accuracy stays within a point of base while FFT, LoRA and
   random sparse tuning lose 1.3 to 5.4 points (their Figure 3).
4. The gain comes from learning the mask, not from the Wanda prior that initialises it
   (their Table 4).

Claim 3 was only shown under GRPO. I am testing it under SFT, which is an extension, not a
replication, and it is the part I care most about. There is a reason to expect it might not
transfer. The paper leans on RL's Razor (forward KL to the base policy on the new task
predicts forgetting). Under RL many policies earn the same reward, so a method can win while
staying close to base. Under SFT every method is pulled toward the same fixed targets, so at
equal new-task fit the three policies should sit at about the same distance from base on the
new task. If forgetting is set by that distance, the three methods forget alike. If the
parameterisation matters on its own, DT still forgets less.

### Object

- Model: `LiquidAI/LFM3-1.7B` (released 2026-10-23, 1.73B parameters, 24 blocks: 13 gated
  short convolutions, 5 sliding-window attention, 6 global attention). Every block has a
  SwiGLU MLP, `w2(silu(w1(x)) * w3(x))`, so `w1` is the gate projection the paper masks.
  That gives 24 x 2048 x 7680 = 377M maskable weights, 21.8% of the model. I picked this size
  because true fp32 AdamW full fine-tuning (weights, gradients, two moments: 16 bytes per
  parameter, 27.7 GB) fits on the card, so the FFT baseline is not a compromise.
  The paper reports that DT trails FFT a little at 1B and closes the gap by 3B, so 1.7B sits
  where a small shortfall would be on-paper behaviour, not a failed replication.
- New task: chosen at kickoff (see the next entry).
- Prior-task suite: the paper's six, via lm-evaluation-harness 0.4.12: HellaSwag, MMLU,
  TruthfulQA (MC1), WinoGrande, IFEval, HumanEval. Suite score is the unweighted mean of the
  six task scores, as in the paper.

### Hypotheses, endpoints, protocol, gates, kill criteria

Written here at kickoff, before any GPU time. After the run I moved that text, unchanged, to
[`PROTOCOL.md`](PROTOCOL.md) so these notes stay readable; every amendment below is listed there
with its reason. In one line each: H1, DT recovers at least 90% of FFT's gain on the new task.
H2 (primary), at matched new-task score DT loses at least a point less on the six-benchmark
suite than FFT and than LoRA, with DT itself within a point of base.

### Headline figure, sketched twice with made-up numbers

x: new-task score on validation. y: change in prior-task suite score from base, points.
One trajectory per method, one marker per saved checkpoint. F = FFT, L = LoRA, D = DT.

If H2 holds:

```
 d_prior
  +1 |
   0 |*---D------D-------D--------D       D stays flat within a point of base
  -1 |    F L
  -2 |        F   L
  -3 |            F     L
  -4 |                 F      L
  -5 |                     F       L
     +---------------------------------> new-task score
      base                      target
```

If H2 is rejected (forgetting follows how far the task was learned, not the method):

```
 d_prior
  +1 |
   0 |*
  -1 |    F L D
  -2 |        F D L
  -3 |            F  L D
  -4 |                 F  D  L
  -5 |                     F  L D
     +---------------------------------> new-task score
      base                      target
```

The two sketches differ in where the D markers sit, which is the thing being measured, so
the object can separate the hypotheses provided the baselines actually move down. A third
picture, all three flat at zero, is the untestable case above.

## 2026-11-16 — kickoff decisions

Sketches reviewed at kickoff. New task: Wordle through a tool-calling interface, trained on
`openenv/wordle-tool-calls-sft` (published 2026-10-19; 1,000 winning solver games). Fallback
if K1 fires: English to Toki Pona translation. Model: LFM3-1.7B. Primary endpoint:
forgetting at matched new-task score; the paper's same-steps protocol is secondary. The
matched step is fixed on seed 42 and applied to all seeds, with test scores reported so the
match can be checked.

Reading the dataset card and generator before touching the GPU turned up the property that
shapes the test. The environment draws its secret from 156 five-letter nouns, the solver's
candidate set is that same list, and every one of the 156 is the answer several times in
training. A model can win a lot by memorising 156 answers. So I score two pools in a local
simulator: the seen-list (those 156, one greedy game each) and a held-out pool of common
words that appear nowhere in training, split once into validation and test.

## 2026-11-16 — harness, and what the base model does at Wordle

The simulator reproduces every environment message of the 1,000 published games byte for
byte. First GPU minutes: 24 base-model games, reading the transcripts.

What I saw: with thinking on, the base model opens with APPLE every time, then spends its
whole turn puzzling over what the feedback block means ("the secret word might be GAME?")
and never issues a second guess inside 1,024 tokens. Given 4,096 tokens it does guess again
(hello, world, smile, brain), ignoring the feedback: 7% of later guesses are consistent with
what it has been told. With thinking off it fires guesses with no reasoning and dies on
invalid words. Wins: 0 of 24 under each of four settings. One of those four was first
reported wrong by a parser bug of mine (the thinking-off run); fixed and rerun.

So the failure is "cannot read the board", not "weak guesses". Win rate within six guesses
measures something I have watched fail, and a win needs the whole chain (format, reading
feedback, proposing a consistent word), so that is the metric.

Fixed here, before any training (amendments 3 and 4 in `PROTOCOL.md`): plain greedy decoding
for every model; the suite's scoring details; LoRA alpha = 2 x rank; DT's open details; the
rule that the held-out pool is the primary score only if the pilot gains 10 points on it;
the rule for the training length T.

## 2026-11-16 — implementation gates

All pass but one (`results/gates_dt.json`, `results/gates_lora.json`, detail in the audit
tables). DT with nothing deleted reproduces base logits bitwise, and its initial mask zeroes
0.184% of gate weights (paper: about 0.18%), 0.040% of the model. The one that failed as
worded: a merged LoRA matches the unmerged one in fp32 but is 0.024 nats off in bf16,
because the update is about one bf16 rounding step. So I evaluate LoRA unmerged. Full
fine-tuning with fp32 AdamW peaks at 26.9 GiB of the card's 31.4.

Every model, base included, is evaluated as a bf16 deployed copy.

## 2026-11-16 — pilot (two crashes, then a result)

LoRA rank 4, lr 1e-4. Attempt 1 ran out of memory at its step-800 evaluation; I had a small
test running on the same card and blamed that. Attempt 2 ran out of memory at its first
evaluation with nothing else running: prefilling hundreds of long conversations at once is
the cause. Batches are now sized by prompt tokens, and nothing else runs on the GPU during a
run. Attempt 3:

| step | val loss | seen-list win (156) | held-out val win (300) |
|---|---|---|---|
| 400 | 0.082 | 46.2% | 1.7% |
| 800 | 0.071 | 56.4% | 1.0% |
| 1200 | 0.067 | 72.4% | 2.0% |
| 1600 | 0.067 | 73.1% | 2.3% |

Read 6 held-out transcripts from attempt 1 at step 200. The format and the board-reading
narration are learned ("In the word but misplaced: A (not 3rd). Ruled out: E, N, R, S.").
The candidate step is not: "Words that fit" lists training-list words that violate the
stated constraints, then the model repeats an earlier guess and is thrown out, or loops on
one word until the budget ends. On held-out secrets no listed word can be right. The dataset
teaches this environment's 156 answers much faster than it teaches Wordle.

Applying the rules written beforehand:

- Held-out gain is 2.3 points, under the 10-point bar. The seen-list win rate is the primary
  new-task score, and I report held-out win rates beside it.
- T. I had worded the length rule for the held-out pool, before knowing it would not move.
  Amendment 6: the rule is applied to the primary score. 56.4% at 800 and 73.1% at 1,600
  are more than 5 points apart, so T = 1,600.

Accepted consequence: the seen-list has no separate test split, so reported new-task scores
carry selection optimism. Seeds 43 and 44 are never used for tuning, and the forgetting
endpoint is untouched by it.

Noise, seen directly: the same step-1,600 adapter scored 73.1% in one batching and 76.9% in
another, with identical validation loss. Six of 156 games flipped on bf16 rounding. I treat
a single 156-game score as plus or minus 4 points.

Cost amendments 5, 7 and 8, all before any method comparison: a 256-token turn budget;
training saves checkpoints every 200 steps and a separate evaluator scores them; suite at
the matched step for three seeds and at step 1,600 for seed 42 only; first-pass grids
extended only when the best value sits on an edge.

## 2026-11-16 — tuning on seed 42

Fourteen runs of 1,600 steps: FFT at five learning rates from 1e-6 to 1e-4, LoRA rank 4 at
three and rank 1 at one, DT at four plus one with the paper's GRPO sharpness ramp. Seen-list
wins at every saved checkpoint and held-out validation wins at the end are in
`results/grid.json`, `images/tuning.png` and the audit tables. Best checkpoints: FFT 85% at
lr 3e-5, LoRA 79% at rank 4 and lr 3e-4, DT 88% at lr 1e-3.

The edge rule fired three times: FFT's best first sat at 3e-6 (so I ran 1e-6), then at 3e-5
(so 1e-4), and DT's sat at 1e-3 (so 3e-3). Every selected value is now interior.

Interpretation. All three methods learn this environment to roughly the same place, 77 to
88% wins in about three guesses, and none of them learns Wordle: held-out words stay under
3% everywhere. DT is not behind; on this seed it is ahead and gets there in a quarter of the
steps. A single 156-game score moves by 5 points or more between neighbouring checkpoints of
the same run, so "best checkpoint" is partly a lucky draw and I do not read the 85 versus 88
versus 79 ordering as real until the other seeds are in.

DT's mask. Soft and hard masks agree from step 400 on (validation loss 0.0625 versus 0.0627
at lr 1e-3), which is the paper's end-of-training gate. The paper's GRPO sharpness ramp
(1e3 to 1e4) freezes the mask by step 200 and learns almost nothing; the SFT ramp is the one
that works. The mask is not as sparse as the paper's headline: at lr 1e-3 it deletes 1.43%
of gate weights, 0.31% of all weights, 5.8 MB compressed. At 5e-4 it is 0.18% of all
weights, at 3e-3 it is 1.4%. The "under 0.05%" figure does not hold in this setting.

Selection and matching, by the written rule. L is the lowest of the three bests, 79.5%
(LoRA), tolerance one standard error, 3.2 points. Earliest saved checkpoint reaching it:

| method | config | matched step | seen-list there | val loss there |
|---|---|---|---|---|
| FFT | lr 3e-5 | 1000 | 78.2% | 0.0706 |
| LoRA | r4, lr 3e-4 | 1000 | 77.6% | 0.0667 |
| DT | lr 1e-3 | 400 | 87.2% | 0.0627 |

DT overshoots. It is at 44.9% at step 200 and 87.2% at step 400, and I only saved every 200
steps. The rule stands, so DT is compared at step 400 with a higher new-task score than the
baselines. Written before any forgetting number exists: if DT at step 400 forgets less than
a baseline, it has learned more and forgotten less and the mismatch only strengthens that
reading. If it forgets as much or more, the mismatch confounds the comparison, and I will
score DT's step-300 checkpoint (saved for the fresh seeds) as the closer match.

## 2026-11-16 — first forgetting numbers (seed 42, matched checkpoints)

Suite change from base: FFT -15.2 (IFEval -56.2, HumanEval -23.8), LoRA 0.0 (IFEval -3.1,
HumanEval +4.9), DT +0.8 (IFEval -2.8, HumanEval +4.3). Per task in the audit tables.

The fully fine-tuned model has become a Wordle bot. Its IFEval answers read "No feedback
yet, so I want an opener with common words. TRADE tests..." or are stray tool calls. LoRA
and DT answer the prompts they are given.

Tentative reading: against FFT the confirming sketch, against LoRA the "nobody forgets"
picture. Neither adapter loses a point on the mean.

A confound I have to deal with before believing the FFT number. FFT's learning rates from
3e-6 to 3e-5 were inside one standard error of each other on Wordle (best checkpoints 79.5,
76.9, 85.3), and my selection rule took the largest of them. 3e-5 is ten times the paper's
SFT range. If FFT at 3e-6 reaches the same Wordle level without this damage, then what I
have measured is "an aggressive learning rate wrecks the model", not "full fine-tuning
forgets". The primary analysis stays as registered. As a labelled diagnostic, one seed, I
will score the suite for FFT lr 3e-6 and lr 1e-5 at step 1200 (79.5% and 76.9% seen-list,
both inside the matching tolerance), and for the adapters at the neighbouring learning
rates that also reach the level: LoRA lr 1e-4 at step 1600, DT lr 5e-4 and lr 3e-3 at step
400. That gives forgetting against learning rate for each method at one Wordle level.

What each outcome would mean, written now: if low-rate FFT stays within a point of base,
the headline becomes "full fine-tuning needs a conservative rate to be safe and loses
nothing at Wordle for it; the adapters were safe at every rate that worked". If low-rate
FFT still loses several points, the method contrast stands on its own.

An own goal: I had deleted the unselected FFT checkpoints to free disk, so the two FFT
diagnostics are retrained from the same seed and configuration rather than rescored.

## 2026-11-16 — primary endpoint: three seeds at the matched checkpoints

Base, rescored on the final evaluation path: suite 48.71 (HellaSwag 41.1, MMLU 49.5,
TruthfulQA 33.0, WinoGrande 55.2, HumanEval 43.3, IFEval 70.1), Wordle 0 of 156, 0 of 300,
0 of 500. The same model under my earlier batch sizes scored 48.44: WinoGrande and IFEval
each moved by about 0.9 from padding alone. So 0.3 points on the suite mean is noise I can
produce without changing a weight.

| | step | seen-list win, 3 seeds | held-out test win | suite change, 3 seeds | mean | 95% interval |
|---|---|---|---|---|---|---|
| FFT lr 3e-5 | 1000 | 78.2, 63.5, 74.4 | 0.4% | -15.2, -15.0, -18.9 | -16.4 | -17.8 to -14.9 |
| LoRA r4 lr 3e-4 | 1000 | 77.6, 72.4, 83.3 | 1.3% | 0.0, -0.6, -0.6 | -0.4 | -1.7 to +0.9 |
| DT lr 1e-3 | 400 | 87.2, 80.1, 84.0 | 2.7% | +0.8, +1.9, +1.7 | +1.5 | +0.4 to +2.5 |

DT minus FFT: +17.8 (16.5 to 19.1). DT minus LoRA: +1.9 (0.9 to 2.8). Intervals are a
paired bootstrap over benchmark items on seed-averaged scores.

By the letter of the registration this is "confirm": both differences exceed a point with
intervals clear of zero, and DT is within a point of base (above it). Two things stop me
writing it that plainly.

First, LoRA does not lose a point either. The LoRA contrast is DT going up, not LoRA going
down, and the suite mean hides what LoRA does lose: IFEval falls 6.8 (-3.1, -8.5, -8.9)
while HumanEval rises 7.1. DT moves -0.6 on IFEval, +6.5 on HumanEval, +1.5 on MMLU in
every seed (+1.3, +1.7, +1.4) and +1.6 on WinoGrande. A seven-point HumanEval rise for both
adapters after Wordle training is not a coding improvement I believe in without reading the
completions, so I queued that.

Second, the FFT number belongs to one learning rate. The diagnostic is queued.

The match itself: on seed 42 the scores were matched by construction except for DT's
overshoot. On fresh seeds FFT drops to 63.5 and 74.4 and LoRA ranges from 72 to 83, which
is the selection optimism I expected. DT is the highest or near it in every seed and
forgets nothing, so rule 9 leaves the step-300 checkpoint unscored.

Sketch check: against FFT this is the confirming sketch, steeper than drawn. Against LoRA
it is neither sketch. LoRA sits on the zero line with DT.

## 2026-11-17 — end of schedule, the learning-rate diagnostic, and two controls

End of the 1,600-step schedule. Seen-list wins on three seeds: FFT 76.9, 55.8, 79.5 (70.7);
LoRA 75.0, 62.8, 60.9 (66.2); DT 88.5, 83.3, 81.4 (84.4). H1: DT recovers 119% of FFT's
gain. Suite change on seed 42: FFT -18.5, LoRA -1.3 (-2.7 to +0.1), DT +0.4 (-0.8 to +1.7).
K3 does not fire.

The diagnostic (`results/lr_diagnostic.json`, table in the audit tables), seed 42, suite
change from base at a similar Wordle score: FFT -0.7 at 3e-6, -3.5 at 1e-5, -15.2 at the
selected 3e-5. LoRA +0.6 at 1e-4, 0.0 at the selected 3e-4. DT -0.8 at 5e-4, +0.8 at the
selected 1e-3, -11.8 at 3e-3.

This is the first branch I wrote down, and worse for my earlier reading than I allowed for.
Low-rate FFT is within a point of base at the same Wordle score. And the second half of my
sentence, "the adapters were safe at every rate that worked", is false: DT at 3e-3 works at
Wordle and is wrecked, 474 of 541 IFEval answers are Wordle text. DT at 5e-4 sits below
LoRA at 1e-4, so the 1.9-point DT-over-LoRA margin is the size of the setting-to-setting
spread. The retrained FFT runs score 83.3% and 74.4% at step 1,200 where the originals
scored 79.5% and 76.9%, more of the same noise.

Decision: the registered endpoint is reported as registered, and demoted. It is confounded
by where each method's Wordle-optimal learning rate fell. The headline is the diagnostic.

HumanEval, reading the completions (`results/humaneval_reading.json`): base stalls on 29 of
164 problems (`pass`, or a TODO and then prose). DT stalls on 22, LoRA on 8, both write
shorter completions, and about half of the problems they gain are ones where base stalled.
A habit learned from short think-then-act turns, not coding skill.

Untrained-mask control: the step-0 mask wins 0 of 156 and 0 of 300.

Stop. Roughly 11 GPU-hours against the 12 I allowed. Verdict in `REPORT.md`: deletion learns
the task as well as full fine-tuning or better; I could not show that it forgets less; the
mask is small but six times denser than advertised; the dataset teaches an environment, not
the game.

## 2026-11-17 — review of the entry, and what it changes

I had a fresh Claude Code agent, one with no part in the run, review the entry against the
paper's TeX source and the raw run files, as an experiment and as a replication. It is a
project agent, so this is a check on my arithmetic and my reading, not outside validation.
It recomputed every headline number from the run directories (all matched) and found no bug
that changes a result. It also found things that were wrong. The full list is in
[`results/audit/review-check.md`](results/audit/review-check.md); I rechecked the first six
myself before accepting them. The ones that change the reading:

- My Wordle interval for DT over FFT ("+13.7, 8.8 to 18.6") resampled secrets with the seeds
  held fixed. The per-seed gaps are 11.5, 27.6 and 1.9; over seeds the interval is -19 to +46.
- The registered H2 result is a selection coin-flip. Choosing by validation loss over the
  same seed-42 checkpoints takes FFT at 3e-6 and DT at 3e-3, which score -0.7 and -11.8: the
  registered result reversed.
- Below the cliff, the IFEval drop is a formatting habit. Answers that do not open a thinking
  block pass at 75 to 81% for every non-collapsed checkpoint (base 76.9%). What changes is
  how many answers open one.
- I had the paper wrong in places. Its SFT runs are 400 steps with DT at 5e-4; it documents
  "under 0.05% deleted" for RL runs only; it tunes per regime; its retention claim is for RL.
- My rank-1 LoRA (4.2 MB) is smaller than the mask at lr 1e-3 (5.8 MB).

One more, about process. The entry was not under version control while it ran, so the order
of protocol and results rests on these notes and on file times. `PROTOCOL.md` was split out
of these notes on the evening of the 16th, during the run, not after it as I wrote.

Decisions:

- Re-analysis, no GPU: intervals over seeds as well as over items; the four-task
  multiple-choice mean beside the suite mean; IFEval split into thinking-block answers and
  the rest; the validation-loss selection counterfactual; the claims table corrected.
- A follow-up, because the only regime where a method difference could show (every method
  short of its cliff) has one seed and an unconverged DT checkpoint. Registered next.
- The report gets rewritten around what the whole body of evidence supports.

## 2026-11-17 — follow-up, written before it runs (amendment 11)

What I have already seen at these settings, seed 42: FFT 3e-6 at step 1,200 is -0.7 on the
suite, LoRA 1e-4 at step 1,600 is +0.6, DT 5e-4 at step 400 is -0.8.

Full text in `PROTOCOL.md` section 7. In short:

**A. Every method at the paper's SFT learning rate, same steps, three seeds.** FFT 3e-6,
LoRA rank 4 at 1e-4, DT 5e-4. All 1,600 steps, suite scored at step 1,600 for every run, so
there is no checkpoint selection. Seed 42 reuses the LoRA pilot and the DT tuning run; FFT
is retrained a third time because no step-1,600 checkpoint survives. The base suite is
re-scored in this session first.

Question: does any method lose prior-task score at these rates, and do the methods differ?
Fixed now: every number gets an interval over seeds and one over items, and a claim needs
both. "Forgets" is a six-task change of -1.0 or lower with both intervals below zero.
"Methods differ" is DT minus a baseline on the four-task multiple-choice mean with the same
sign in all three seeds and both intervals clear of zero. If nothing forgets, H2 is
untestable in this regime.

Sketches, multiple-choice mean change from base, one letter per seed:

```
a method difference survives            nothing to separate
 +1 |           D D D                    +1 |
  0 |-----------------------              0 |--F-D-L---D-L-F---L-F-D--
 -1 |  F F F                             -1 |
    |        L L L
```

**B. DT at the paper's exact SFT recipe.** 400 steps with the ramp over those 400, lr 5e-4,
eta 1/2048, lambda 1e-5. Seeds 42, 43, 44. Seen-list and held-out wins, mask density and
size, suite, all at step 400. Descriptive, no threshold: does it learn the task on the
paper's schedule, how dense is the mask, and is it smaller than the rank-1 LoRA. The
baselines' step-400 Wordle scores come free from the runs in A, since their learning rate
is constant.

Budget: about 4.5 GPU-hours on top of the 11 used, so the 12-hour budget becomes 16.

## 2026-11-17 — follow-up results

Base, re-scored in this session: identical to yesterday's on every item of all six
benchmarks. No session effect to correct, so changes are against the same base.

**A. The paper's learning rates, step 1,600, three seeds** (`results/final.json`, cell
`paper_rates`). Suite change is the six-task mean; MC is the four-task multiple-choice mean.

| | seen-list win, seeds 42 / 43 / 44 | suite change per seed | mean | over seeds | over items | MC |
|---|---|---|---|---|---|---|
| FFT 3e-6 | 78.2 / 67.3 / 61.5 | -0.67 / -1.01 / -0.43 | -0.70 | -1.43 to +0.03 | -1.95 to +0.53 | -0.17 |
| LoRA r4 1e-4 | 76.9 / 57.7 / 70.5 | +0.62 / +0.05 / -0.45 | +0.08 | -1.26 to +1.41 | -1.11 to +1.29 | -0.43 |
| DT 5e-4 | 82.1 / 82.1 / 78.2 | -0.62 / -0.11 / +0.07 | -0.22 | -1.11 to +0.66 | -1.10 to +0.67 | +0.37 |

The rules as I wrote them this morning:

- Forgets (six-task change of -1.0 or lower, both intervals below zero): no method.
- Methods differ on the multiple-choice mean: DT minus LoRA is +0.80, same sign in all three
  seeds, +0.12 to +1.47 over seeds and +0.40 to +1.19 over items. Yes. DT minus FFT is +0.54,
  same sign, but -0.11 to +1.19 over seeds. No.
- Nothing forgets, so H2 is untestable in this regime.

Sketch check: on the six-task mean this is the "nothing to separate" sketch. On the
multiple-choice mean it is neither: no method falls, DT sits a little above zero (+0.30 to
+0.44 on each seed) and LoRA a little below. Most of that gap is MMLU, DT +0.8 and LoRA -1.1.

What does move is IFEval: FFT -7.0, DT -2.6, LoRA -2.1. Answers that open a thinking block,
of 541: FFT 183, 219, 171; LoRA 123, 115, 138; DT 124, 85, 84; base 60. The other answers
pass at 76 to 81% everywhere (base 76.9%). So the same formatting habit as before, and full
fine-tuning picks it up most.

Wordle along training, mean of three seeds at steps 400, 800, 1,200, 1,600: FFT 57, 68, 81,
69. LoRA 41, 59, 71, 68. DT 70, 82, 82, 81. FFT falls back between 1,200 and 1,600 on every
seed while its validation loss rises. DT ends above FFT on each seed (by 4, 15 and 17
points), but over three seeds the interval on that gap is -5 to +29, and at FFT's best step
the two are level.

**B. DT at the paper's exact recipe, step 400.** Seen-list wins 64.7, 69.2, 60.3 (mean
64.7); held-out test 3.2%; suite +0.18; mask 0.136% of all weights, 3.3 MB. The baselines at
step 400, from the runs in A: FFT 56.8, LoRA 40.8. So on the paper's schedule DT is ahead of
both on every seed, and none of the three has finished learning. DT loses 22 to 46 of 156
games to a turn with no tool call, and 18 to 44 of its turns hit the 256-token cap, so those
losses are mostly cut-off turns. The baselines have 0 to 1 cut-off turns. My turn budget
costs DT games and nobody else.

One correction to this morning's review notes. The mask at lr 5e-4 is 4.0 MB after 1,600
steps and 3.3 MB after 400, both under the 4.2 MB rank-1 LoRA, so "not the smallest adapter"
is true only of the 5.8 MB mask at lr 1e-3.

Housekeeping: re-ran `gates.py lora`; `results/gates_lora.json` now records 153 modules,
other numbers unchanged.

Decision: stop. About 15 GPU-hours in total against the 16 allowed. What the whole body of
evidence says:

1. DT learns this task as well as FFT. It is at or above FFT at step 1,600 in all six
   seed-by-setting pairs; with three seeds per setting I cannot say it is better.
2. At the paper's learning rates nothing forgets. At higher rates FFT and DT both turn into
   Wordle bots. H2 is untestable where the methods are sound and confounded with learning
   rate where they are not.
3. DT is about a point above LoRA on multiple-choice accuracy, in both regimes, every seed.
   Small, and neither model is damaged.
4. The mask is 0.14 to 0.30% of the model depending on rate and length, not under 0.05%.

Closing out. `REPORT.md` and `README.md` now state the result from all of the above as one
piece. The figures are redrawn around the learning-rate axis, and the analysis reproduces
byte for byte from `results/evidence.tar.gz` alone.

## 2026-11-17 — what the methods cost, and what the masks contain

Two things the entry did not say. Neither was registered (amendment 12): no endpoint, no
threshold, description of runs I already had. New GPU time is three analysis passes and nine
seen-list evaluations, about 20 minutes.

**Cost** (`code/scripts/cost.py`, `results/cost.json`). From the step timings, leaving out
steps that follow a save: FFT 0.380 s a step, LoRA 0.347, DT 0.333, the same to three
figures across every run of a method. Peak memory 26.9, 10.2 and 15.9 GiB. DT's 5.7 GiB over
LoRA is one logit, one gradient and two Adam moments for 377M weights (5.6 GiB). The Wanda
pass takes 23 s.

**Masks** (`code/scripts/mask_anatomy.py`, `results/mask_anatomy.json`). The paper's RL
masks end at the density they start with (its Table 7), which reads as a starting mask plus
a few flips. Under SFT here the deployed mask is mostly new.

- I can rebuild the starting logits from the seed exactly: zero mismatches against the saved
  step-0 mask in all twelve runs.
- At the paper's rates 85% of the deployed deletions were made in training. Of the 693k
  starting deletions, the 184k chosen by the Wanda score stay (0.3% put back) and 45% of the
  509k chosen by init noise are put back.
- The mask is made early: 84 to 86% of the learned count by step 400, and 9k added against
  4k undone in the last 200 steps (seed 42).
- First reading of seed agreement: two seeds share 6% of learned deletions, one seed at two
  learning rates shares about half. That pointed at the starting noise, so I cut by starting
  logit. Deleted share falls from 28% for logits starting just above zero to 0.02% at the
  0.03 mean. Half the learned deletions come from the 2% of weights nearest zero.
- Ablations, seen-list wins by seed. Deployed mask: 82.1, 82.1, 78.2. Starting deletions put
  back: 81.4, 81.4, 87.8. Learned deletions that started below 0.02 (91% of them): 71.2,
  68.6, 80.1. Those that started at 0.02 or above: 0, 0, 0, validation loss 2.21 against
  2.50 for base. I fixed the 0.02 split before computing the starting-logit table.
- The RL-ramp run from tuning froze at 1.47M learned deletions by step 200, spread evenly
  over weight magnitude. That mask is close to noise, which fits its 10% win rate.

What I cannot explain: the paper's RL masks stay at their starting density (its Table 7),
and its SFT masks (1.0 MB on a 1B Llama, about 268M gate weights by my count) are about the
size an untrained mask would be, on what is nominally my recipe. Mine grow 3.4 times in 400
steps.

Reading: the method finds one of many masks that work. Training picks among the weights
whose logits happen to start near zero, and the task decides how many per block, per
dimension and per unit, not which ones.
