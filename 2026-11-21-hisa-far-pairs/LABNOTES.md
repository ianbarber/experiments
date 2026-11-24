# Lab notes: what does HiSA give up?

Written as I go, on `local-3090`. Orchestrated with Claude Code. Seed 42 unless stated.

## 2026-11-21

### Setup

Question: HiSA (arXiv 2611.26363) drops pair l's part of the attention score once its rotation passes 90°.
Are the dropped parts noise, or exact-offset information that prose rarely needs? Hypothesis, confirming and
rejecting results, kill criteria and both figure sketches are in `PROTOCOL.md`, fixed before Stage 1.

Both sketches (made-up data), copied here from the protocol when I assembled the entry; they were fixed
there before Stage 1 and have not changed. Stage 2, accuracy against D (o = offset recall, c = content
recall):

```
 if the hypothesis holds                      if it is rejected
 1.0 |c c c c c c c c c c   all variants      1.0 |c c c c c c c c c c
     |o o o o o o o o .     dense                 |o o o o o o o o . .   dense
     |o o o o o o .         hisa-360              |o o o o o o o o . .   hisa-360
     |o o o o .             hisa-90               |o o o o o o o o . .   hisa-90
 0.0 |________._._._._._                      0.0 |__________________
      2 4 8 16 .. 512 1024                         2 4 8 16 .. 512 1024
```

Stage 1, extra loss from hisa-90 against dense (grid = newline tokens):

```
 if the hypothesis holds                      if it is rejected
 prose   |#                                   prose   |#
 code    |##                                  code    |#
 tables  |###                                 tables  |#
 grid-16 |##########                          grid-16 |#
 grid-64 |################                    grid-64 |
 copy    |                                    copy    |
```

One thing out of order: the Stage 0 checks and a speed benchmark ran on the GPU before these notes existed.
They measure nothing about the hypothesis.

Environment: Python 3.12, torch 2.15 (cu132), transformers 5.24, the paper repo at commit 0fae2ce.

### Reading the paper and the repo before coding

- The paper's Section 5 already has an influence measure per pair and distance band for SmolLM-135M on
  OpenWebText, and Table 6 has the no-retraining cost for the authors' 4K model (2.906 → 2.937). So "the
  number the paper does not report" is narrower than the plan assumed: non-prose data, matched controls,
  this checkpoint at 2K.
- The repo's reference sends keys with no kept pair out of the softmax. I keep that convention.
- The repo's Hugging Face patch was written against transformers 5.18; with 5.24 it still works (the extra
  `cache` argument lands in `**kwargs`).

### Stage 0

All pass (`results/stage0_checks.json`).

- Pair layout is split-half: norm of (l, l+32) is preserved by the rotation to 4e-6; the interleaved pairing
  is off by 23. Rotation angle of pair l at position p is p·θ_l.
- inv_freq equals 10000^(-l/32) to 7e-8 relative.
- All-true table through the blocked path: max logit difference to the stock model 2.1e-4.
- hisa-45/90/360 tables equal the repo's `pair_masks`; attention outputs match the repo reference to 7e-7;
  the whole model matches `patch_llama(impl="reference")` to 6e-5 in logits; Triton kernel to 2.7e-4.
- Controls keep the same number of pairs as hisa-90 at every distance (32, 32, 26, 19, 11, 7 pairs at
  d = 0, 1, 8, 64, 512, 2047).

Speed at 2048 tokens, fp32: dense 0.05 s per window, window tables 0.19 s, the per-distance random control
1.07 s. Training, batch 8, bf16: 0.51 s per step dense, 0.59 s with the Triton kernel.

### Revisions made to the plan

Listed at the top of `PROTOCOL.md`. The two that change what can be concluded: the window edges give HiSA
exact-distance cues for free (so I record where its wrong answers come from), and the content-recall twin
needed unique keys because a 256-word stream repeats.

### Data

Built with `code/build_data.py`. Prose: first 256 OpenWebText documents with at least 2048 tokens; the next
256 feed the copy probe and 64 more the zero-mean check. Code: 256 Python files. Tables: 85 CSV, 86 JSON
lines, 85 markdown. Grids round-trip through the tokenizer. The "words" are whatever lower-case alphabetic
strings are single tokens with and without a leading space, so some are fragments ("pol", "stud").

### Zero-mean check (the "zeroing is only right if the dropped parts average to zero" pitfall)

`code/dropped_mean.py`, 64 held-out prose windows. For every layer, head, pair and exact distance I took the
mean over queries of the pair's score part (a cross-correlation, so an FFT does it).

The dropped parts do not average to zero. Pooled over layers and heads, in score units:

| rotation | rms of part | rms of its per-distance mean | share of energy in the mean |
|---|---|---|---|
| < 45° | 1.56 | 1.42 | 0.84 |
| 45–90° | 2.76 | 2.67 | 0.94 |
| 90–180° | 2.18 | 2.10 | 0.93 |
| 180–360° | 0.44 | 0.35 | 0.62 |
| 360–720° | 0.43 | 0.34 | 0.62 |
| 720°+ | 0.30 | 0.22 | 0.53 |

Reading: most of the energy of a pair's part, at every rotation, does not depend on which tokens are
involved. It is a fixed function of distance, a learned relative-position kernel. The typical cell is not
like that (median |mean|/rms is 0.15 to 0.3), so the energy sits in a minority of large cells. Summed over
the pairs hisa-90 drops, the kernel it removes has rms 3.4 and maximum 24 score units. Parts beyond 180° are
about five times smaller than those at 90–180°, in line with the paper's "suppressed beyond 180°".

Decision (the branch the protocol names): add `hisa-90-mean` to Stage 1, hisa-90 with the dropped parts
replaced by this mean. It is a diagnostic, not a model anyone would deploy.

### Dense on the grids: the gate

Looked at the dense model's newline probability around the true position before anything else.

| C | rows per window | P(newline) at true position | one cell early |
|---|---|---|---|
| 8 | 227 | 0.53 | 0.18 |
| 16 | 120 | 0.45 | 0.20 |
| 32 | 62 | 0.14 | 0.10 |
| 64 | 31 | 0.005 | 0.005 |
| 128, 256 | 15, 7 | 0.001 | 0.001 |

What I see: at C = 8 and 16 the model has a peak at the right place with leakage one and two cells early
and none late; it sharpens slowly over the first rows. At C = 32 the peak is barely above its shoulders. At
64 and beyond it never predicts a newline. Gate (≥ 0.5): only C = 8 passes. C = 16 misses at 0.45 and I
report it as descriptive, not as evidence. The threshold stays where it was set.

So Stage 1 can say little about far exact offsets: the dense model does not use them on this probe beyond
17 tokens. Stage 2 carries that question, as the plan expected.

### Stage 2 task generator

`code/stage2_task.py`. About 105 queries per sequence (55 offset, 50 content). Content queries cannot exist
at D = 2 (the answer slot would be the marker itself), so the twin covers D = 4…1024. A 60-step smoke run
at batch 2: content recall is at 0.99 after 50 steps (induction is already in the model), offset recall is
at 0.26 for D = 2 and at chance beyond 16.

### Stage 1, first pass (256 windows per data type, 64 per grid; `results/stage1_summary.json`)

Extra loss against dense, nats per token, 95% bootstrap interval over windows:

| condition | prose (2.868) | code (1.372) | tables (0.712) |
|---|---|---|---|
| hisa-45 | +1.556 | +2.013 | +1.838 |
| hisa-90 | +0.218 [0.208, 0.230] | +0.267 [0.259, 0.276] | +0.083 [0.078, 0.089] |
| hisa-180 | +0.009 [0.007, 0.012] | +0.009 [0.007, 0.010] | +0.005 [0.004, 0.006] |
| hisa-360 | +0.003 | +0.001 | +0.002 |
| hisa-720 | +0.0007 | +0.0002 | +0.0001 |
| hisa-90-mean | +0.287 | +0.250 | +0.121 |
| band-90-180 | +0.699 | +0.893 | +0.513 |
| band-180-360 | +0.013 | +0.013 | +0.007 |
| band-360-720 | +0.002 | +0.001 | +0.002 |
| band-720+ | +0.0007 | +0.0002 | +0.0001 |
| ctrl-perm (3 seeds) | +2.23, +3.05, +4.57 | +3.17, +4.36, +5.44 | +2.08, +3.05, +4.24 |
| ctrl-iid | +4.16 | +5.57 | +4.13 |
| ctrl-slow | +5.08 | +6.14 | +5.00 |

Tentative reading, written before any follow-up:

- This is not the picture either sketch drew. hisa-90 costs 0.22 nats on prose, seven times the paper's
  0.031 for its own model. That is the "broadly used" row of the plan's table.
- Everything beyond 180° is close to free on every data type (hisa-180: +0.009 or less). The cost sits in
  the 90–180° band.
- The controls are 10 to 25 times worse than hisa-90 on prose and code (25 to 60 times on tables). Which pairs are dropped matters far more than how
  many; the fast end at far distance is the cheapest place to cut, and the slow end the most expensive.
- Dropping only the 90–180° band (+0.70) is worse than dropping it and everything beyond (+0.22). The
  parts beyond 180° are therefore not independent of the band below them: left alone they do damage.
- Mean ablation does not help (prose is worse, code slightly better). The cost is not the loss of the
  content-independent distance kernel that the zero-mean check found.
- Copy accuracy without retraining: dense 0.83, hisa-90 0.53, hisa-180 0.76, hisa-360 0.83.
- Grids, C = 8 (the only size that passes the gate): P(newline) at the true position 0.53 dense, 0.21
  hisa-90, 0.58 hisa-180, 0.53 hisa-360; newline loss +1.02 [0.98, 1.05] for hisa-90 and −0.15 for
  hisa-180. C = 16 (descriptive) looks the same: 0.45, 0.12, 0.52, 0.43.

Then I looked at the loss difference by position, in 64-token bins, and it is not smooth. hisa-90 on prose
is flat at +0.02 to +0.04 up to position about 1150, steps up to about +0.2, and steps again near 1570 to
+0.5 and climbs to +0.9. Code does the same. The steps are at 1178 and 1571, the 90° windows of pairs 23
and 24. A cost that switches on at a fixed position, rather than growing with the amount of far context,
means a key at a fixed position: the first token of the window, whose distance from the query is the
query's position. Guess: heads that park attention on the first token do it through pairs 23 and 24, and
past their windows that score part is cut.

**Amendment 1 (2026-11-21).** New diagnostic conditions `<table>-keep1`: the same table, but the first key
of the sequence keeps all 32 pairs. If the guess is right, hisa-90-keep1 is flat at about +0.03 at every
position. Also `-keep4` to see whether more than one leading token is involved. Not in the protocol; added
because the position profile was not anticipated by either sketch.

### First-token guess: wrong

`hisa-90-keep1` on prose: 3.091 against 3.086 for hisa-90, the same profile by position. `-keep4` the same.
The first token is not it. Retracted.

The steps are ramps, not jumps (they build over a few hundred positions after 1178 and after 1571), which
fits "more and more keys fall outside the window of one pair" better than "one key".

**Amendment 2 (2026-11-22).** `pair-<l>-90`: cut a single pair beyond 90°, everything else dense. Prose,
256 windows, extra loss:

| pair | 90° window | all tokens | positions 1280–1408 | last 128 |
|---|---|---|---|---|
| 18 | 280 | +0.003 | 0.00 | 0.01 |
| 19 | 373 | +0.001 | 0.00 | 0.00 |
| 20 | 497 | +0.006 | 0.01 | 0.01 |
| 21 | 663 | +0.002 | 0.00 | 0.00 |
| 22 | 884 | +0.011 | 0.02 | 0.03 |
| 23 | 1178 | +0.475 | 0.79 | 1.53 |
| 24 | 1571 | +0.070 | 0.00 | 0.36 |

Pair 23 alone costs more than twice what the whole hisa-90 rule costs (+0.218); pair 24 is next; pairs 18
to 22 are nearly free. So in this checkpoint the cost of hisa-90 at 2K is the cost of cutting two specific
pairs, and the other dropped pairs partly offset it. It is not a property of "the 90–180° band" in
general: pair 22 spends positions 884 to 1767 in that band and costs 0.01. This also qualifies the cheap
hisa-180 result: at 180° the windows of pairs 23 and 24 are longer than 2048, so that rule never touches
them here.

Pairs 25 to 31 are never cut at 2K under any budget tested, so I cannot say whether 23 and 24 are special
or whether every pair from 23 up is like this.

Next: the same single-pair cut for pair 23, one layer at a time and then by head (`code/localise_pair.py`,
64 prose windows), queued behind the Stage 2 runs. This is the localisation the original plan had as
Stage 3, pointed at pair 23 rather than at the grids.

### Stage 2 pilot

Dense, 1000 steps, batch 8. Both learning rates reach about 1.00 on every offset D up to 1024 and on
content recall by step 500 (bf16, 32 sequences). Final fp32 on 128 sequences: 1e-4 is 1.00 everywhere,
3e-4 is 0.99 at D = 1024. Task installed; the kill criterion is not met. Learning rate 1e-4.

So the dense model learns exact recall at 1024 tokens with ease. "Exact far addressing is hard for RoPE
models" is not what happens here.

### Grids and copy, split by position

Because the pair-23 effect lives at late positions, I split the grid and copy metrics into positions before
1100 and after 1280 (`results/stage1_early_late.json`).

- Grid C = 8, P(newline) at the true position, early / late: dense 0.51 / 0.54, hisa-90 0.20 / 0.21,
  hisa-180 0.56 / 0.60, hisa-360 0.51 / 0.54. The grid cost of hisa-90 is the same early and late, so it is
  not the pair-23 effect. It is a short-range effect: the newline is 9 tokens back, where pairs 4 to 6 are
  between 90° and 180°.
- Copy, repeats that end before position 1100 (74 windows): dense 0.85, hisa-90 0.71, hisa-180 0.85,
  hisa-360 0.86. By gap for hisa-90: 0.84 at 64, 0.78 at 128, 0.66 at 256, 0.50 at 512 (12 to 22 windows
  each). So without retraining, hisa-90 hurts copying more the further back the source is, also where
  pair 23 is untouched.

### Stage 2 (seven runs, 4000 steps, batch 8, lr 1e-4, 33 to 39 minutes each)

Final fp32 accuracy on 128 held-out sequences, about 675 queries per cell:

| variant (seed) | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 | 512 | 1024 |
|---|---|---|---|---|---|---|---|---|---|---|
| dense (42) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.999 | 1.000 | 1.000 | 1.000 |
| dense (43) | 1.000 | 1.000 | 1.000 | 1.000 | 0.999 | 0.999 | 1.000 | 1.000 | 1.000 | 1.000 |
| hisa-45 (42) | 1.000 | 1.000 | 1.000 | 0.956 | 0.979 | 0.634 | 0.065 | 0.074 | 0.023 | 0.034 |
| hisa-90 (42) | 1.000 | 1.000 | 0.997 | 0.998 | 0.586 | 0.933 | 0.074 | 0.033 | 0.028 | 0.032 |
| hisa-90 (43) | 0.999 | 1.000 | 1.000 | 0.998 | 0.356 | 0.653 | 0.077 | 0.026 | 0.041 | 0.032 |
| hisa-180 (42) | 1.000 | 0.999 | 0.999 | 0.991 | 0.999 | 0.859 | 0.859 | 0.071 | 0.026 | 0.038 |
| hisa-360 (42) | 1.000 | 1.000 | 1.000 | 1.000 | 0.993 | 0.943 | 0.361 | 0.206 | 0.091 | 0.028 |

Content recall is 0.999 or 1.000 for every variant at every D from 4 to 1024.

What I see:

- Dense is at ceiling everywhere by step 1000. Every HiSA variant is at ceiling to D = 16, then falls to a
  floor of 0.03 to 0.08. The floor is above chance (0.004): the wrong answers are tokens near the target.
- The cliff moves out with the budget: last D at 0.9 or better is 32 for hisa-45 (with 0.63 at 64), 16 for
  hisa-90 (seed mean 0.47 at 32 and 0.79 at 64), 32 for hisa-180 (0.86 at 64 and 128), 64 for hisa-360
  (0.36 at 128, 0.21 at 256). Not the clean factor of four the plan guessed, and hisa-45 is not worse than
  hisa-90.
- hisa-90 is worse at 32 than at 64 in both seeds. I think this is the window-edge effect from the
  revisions: the nearest edges are 28 and 38 around 32, and 50 and 67 around 64, but I have not tested it.
- Where HiSA is wrong it is nearly right. At the cliff the wrong answers are the neighbours: hisa-90 at
  D = 32, 268 of 301 (seed 42) and 337 of 468 (seed 43) wrong answers are the token one position off;
  hisa-360 at D = 128, 332 of 431. Further out the errors spread over the next few to few dozen positions
  (hisa-90 at 512: mean signed error +6 in both seeds; at 256: −5), and only at 1024 does a fifth of them
  land nowhere within 32 positions. So HiSA knows about where to look and cannot resolve which token.
- The HiSA runs were still improving at the cliff when they stopped (hisa-90 seed 42 at D = 64: 0.13 at
  step 1000, 0.78 at 2000, 0.90 at 3000, 0.93 final), and the two hisa-90 seeds differ by 0.2 to 0.3 at 32
  and 64. The position of the cliff at 4000 steps is not its position at convergence. Beyond the cliff
  (128 and up for hisa-90) nothing moved over 4000 steps.

Compared with the sketch: this is the "hypothesis holds" sketch, with a less tidy ordering of the budgets.

### Where pair 23 matters (64 prose windows; `results/localise_pair23.json`)

Pair 23 cut beyond 90° in all layers: +0.47 over all tokens, +1.22 over positions from 1280 on. One layer
at a time, positions from 1280 on: layer 13 gives +0.49; every other layer is +0.03 or less. Within layer
13, cutting it in one query head at a time gives +0.003 to +0.06 (largest: head 8), which sums to +0.15,
well short of the layer's +0.49. So it is one layer, spread over its heads, and more than additive across
them. I did not look at the attention patterns.

### Verdict against the protocol

Mixed, in the protocol's sense.

- Held: Stage 2. There are offsets (128 to 1024) where dense is at 1.00, hisa-90 at 0.08 or less, content
  recall at 1.00 for both, and the hisa-360 cliff is further out than the hisa-90 cliff.
- Held in a narrow form: on the one grid that passes the gate, hisa-90 costs +1.02 nats on newline tokens
  against +0.03 per token on prose at positions where pair 23 is untouched.
- Did not hold: "nearly free on prose". hisa-90 costs +0.22 on prose over the whole 2K window, which is the
  plan's "broadly used" row. Nearly all of it is pairs 23 and 24 in the second half of the window.
- Not testable here: anything about parts beyond 180° for pairs 23 and up, and exact offsets beyond 17
  tokens in the pretrained model (the dense model fails the grid gate there).

## 2026-11-23

### Review of the write-up, and Amendment 3

A review of the entry found one error and three weak points. Details and predictions are in `PROTOCOL.md`,
Amendment 3, written before running anything.

- Error: I wrote that pairs 25 to 31 are never cut under any budget tested. hisa-45 cuts pairs 25, 26 and
  27 (at 1048, 1397, 1863). Only 28 to 31 are never cut.
- The untidy ordering of the Stage 2 cliffs lines up with window edges, which I had flagged as a risk in
  the revisions and then not checked.
- "Cannot learn" rests on 4000 steps.
- Suggested and added: the dense Stage 2 model under the band tables; pair 23 mean-ablated alone; pairs 22
  to 27 cut singly at 45°; one 16,000-step hisa-90 run.

### Dense Stage 2 models under the tables (two seeds, 128 held-out sequences; `results/stage2_dense_masked.json`)

Offset recall, mean of the seeds, no retraining:

| removed | 2 | 4 | 8 | 16 | 32 | 64 to 1024 | content |
|---|---|---|---|---|---|---|---|
| 90–180° | 0.13 | 0.14 | 0.90 | 0.69 | 0.88 | 0.84 to 1.00 | 0.55 to 0.63 |
| 180–360° | 1.00 | 0.44 | 0.49 | 1.00 | 1.00 | 1.00 | 1.00 |
| 360–720° | 1.00 | 1.00 | 0.55 | 0.99 | 1.00 | 1.00 | 1.00 |
| beyond 720° | 1.00 | 1.00 | 1.00 | 0.49 | 0.01 | ≤ 0.01 | 1.00 |

hisa-180, 360 and 720 each leave content recall at 1.00 and offset recall at 0.01 from D = 32 on. hisa-90
and hisa-45 also damage content recall (0.77 to 0.94, and 0.01), and the shuffled control destroys both.

The prediction held, through a band I had not named: it is the parts beyond 720°, the ones that cost
+0.0007 on prose, that carry the far address. Each band bites only where pair 0 sits in it (1 rad per
token: 115°, 229°, 458°, 917° at D = 2, 4, 8, 16). I added single-pair cuts to check that (dated addition
in the protocol): pair 0 alone beyond 90° gives 0.11, 0.02, 0.58, 0.60, 0.98 at D = 2 to 32 and 0.40 to
0.80 beyond, very different between the seeds (0.33 against 0.79 at 64); pairs 1, 2 and 3 alone give 0.95
or better everywhere. So mainly pair 0, not only pair 0.

### Pair 23 (prose, 256 windows; `results/stage1_single_pair.json`)

- `pair-23-90-mean`: +0.063 against +0.475 for the plain cut. Most of what pair 23 does is a fixed function
  of distance. The means I saved for the zero-mean check show it: in layer 13 the pair-23 part averages -8
  to -16 score units at 1000 tokens in all nine heads and is still -3.5 to -5 at 2000. It is a penalty on
  far keys, and the 90° cut lifts it for every key beyond 1178. The reviewer's picture was the cosine
  crossing zero at 1178; in this layer the part is negative on both sides of that.
- Single pairs at 45°, cost past the window: pair 22 +0.023, 23 +1.585, 24 +0.387, 25 +0.060, 26 +0.018,
  27 +0.026. Pairs 25 to 27 are not like 23. Not every non-wrapping pair is a ruler here; 23 is, 24 partly.

### The long hisa-90 run (16,000 steps, 2 h 34 min)

Offset recall, 32 sequences in bf16 during training, final in fp32 on 128:

| step | 64 | 128 | 256 | 512 | 1024 |
|---|---|---|---|---|---|
| 4000 | 0.92 | 0.32 | 0.05 | 0.02 | 0.05 |
| 7000 | 0.99 | 0.95 | 0.77 | 0.68 | 0.05 |
| 9000 | 0.98 | 0.95 | 0.93 | 0.86 | 0.05 |
| 11,000 | 0.99 | 0.98 | 0.94 | 0.92 | 0.05 |
| final | 0.995 | 0.996 | 0.981 | 0.964 | 0.038 |

By the criterion I wrote down beforehand, "cannot learn" is retracted for 128 to 512 and stands only as
"not in 16,000 steps" for 1024. The 4000-step cliff was a snapshot of a slow learner. Dense needs under
1000 steps for all of it, so the cost is about a factor of ten in steps. At 1024 the wrong answers are
still near the target (559 of 687 within 32 positions).

The pre-registered confirming pattern was met at the pre-registered 4000 steps, and I reported it as
"cannot". That word was mine, not the protocol's, and it was wrong.

### Window edges (`results/stage2_edges.json`)

Accuracy at the 4000-step cliff against the nearest edges: hisa-180 at 32 (edge at 32) 1.00; hisa-45 at 32
(25, 34) 0.98; hisa-360 at 64 (63, 84) 0.94; hisa-90 at 64 (50, 67) 0.93 and 0.65; hisa-180 at 128 (100,
133) 0.86; hisa-90 at 32 (28, 38) 0.59 and 0.36; hisa-360 at 128 (112, 149) 0.36; hisa-90 at 128 (118, 158)
0.07. Near an edge is learned first. So the budget ordering in the Stage 2 figure is confounded with edge
positions, and I have no edge-free HiSA to compare. After the fact, few points.

### Verdict, revised

Still mixed, with the Stage 2 half restated: the dropped parts are where a dense model keeps the exact far
address; a hisa-90 model can still learn exact recall to 512 tokens, about ten times more slowly, and did
not learn 1024.
