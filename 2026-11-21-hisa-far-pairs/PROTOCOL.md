# What does HiSA give up? Protocol

Revised plan, fixed on 2026-11-21 after the Stage 0 checks and before any Stage 1 or Stage 2 measurement.
Later changes are dated amendments at the end.

## Revisions to the original plan

1. **The paper already reports part of Stage 1.** Its Section 5 measures the influence of every pair by distance
   band in this model on OpenWebText (the 0.5% figure is parts beyond 720° relative to parts below 45°), and its
   Table 6 gives the cost of switching without training (2.906 → 2.937) for the authors' own 4K model at 4K
   context. Stage 1 adds loss on non-prose data, matched controls, and the 600B-token model at 2K. I do not
   re-implement the paper's Eq. 20; the prose band ablations are the consistency check (cost should fall with
   rotation and be smallest beyond 720°). The 0.031 is a loose yardstick only: different model, and at 2K fewer
   parts are dropped (37% of products kept against 29% at 4K).
2. **Random control.** "Same number of pairs dropped per distance, chosen at random" has two readings.
   `ctrl-perm` (primary, 3 seeds) assigns the 32 hisa-90 windows to the pairs by a random permutation: each
   pair still has one contiguous window. `ctrl-iid` (one seed) draws the dropped set independently at every
   distance, which also makes neighbouring keys lose different pairs, a second kind of damage. `ctrl-slow` is
   the reversed assignment (slowest pair gets the shortest window).
3. **HiSA's window edges are exact-distance cues.** Pair l's part is cut to zero at d = ⌈w_l⌉, and the part
   need not be small there, so a HiSA model can tell d < ⌈w_l⌉ from d ≥ ⌈w_l⌉ exactly. At 90° the edges are
   2, 3, 4, 5, 7, 9, 12, 16, 21, 28, 38, 50, 67, 89, 118, 158, 210, 280, 373, 497, 663, 884, 1178, 1571. The
   "resolution about D/3" argument therefore gives the width of the band HiSA can isolate for free, not a hard
   limit, and offsets 2 to 4 are single-offset bands. Stage 2 records where wrong answers come from (offset of
   the token the model named) so that band-limited errors are visible.
4. **Stage 2 task fixes.** (a) A 2048-token stream over 256 words repeats every word about 8 times, so the
   content twin needs keys that are unique in the sequence; they come from a separate 512-word key vocabulary.
   (b) The loss is taken on answer tokens only: the stream is unpredictable and its loss is gradient noise.
   (c) About 110 queries per sequence instead of a few dozen. (d) Offset targets never fall on a key, so a key
   is never duplicated before its query.
5. **Zero-mean check is done up front** and is cheap: the mean of a pair's part over all queries at a fixed
   distance is a cross-correlation, computed by FFT. If the dropped parts have a clearly non-zero mean, a
   mean-ablation variant of hisa-90 (dropped parts replaced by their per layer, head, pair and distance mean on
   held-out prose) is added to Stage 1.
6. **Implementation.** The 32-matmul sketch costs about 4 s per window. The table attention used here is exact
   and blocked (per block distance, pairs with a constant keep value go through one matmul): 0.19 s per window.
   Training uses the paper's Triton exact kernel (forward and backward; logits agree with the table attention
   to 3e-4 in fp32).
7. **Sizes.** Grids use 64 windows per C (each has 8 to 227 rows). Stage 2 uses batch 8, 4000 steps.
8. Stage 3 is not part of this protocol. It is run only if Stage 1 shows a grid effect to localise.

## Hypothesis

In SmolLM-135M the score parts that HiSA drops (pair rotation beyond 90°) carry exact token-offset
information that prose rarely needs: dropping them is nearly free on prose, costly where token offsets are
regular, and a model without them cannot learn far exact-offset recall that the dense model can.

Confirms: Stage 1 shows a hisa-90 cost on gated grids' newline tokens that is several times the prose cost;
Stage 2 shows offsets D where dense is ≥ 0.9 accurate and hisa-90 ≤ 0.5 while content recall is ≥ 0.9 for
both, and the hisa-360 cliff is further out than the hisa-90 cliff.

Rejects (noise reading): hisa-90 cost on gated grids no larger than on prose, and in Stage 2 hisa-90 within
0.05 of dense at every D where dense is ≥ 0.9.

Anything else is reported as mixed, with which half held.

## Setup

- Model: HuggingFaceTB/SmolLM-135M (30 layers, 9 query heads over 3 KV heads, head dim 64, 32 pairs,
  rope_theta 10000, 2048 positions). Context 2048. fp32 for all evaluation.
- Every condition is a keep-table M[pair, distance]; keys at a distance where no pair is kept leave the
  softmax (the paper's convention; it never happens at 2K for the tables here).

## Stage 0: sanity checks (done, all pass; `results/stage0_checks.json`)

Pair layout (split-half; the interleaved pairing fails the norm test), rotation angle = position × θ_l,
inv_freq = 10000^(-l/32), all-true table reproduces stock logits (2e-4), tables and outputs agree with the
paper repo's reference and its `patch_llama(impl="reference")`, controls drop the same count per distance.

## Stage 1: drop at inference, no training

| Name | Keeps pair l at distance d when |
|---|---|
| dense | always |
| hisa-45, 90, 180, 360, 720 | d·θ_l < budget |
| band-90-180, 180-360, 360-720, 720+ | rotation is outside that band |
| ctrl-perm-s0..2 | hisa-90 windows permuted over pairs |
| ctrl-slow | hisa-90 windows in reverse order |
| ctrl-iid-s0 | hisa-90 count per distance, pairs drawn independently per distance |
| hisa-90-mean | only if the zero-mean check fails: dropped parts replaced by their mean |

Data, 2048-token windows, the same windows for every condition:

- Prose: 256 OpenWebText documents of at least 2048 tokens (first 2048 tokens; the dataset has no official
  validation split, so these are the first such documents of the stream).
- Code: 256 Python files of at least 2048 tokens (codeparrot-clean-valid).
- Tables: 256 windows of real tabular text: 85 CSV, 86 JSON lines, 85 markdown tables (vega-datasets and
  seaborn-data; the JSON-lines and markdown windows are re-serialisations of real records).
- Grids: rows of C single-token cells and a newline, cells uniform over 256 words, C in 8…256, 64 windows
  each. The text round-trips through the tokenizer to exactly these tokens.
- Copy: 256 prose windows with a 32-token span repeated after a gap of 64, 128, 256, 512, 1024 or 1536.

Metrics: mean loss (nats per token) and the paired difference to dense with a bootstrap 95% interval over
windows (10,000 resamples); the difference by position (0–512, 512–1024, 1024–2048) and by token class
(newline, delimiter, other) for code and tables; for grids the loss and probability of the newline at its
true position, whether newline is the top prediction there, the newline probability at offsets ±1…3, and the
loss on the first cell of a row; for copy the top-1 accuracy on tokens 5–32 of the repeated span, by gap.

Gate: a grid size C counts only if the dense model's mean newline probability at the true position is at
least 0.5 on rows after the third. Sizes that fail are reported as "dense cannot do it" and not interpreted.

## Stage 2: can HiSA learn exact offsets?

Sequence of 2048 tokens. Stream tokens are uniform over 256 words. Two query types are embedded in it:

- Offset: `[marker_D] [answer]`, answer = the token exactly D positions before the marker, one marker token
  per D in {2, 4, 8, 16, 32, 64, 128, 256, 512, 1024}.
- Content: `[cmarker] [key] [answer]`, where the key is a word that occurs exactly once earlier, exactly D
  positions before its copy, and the answer is the token that followed it.

About 55 queries of each type per sequence, D uniform over the ten values, position uniform where feasible.
Loss on answer tokens only. Chance is about 1/256.

Runs: pretrained SmolLM-135M, all weights tuned, the mask on from step 0. AdamW (β 0.9/0.95, weight decay
0), batch 8, 4000 steps, 200 warmup steps then cosine to 10%, bf16 autocast, gradient clip 1.0. Learning
rate chosen on dense from {1e-4, 3e-4} by a 1000-step pilot and reused. Variants: dense and hisa-90 (seeds
42 and 43), hisa-360, hisa-45, hisa-180 (seed 42). Evaluation: 128 held-out sequences, fp32, accuracy on
answer tokens by type and D, and for wrong offset answers the offset of the stream token the model named.

Read-out: one figure, accuracy against D, solid for offset recall and dashed for content recall, one colour
per variant.

## Kill criteria

- Any Stage 0 check fails: stop.
- Stage 2 is "not installed" if dense, at the better learning rate, ends below 0.9 offset accuracy at D ≤ 8
  or below 0.8 content accuracy averaged over D. Then Stage 2 is reported as "could not test" and no other
  synthetic task is substituted. If dense learns near offsets but not far ones, that is the finding the
  original plan names (where dense gives out), not a kill.
- If no grid size passes the gate, Stage 1 makes no claim about exact offsets and the question rests on
  Stage 2 alone.

## Sketches (made-up data)

Headline figure, Stage 2, accuracy against D (o = offset recall, c = content recall):

```
 if the hypothesis holds                      if it is rejected
 1.0 |c c c c c c c c c c   all variants      1.0 |c c c c c c c c c c
     |o o o o o o o o .     dense                 |o o o o o o o o . .   dense
     |o o o o o o .         hisa-360              |o o o o o o o o . .   hisa-360
     |o o o o .             hisa-90               |o o o o o o o o . .   hisa-90
 0.0 |________._._._._._                      0.0 |__________________
      2 4 8 16 .. 512 1024                         2 4 8 16 .. 512 1024
```

Stage 1, extra loss from hisa-90 against dense (nats per token; grid = newline tokens):

```
 if the hypothesis holds                      if it is rejected
 prose   |#                                   prose   |#
 code    |##                                  code    |#
 tables  |###                                 tables  |#
 grid-16 |##########                          grid-16 |#
 grid-64 |################                    grid-64 |
 copy    |                                    copy    |
```

## Limits known in advance

One model, 64-dim heads, 135M parameters, context 2K. The slowest seven pairs never drop at 2K (their 90°
window exceeds 2048). Stage 1 mixes "information used" with distribution shift; Stage 2 does not.

## Amendments

**Amendment 1 (2026-11-21, after the first Stage 1 pass).** The cost of hisa-90 on prose switches on at
fixed positions (about 1178 and 1571), which neither sketch anticipated. Added diagnostic conditions
`<table>-keep1` and `-keep4` (the first 1 or 4 keys of the sequence keep every pair) to test whether the
first token is responsible. It is not.

**Amendment 2 (2026-11-22).** Added `pair-<l>-90` (one pair cut beyond 90°, the rest dense) for pairs 18 to
24 on prose, and a localisation of the pair-23 cut by layer and by head (`code/localise_pair.py`, 64 prose
windows). This takes the place of the optional Stage 3 of the original plan, aimed at pair 23 instead of
the grids. Also added: grid and copy metrics split into positions before 1100 and after 1280, to separate
them from the pair-23 effect.

**Outcome of the pre-declared branches.** Zero-mean check failed, so `hisa-90-mean` was run. Grid gate:
only C = 8 passes. Stage 2 learning rate: 1e-4 (pilot: both rates reach 1.00). Kill criteria not met.
