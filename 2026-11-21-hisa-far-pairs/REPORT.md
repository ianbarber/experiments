# What does HiSA give up?

## Question

With rotary position embeddings the attention score between a query and a key is a sum of 32 parts, one per
frequency pair, and pair l is rotated by (distance × θ_l). HiSA (arXiv 2611.26363) keeps a pair's part only
while that rotation is below a budget, 90° by default. The paper shows that models trained this way reach a
lower prose loss than dense ones and argues that dense models learn to suppress the dropped parts anyway.

Two readings fit the paper's evidence. The dropped parts are noise. Or they carry exact token-offset
information that prose almost never needs. The paper evaluates prose only; the readings differ on data
where token offsets matter.

Hypothesis, fixed before measuring ([PROTOCOL.md](PROTOCOL.md)): the dropped parts carry exact-offset
information; dropping them is nearly free on prose, costly where offsets are regular, and a model without
them cannot learn far exact-offset recall that the dense model can.

## Method

Model: HuggingFaceTB/SmolLM-135M (30 layers, 9 query heads over 3 key heads, 32 pairs per head, RoPE base
10000), context 2048, fp32 for evaluation.

Every condition is a table M[pair, distance] saying which parts are kept. The implementation is exact and
was checked against the paper's reference code (Stage 0: pair layout, frequencies, identity with the stock
model to 2e-4 in logits, agreement with the paper's reference to 6e-5).

**Stage 1, drop at inference, no training.** Conditions:

- `hisa-Φ` for Φ = 45, 90, 180, 360, 720°.
- `band-a-b`: drop only the parts rotated between a and b.
- Controls that drop the same number of pairs as hisa-90 at every distance: `ctrl-perm` (the 32 hisa-90
  windows shuffled over the pairs, 3 seeds), `ctrl-iid` (random pairs at each distance), `ctrl-slow` (the
  windows in reverse order, so the slowest pairs are cut first).
- `hisa-90-mean`: dropped parts replaced by their mean over queries, per layer, head, pair and distance,
  measured on 64 held-out prose windows. Added because the dropped parts do not average to zero.
- Added after the first pass: `-keep1` (the first key keeps every pair) and `pair-l-90` (only pair l is cut).

Data, 2048-token windows, identical across conditions: 256 OpenWebText documents; 256 Python files; 256
windows of real CSV, JSON-lines and markdown tables; synthetic grids of C single-token cells per row and a
newline, C from 8 to 256, 64 windows each; 256 prose windows with a 32-token span repeated after a gap of
64 to 1536. Differences are paired by window with a bootstrap 95% interval.

A grid size counts only if the dense model puts at least 0.5 probability on the newline at its true
position.

**Stage 2, can HiSA learn exact offsets?** A 2048-token stream of random words with two kinds of query
embedded in it, about 105 per sequence. Offset: a marker for D, then the token exactly D positions before
the marker. Content: a marker, a key that occurred once D positions earlier, then the token that followed
it. D from 2 to 1024 (content from 4). Pretrained SmolLM-135M, all weights tuned, loss on answer tokens,
4000 steps of batch 8 at 1e-4, mask on from step 0. Variants: dense and hisa-90 (two seeds), hisa-45,
hisa-180, hisa-360 (one seed). Accuracy on 128 held-out sequences, about 675 queries per variant and D.

**Added after a review of the first write-up** (Amendment 3 of the protocol, predictions written before
running): one hisa-90 run of 16,000 steps; the two fine-tuned dense models evaluated with rotation bands and
single pairs removed at inference; pair 23 mean-ablated alone; pairs 22 to 27 cut singly beyond 45°.

## Results

### 1. HiSA learns far exact offsets many times more slowly than dense, and not at 1024

![Stage 2](images/stage2_accuracy_by_distance.png)

*Accuracy on answer tokens after fine-tuning, by distance D, dense against four HiSA budgets at 4000 steps
and hisa-90 at 16,000 steps (128 held-out sequences, about 675 queries per point). Solid: offset recall.
Dashed: content recall (1.00 everywhere). Dense and hisa-90 at 4000 steps are means of two seeds; small dots
are the seeds. Look at where each solid line leaves 1.0, and at how far the open squares move it.*

| variant, steps | 16 | 32 | 64 | 128 | 256 | 512 | 1024 |
|---|---|---|---|---|---|---|---|
| dense, 4000 (2 seeds) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| hisa-45, 4000 | 0.956 | 0.979 | 0.634 | 0.065 | 0.074 | 0.023 | 0.034 |
| hisa-90, 4000 (2 seeds) | 0.998, 0.998 | 0.586, 0.356 | 0.933, 0.653 | 0.074, 0.077 | 0.033, 0.026 | 0.028, 0.041 | 0.032, 0.032 |
| hisa-180, 4000 | 0.991 | 0.999 | 0.859 | 0.859 | 0.071 | 0.026 | 0.038 |
| hisa-360, 4000 | 1.000 | 0.993 | 0.943 | 0.361 | 0.206 | 0.091 | 0.028 |
| hisa-90, 16,000 (1 seed) | 1.000 | 0.999 | 0.995 | 0.996 | 0.981 | 0.964 | 0.038 |

All variants are at 0.997 or better for D = 2, 4, 8, and content recall is 0.998 or better for every
variant at every distance.

The dense model reaches ceiling at every distance within 1000 steps. Exact addressing at 1024 tokens is not
hard for a RoPE model.

At 4000 steps, the budget fixed in the protocol, every HiSA variant has a cliff: 1.00 to D = 16 and a few
percent from 128 or 256 on. The first write-up read that as "cannot learn". The 16,000-step run shows it was
mostly speed. hisa-90 passes 0.9 at D = 128 by step 7000, at 256 by step 9000 and at 512 by step 11,000
(checks every 1000 steps on 32 sequences), and ends at 0.96 or better up to 512. D = 1024 does not move: it
is between 0.02 and 0.05 at every check and 0.04 at the end. So for this task a hisa-90 model needs roughly
7 to 11 times the dense model's steps for 128 to 512 tokens, and has not learned 1024 in 16 times. This is
one seed, and the longer run has a longer learning-rate schedule, so it is not a continuation of the
4000-step run.

How HiSA gets exact offsets at all is not tested here. Its hard window edges are the obvious candidate: a
pair's part is cut at a fixed distance, which marks that distance exactly. The 4000-step numbers fit that.
Accuracy at the cliff follows how close D is to an edge of that variant, not the budget:

| variant at D | nearest edge at or before D, and after | accuracy at 4000 steps |
|---|---|---|
| hisa-180 at 32 | 32, 42 | 0.999 |
| hisa-45 at 32 | 25, 34 | 0.979 |
| hisa-360 at 64 | 63, 84 | 0.943 |
| hisa-90 at 64 | 50, 67 | 0.933, 0.653 |
| hisa-180 at 128 | 100, 133 | 0.859 |
| hisa-90 at 32 | 28, 38 | 0.586, 0.356 |
| hisa-360 at 128 | 112, 149 | 0.361 |
| hisa-90 at 128 | 118, 158 | 0.074, 0.077 |

This is a reading of existing numbers after the fact, with a handful of points. It does mean the budget
ordering in the figure is confounded with where the edges happen to fall, and that a HiSA variant with soft
edges could be worse than these. For hisa-90 at D = 1024 the nearest edges are 884 and 1178.

Where HiSA fails it is nearly right. At the 4000-step cliff most wrong answers are the token one position
off (hisa-90 at D = 32: 268 of 301 and 337 of 468 in the two seeds). At D = 1024 after 16,000 steps, 559 of
687 wrong answers are tokens within 32 positions of the target.

### 2. In the dense model the far address is in the parts that are free on prose

![Dense under bands](images/stage2_dense_under_bands.png)

*Accuracy of the dense model fine-tuned on the recall task when one rotation band is removed at inference,
no retraining (two seeds, small dots; 128 held-out sequences). Solid: offset recall. Dashed: content
recall, which only the 90–180° band disturbs. Look at the darkest line: removing the parts rotated beyond
720° ends exact recall from D = 32 on.*

| removed at inference | 2 | 4 | 8 | 16 | 32 | 64 to 1024 | content recall |
|---|---|---|---|---|---|---|---|
| nothing | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 90–180° | 0.13 | 0.14 | 0.90 | 0.69 | 0.88 | 0.84 to 1.00 | 0.55 to 0.63 |
| 180–360° | 1.00 | 0.44 | 0.49 | 1.00 | 1.00 | 1.00 | 1.00 |
| 360–720° | 1.00 | 1.00 | 0.55 | 0.99 | 1.00 | 1.00 | 1.00 |
| beyond 720° | 1.00 | 1.00 | 1.00 | 0.49 | 0.01 | 0.01 or less | 1.00 |
| pair 0 alone, beyond 90° | 0.11 | 0.02 | 0.58 | 0.60 | 0.98 | 0.40 to 0.80 | 1.00 |
| pair 1, 2 or 3 alone, beyond 90° | 0.99+ | 0.95+ | 1.00 | 0.99+ | 1.00 | 0.95+ | 1.00 |

Means of two seeds. The parts rotated beyond 720° cost +0.0007 nats per token on prose in the pretrained
model (section 3). In the model trained to recall by exact offset they carry the whole address from 32
tokens on, and content recall does not notice their absence. That is the two readings of the paper's
evidence side by side: the same parts are free where nothing needs an exact far offset and essential where
something does.

Each band hurts only at the distances where the fastest pair sits in it (pair 0 turns one radian per token:
115° at D = 2, 229° at 4, 458° at 8, 917° at 16). Cutting pair 0 alone beyond 90° removes much of the offset
recall at most distances, unevenly and differently in the two seeds; cutting pair 1, 2 or 3 alone does
nothing. So the address rests mainly on the fastest pair, with help from others, since the band removals do
more than the single pair. This is no retraining, so it shows what this dense model uses, not what a model
needs; section 1 is the evidence on that.

### 3. Without retraining, hisa-90 is not free on this checkpoint

![Stage 1 by data type](images/stage1_cost_by_data.png)

*Extra loss against dense for hisa-90, its mean-ablation variant and three controls that drop the same
number of pairs, log scale, 256 windows per data type. Bars: mean; lines: bootstrap 95% interval (for the
shuffled control, the range over 3 seeds). Look at the gap between hisa-90 and the controls, and at how
little the three data types differ.*

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
| ctrl-perm (3 seeds) | +2.23 to +4.57 | +3.17 to +5.44 | +2.08 to +4.24 |
| ctrl-iid | +4.16 | +5.57 | +4.13 |
| ctrl-slow | +5.08 | +6.14 | +5.00 |

- hisa-90 costs 0.22 nats on prose, against 0.031 in the paper for its own 4K model. The comparison is
  loose (different model, different context), but this checkpoint leans on the dropped parts more.
- Which pairs are dropped matters far more than how many. The same count taken from anywhere else costs 2
  to 6 nats (10 to 23 times the hisa-90 cost on prose and code, 25 to 60 times on tables); taken from the
  slow end it is worst.
- Dropping only the 90–180° band is three times worse than dropping it together with everything beyond.
  The parts beyond 180° are not independent of the band below them.
- Replacing the dropped parts by their mean does not help. Most of the energy of the dropped parts is a
  fixed function of distance (93% in the 90–180° band), but that is not what the model misses.
- Code and tables do not stand out from prose. On tables the cost per newline token is +0.16 against +0.10
  on other tokens; on code newlines cost less than other tokens.
- Copy accuracy on the repeated span: dense 0.83, hisa-90 0.53, hisa-180 0.76, hisa-360 0.83. Restricted to
  repeats before position 1100: 0.85, 0.71, 0.85, 0.86, and for hisa-90 it falls with the gap (0.84 at 64,
  0.50 at 512). The paper's copy result is for trained HiSA models; this is the untrained switch.

### 4. Nearly all of the prose cost is two pairs

![Cost by position](images/stage1_cost_by_position.png)

*Extra loss against dense by position in the window, for the whole hisa-90 rule and for single pairs cut
beyond 90° (prose, 256 windows, 64-token bins). Dotted lines: the distances at which pairs 22, 23 and 24
pass 90°. Look at where the hisa-90 line leaves the floor, and at the pair-23 line above it.*

The cost of hisa-90 is flat at +0.02 to +0.04 for the first 1150 positions and then climbs to +0.9, with a
second rise near 1570. Those are the 90° windows of pairs 23 and 24 (1178 and 1571 tokens).

| only this pair cut beyond 90° | 18 | 19 | 20 | 21 | 22 | 23 | 24 |
|---|---|---|---|---|---|---|---|
| extra loss on prose | +0.003 | +0.001 | +0.006 | +0.002 | +0.011 | +0.475 | +0.070 |

Cutting pair 23 alone costs more than twice the whole hisa-90 rule; the other dropped pairs partly offset
it. Pair 22 spends positions 884 to 1767 between 90° and 180° and costs 0.01, so this is about the pair,
not the angle band. A first guess, that the first token of the window is the key involved, was wrong:
letting the first key keep every pair changes nothing (3.091 against 3.086).

The pair-23 cost sits in layer 13: cut there alone it costs +0.49 over positions from 1280 on, against
+0.03 or less in any other layer (64 windows). Inside layer 13 no single head carries it (largest +0.06,
all nine sum to +0.15).

What pair 23 does there is mostly a fixed function of distance. Replacing its cut part by the per-distance
mean, measured on 64 other prose windows, brings the cost from +0.475 to +0.063. In layer 13 that mean is a
penalty on far keys: -8 to -16 score units at 1000 tokens in every head, still -3.5 to -5 at 2000. Cutting
the pair beyond 1178 removes the penalty, and keys more than 1178 tokens back gain that much score at once.
I did not look at where the attention then goes.

It is not a property of every slow pair. Cut singly beyond 45°, over the positions past each pair's window:

| pair (45° window) | 22 (442) | 23 (589) | 24 (786) | 25 (1048) | 26 (1397) | 27 (1863) |
|---|---|---|---|---|---|---|
| extra loss past the window | +0.023 | +1.585 | +0.387 | +0.060 | +0.018 | +0.026 |

Pairs 25 to 27 behave like pair 22, not like pair 23. Pair 27 has only 185 positions past its window.

This qualifies the cheap hisa-180 result. At 180° the windows of pairs 23 and 24 are longer than 2048
tokens, so that rule never cuts them here. Pairs 28 to 31 are never cut at 2K under any budget tested.

### 5. Short exact offsets use the 90–180° parts

![Grids](images/stage1_grids.png)

*Probability the model gives the newline at its true position on synthetic grids, by cells per row, dense
against four ways of dropping parts (64 windows per size; rows after the third). Only C = 8 passes the
gate. Look at hisa-90 against dense and hisa-360 there.*

The dense model passes the gate only at C = 8 (0.53; 0.45 at C = 16, 0.14 at 32, nothing from 64 on). At
C = 8, hisa-90 lowers the probability to 0.21 and raises the newline loss by 1.02 nats [0.98, 1.05];
hisa-180 and hisa-360 do not hurt (0.58 and 0.53). The effect is the same before position 1100 (0.51 to
0.20) and after 1280, so it is separate from pair 23. C = 16, which misses the gate, shows the same pattern.
The pretrained model does not place newlines at offsets beyond 33 tokens at all, so Stage 1 cannot speak to
far exact offsets; Stage 2 does.

## Limitations

- One model, 135M parameters, 64-dimensional heads, context 2048. At 90° the seven slowest pairs are never
  cut; under any budget tested, the four slowest.
- Stage 1 has no retraining: it measures what this dense checkpoint uses, mixed with distribution shift.
  The paper reports that continued training recovers the loss on prose; I did not continue training on
  prose, so the +0.22 is not a claim about a trained HiSA model.
- Stage 2 is one synthetic task and one learning rate chosen on dense, with two seeds for dense and hisa-90
  at 4000 steps and one for the other budgets. The 16,000-step result is one seed and one budget; whether
  D = 1024 would be learned with more steps, and how the other budgets behave with more steps, is not known.
- Window edges and budget are confounded in Stage 2; no soft-edged HiSA variant was trained.
- The band and single-pair removals on the fine-tuned dense model are without retraining.
- The grid probe passes its gate at one size only.
- The pair-23 finding is located (layer 13) and described as a distance penalty from saved means. I did not
  inspect attention patterns.
- OpenWebText has no official validation split; the prose windows are the first long documents of the
  stream.

## Verdict

Mixed, scoped to SmolLM-135M at 2K.

The dropped parts are not noise. A dense model trained to recall by exact offset puts the far address in
parts rotated beyond 720°, mainly those of the fastest pair, and the same parts cost nothing measurable on
prose. That is the "rarely needed capability" reading.

The hypothesis also said a model without those parts cannot learn far exact-offset recall. At the 4000
steps fixed in the protocol the confirming pattern is there: dense at 1.00, hisa-90 at 0.08 or less from 128
tokens on, content recall untouched. Given 16,000 steps hisa-90 reaches 0.96 or better out to 512 tokens and
stays at 0.04 at 1024. So "cannot" is wrong as stated. What HiSA gives up on this task is speed, roughly a
factor of ten in training steps, and the longest distance tested. Its window edges probably supply the
exact positions it does learn.

Against the sketches drawn before the runs: Stage 2 at 4000 steps is the "hypothesis holds" sketch with an
ordering of the budgets that follows window edges; the 16,000-step line moves most of the way to the
"rejected" sketch. Stage 1 is neither sketch: the prose bar is far larger than drawn, code and tables do not
exceed prose, and of the grids only C = 8 could be scored (+1.02 nats on its newline tokens).

"Nearly free on prose" does not hold for this checkpoint at 90° without retraining: +0.22 nats, almost all
of it from two slow pairs in the second half of the window, +0.03 elsewhere. Beyond 180° the parts are
close to free on prose, code, tables and the one usable grid, for the pairs that 2K can reach.
