# What does HiSA give up? The exact far address, slow to relearn, and two slow pairs (SmolLM-135M at 2K)

**Date:** 2026-11-21 · **Machine:** `local-3090` (one RTX 3090, 24 GB)

## Brief

HiSA (arXiv 2611.26363) drops the part of the attention score that RoPE pair l contributes once that
pair's rotation between query and key passes 90°, and argues that dense models learn to suppress those
parts anyway. Are the dropped parts noise, or exact token-offset information that prose rarely needs? I
tested this on pretrained SmolLM-135M at 2048 tokens in two ways: dropping the parts at inference on prose,
code, tables, synthetic grids and a copy probe, with matched controls; and fine-tuning dense and HiSA
variants on a task where the exact offset is the only route to the answer.

## Headline results

- **Dense learns exact-offset recall at once; HiSA learns it about ten times more slowly, and not at
  1024 tokens.** Dense recalls the token exactly D positions back at 1.00 for every D from 2 to 1024 within
  1000 fine-tuning steps. At 4000 steps hisa-90 is at 1.00 to D = 16 and at 0.08 or less from 128 on (two
  seeds). One hisa-90 run of 16,000 steps reaches 0.96 or better out to D = 512 and stays at 0.04 at 1024.
  Content recall at the same distances is 1.00 for every variant. The first write-up said HiSA "cannot
  learn" this; the longer run, added after review, retracts that.
- **In the dense model the far address sits in parts that are free on prose.** Removing only the parts
  rotated beyond 720° from the fine-tuned dense model, with no retraining, takes exact recall from 1.00 to
  0.01 for every D from 32 on and leaves content recall at 1.00 (two seeds). The same removal costs +0.0007
  nats per token on prose. The address rests mainly on the fastest pair.
- **Where HiSA's cliff falls after 4000 steps follows its window edges, not its angle budget.** A cut at a
  fixed distance marks that distance exactly, and accuracy at the cliff tracks how near D is to one
  (hisa-180 at D = 32, edge at 32: 1.00; hisa-90 at 32, edges at 28 and 38: 0.59 and 0.36). This is a
  reading after the fact.
- **Without retraining, hisa-90 is not free on this checkpoint:** +0.218 nats per token on prose [0.208,
  0.230], +0.267 on code, +0.083 on tables (256 windows each), against the paper's +0.031 for its own
  model. On prose it is still 10 to 23 times cheaper than dropping the same number of pairs chosen any other
  way (+2.2 to +5.1).
- **Nearly all of that cost is two pairs.** It is +0.03 until position 1178 and then climbs to +0.9. Cutting
  pair 23 alone costs +0.475, more than the whole rule; pair 24 alone +0.070; pairs 18 to 22 at most +0.011
  each. In layer 13, pair 23 is a fixed penalty on keys far back (-8 to -16 score units at 1000 tokens), and
  the cut removes it: putting back its per-distance mean brings +0.475 down to +0.063. Slower pairs (25 to
  27, cut at 45°) do not behave this way.
- **Beyond 180° the dropped parts are close to free on everything tested without retraining** (hisa-180:
  +0.009 or less), but at 2K that rule never reaches pairs 23 and 24.
- **Short exact offsets use the 90–180° parts.** On the one grid size the dense model can do (8 cells per
  row), hisa-90 cuts the probability of the newline at the right place from 0.53 to 0.21; hisa-180 and
  hisa-360 leave it alone.
- Verdict: mixed. The dropped parts are a rarely needed capability, not noise. Giving them up costs speed
  and the longest distance on exact-offset recall, not the ability. "Nearly free on prose" does not hold
  for this checkpoint at 90°. One model, one context length, one seed for the long run.

![Accuracy by distance after fine-tuning on exact-offset recall](images/stage2_accuracy_by_distance.png)

*Accuracy on answer tokens after fine-tuning pretrained SmolLM-135M, by distance D to the token to recall
(128 held-out sequences, about 675 queries per point; dense and hisa-90 at 4000 steps are means of two
seeds, small dots are the seeds). Solid lines: recall the token exactly D positions back. Dashed: recall
the token after a unique key D positions back, 1.00 for every variant. Look at where each solid line
leaves 1.0, and at the open squares: the same hisa-90 model given four times the steps.*

## Contents

| File | What |
|---|---|
| [REPORT.md](REPORT.md) | Method, results, limitations, verdict |
| [PROTOCOL.md](PROTOCOL.md) | The plan as revised before Stage 1, sketches, kill criteria, three dated amendments |
| [LABNOTES.md](LABNOTES.md) | Execution log in order, including the first-token guess and the "cannot learn" reading, both retracted |
| [code/](code/) | Keep-table attention, checks, evaluation, task, training, summaries, figures; see `code/README.md` |
| [results/](results/) | Stage 0 checks, Stage 1 summary, Stage 2 logs and summary, dense model under bands, single-pair cuts, window edges |
| [images/](images/) | Figures |
