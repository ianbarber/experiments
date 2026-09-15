# Lab notes — failure-conditioned repair, paused

**Machine:** `dgx-spark`. The pairing pilot ran 9 Sep; GPU competence work
began 10 Sep.

## 9 Sep — diagnostic pilot

Ran the pairing experiment as specified in [code/RESEARCH_PLAN.md](code/RESEARCH_PLAN.md)
and [code/PROTOCOL.md](code/PROTOCOL.md). Collected 1,007 failures; they
collapsed to four generic strings. 19% of accepted pairs byte-identical.
Kept the 1.27 pp reactive-vs-shuffle effect, below the 5-point bar. Did not
recode the 0/288-valid prospective arm as a success. Direct correction
learned to REPORT. [Pilot report](STAGE1.md); notes in
[STAGE1_LABNOTES.md](STAGE1_LABNOTES.md).

The next two stages tried to build a vehicle that could identify the same
hypothesis on the same 3B base.

## 10 Sep — competence factorial; repair canceled

Ran a 2×2 (answer order × CLEAR weight) on a ledger-audit task, two seeds,
384 shared validation cases. Design review caught, before any repair
outcome:

- a wrapper cue that predicts 448/512 proposed repair-training decisions
- donor swap that also swaps case facts

Canceled induction and repair during training. Completed the eight
competence validations. Decision-last helped; that result is not a repair
finding. [Competence](competence/).

## 12–13 Sep — calibration, then pause

Asked whether the 3B setup could install a selective, wording-robust
mistake — a prerequisite, not the hypothesis. First seed: familiar-view
induction worked, rewritten-view gates all failed (0/18). Second seed: one
exact-boundary pass, no recipe in both seeds. Progression rule failed.
Stopped before reflection collection.

Paused 13 Sep. [Calibration](calibration/).
