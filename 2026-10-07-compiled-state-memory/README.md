# Can seeded states carry a memory set? (Qwen3.8-27B)

**Date:** 2026-10-07 – 2026-10-15 · **Machine:** `dgx-spark` (GB10, 128 GB unified memory)

## Brief

**Question.** Can one pass over a memory set produce a compact state that, seeded into an unchanged
model, carries general knowledge of that memory: knowledge usable for questions not known when the
state is built? The test is whether, once the memory is beyond trivial size, an agent seeded this
way answers better than the same agent with a text hint instead.

The seed must be derived from the whole memory and built before the question is known. It must be
fixed in size or clearly sublinear in the memory, and enter the model with no training, no fitted
reader, nothing fitted on test questions, and no raw or partial KV. Text hints (a keyword index, a
one-line note per file) are the baseline to beat, not a candidate.

Qwen3.8-27B is a hybrid with two places to seed. Its 48 Gated DeltaNet (GDN) layers have a
fixed-size recurrent state, which one compile pass produces for free. Its 16 global-attention layers
have a key–value cache (KV) that grows with the memory, so they need a derived seed. The study took
the question in three parts:
1. **The GDN state** (G1, a pilot, H1). Does it carry the memory, or tell an agent where to look?
2. **The attention layers** (Stage 0 and 1, stopped). Can a seed derived from the KV carry it
   compactly?
3. **Room above text hints.** Does the whole memory in context, the most any seed could carry, beat
   the best text hint on answer accuracy at ~110k and ~200k tokens?

The question in this form is the write-up's statement. I registered G1 (2026-10-07) and H1
(2026-10-08) under narrower questions about the GDN state. When the state showed no cue signal (H1
INCONCLUSIVE), I changed the question to the attention layers and set the seed conditions
(2026-10-10). When seed development kept deciding on proxies, I stopped it and added the
answer-accuracy test against text hints (2026-10-13).

A related entry, [2026-10-13](../2026-10-13-hybrid-attention-retrieval/), traces how recurrent
blocks, global-attention blocks and their MLPs contribute to recalling a historical association in
Qwen3.5-4B.

## Headline results

**Short answer: not answered.** The one compact candidate tested, the GDN state alone, holds
user-bound facts that the model reads only weakly, and it gave no cue on real histories. The
attention seed I built keeps much of the memory's effect on the facts it was fitted toward, but it
grows with the memory and was never scored on answers, so the full question was not tested. On 120
calibration questions at ~110k tokens, the whole memory in context beats the best text hint by 17
points: there is room above text hints, but whether a seed reaches it was not measured.

### 1. The GDN state: stored well, read weakly

- **Pre-registered gate (G1; 48 sealed items per block; 97.5% intervals).** The state carries
  user-bound facts across the session boundary: role-swap binding 0.39 nats [0.24, 0.53], all 48
  items positive, but only 5.0% [3.4, 6.6] of the in-context effect. The model's answer prefers the
  user's value in 54% of states (50% is chance). Preferences are 0.19 [0.11, 0.26], below the 0.25
  smallest effect of interest. Unrelated questions are not harmed.
- **Stored well (exploratory, synthetic development items).** A linear decoder on 16 heads recovers
  the user's value at 0.93–1.00, out to ~100k tokens written after the fact and with up to 128
  interfering facts. The model's own push is 0.5–1 nat against a ~2-nat prior gap.
- **Not a retrieval cue.** In a pilot, the state does not separate stores with the needed fact
  from stores without it in the agent's first-token tool-call log-odds (AUROC 0.51–0.58); a
  six-word topic list does (0.73–0.97). On LongMemEval_S (H1; 80 questions, ~108k-token real
  histories), swapping the evidence out moves those log-odds by +0.03 [−0.02, +0.08], and the first
  tool call is identical in all 80. The registered verdict is INCONCLUSIVE: the validity controls
  fell under the registered threshold on that measure.

### 2. The attention layers: a ceiling, and a seed that isn't compact (exploratory, stopped)

- **Stage 0** (16 development questions, ~108k tokens; gold-answer log p on the 9 with a short
  gold). With the memory's KV and GDN state loaded together, the gold answer's log p rises almost as
  much as with the memory in context: +10.5 nats [5.7, 15.2] against +11.1 [6.1, 16.0]. The KV
  alone, an off-distribution condition, gives +3.8.
- **A derived seed** (two development questions). Its keys are placed where self-study attention
  lands (self-study: questions the model writes about the memory without seeing it). It recovers
  0.67–0.77 of the KV-plus-state effect on generated facts it was fitted toward at 64× smaller than
  the KV, and 0.81–0.86 at 32× (216–229 MB). Facts the fit isn't aimed at get 0.26–0.46 at 64×.
- **Not compact, not scored.** At a fixed ratio the seed grows linearly with the memory, and it
  can't be built beyond the context window. No seed has been scored on answer accuracy.

### 3. Room above text hints: yes at ~110k (calibration screen, not confirmatory)

120 calibration questions, stores from LongMemEval_M, one agent design (BM25 search, 300-character
snippets, at most 8 calls). GO: the gap's 95% lower bound is above 0 and the gap is at least ≈ 0.08,
the smallest effect the 235-question test could detect. Labels are Qwen3.8-27B's, with eight blind
Claude agents' labels replacing them on the 480 decision answers.

| | ~25k tokens | ~110k | ~200k |
|---|---|---|---|
| whole memory in context (+ tools) | 0.91 | 0.88 | 0.83 |
| agent with memory tools, no hint | 0.75 | 0.72 | 0.69 |
| agent + keyword index | 0.73 | 0.71 | 0.68 |
| agent + one-line note per file | 0.74 | 0.70 | 0.73 |

- Text hints add no detectable accuracy: the best hint minus the bare agent is −0.01 to +0.04, with
  every 95% interval spanning 0.
- Whole memory beats the best text hint by +0.17 [+0.09, +0.24] at ~110k: GO under both judges. At
  ~200k the margin is +0.09 [+0.01, +0.18], GO by a single label (INCONCLUSIVE on Qwen's labels).
- The room is mostly in questions that combine several sessions. About half of the agents' wrong
  answers came after they had seen every evidence file (in a search snippet or a read).
- Both sizes fit in the context window, where the memory in context is itself a working
  alternative. A seed's own ceiling is the memory's KV plus GDN state, not the text, and no seed has
  been run against the room.

## Contents

| Path | What |
|---|---|
| [`REPORT.md`](REPORT.md) | Full write-up: method, results, limitations, verdict |
| [`LABNOTES.md`](LABNOTES.md) | Lab notes, condensed from the full log, in order, failures included |
| [`results/audit/LABNOTES_full.md`](results/audit/LABNOTES_full.md) | The full contemporaneous log (~42k words), every setup, rule, amendment and readout |
| [`PREREG.md`](PREREG.md) | Pre-registrations of G1 and H1, with dated deviations |
| [`LIT_REVIEW.md`](LIT_REVIEW.md), [`refs.bib`](refs.bib) | Literature review (written at the start, under the original GDN-state framing) and bibliography |
| [`code/`](code/README.md) | Harness, experiments, analyses, run scripts; environment, data sources and commands |
| `results/phase1/`, `results/pilot/` | Gate, probe and pilot records with their summaries |
| `results/phase2/` | H1 records and summaries, Stage 0 records, Stage 1 readouts, room check (`room/`: agent records, judge labels, second-judge items, key and labels, final readout) |
| `results/audit/` | Injection-suite and seed plumbing checks, throughput and overhead, the G1 smoke |
| `results/gdn_*` | Measured GDN retention (4B, 27B) and nominal half-lives |
| `images/` | Report figures |
