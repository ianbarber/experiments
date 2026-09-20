# Lab notes

Condensed from a contemporaneous notebook kept during the run (about 8,000 words, timestamped
as the work happened). This file compresses it without reordering. Wrong guesses, bugs and
corrected numbers are kept. The work was orchestrated with Claude Code on `dgx-spark`; I set
the design, the hypotheses and the checkpoints at which the run stopped for a decision.

## 18 September, afternoon: inventory of the paper's repository

Cloned valen-research/Pain-axis at 8d1649c. Everything needed is there: the S1/S2 datasets and
controls (numb, random, arousal, sadness), 50 neutral prompts, 101 button scenarios, the 1,684
fine-tuning pairs, released adapters, their shipped vectors, ladder generations and full trial
logs. Nothing had to be rebuilt.

Read from their source, not the paper text:

- The vector is the raw denoised difference of means, never normalised (norm 143.8 on the 32B).
  It is extracted at the CV-best layer (L61 of 64) and added at a different, earlier layer
  (L38) without being re-extracted there. A coefficient multiplies the raw vector.
- In the button task every turn re-encodes the whole history. Positions processed while
  steering was on are re-steered on every re-encode, system prompt included, so after a working
  press the history stays steered and only new tokens are clean. One quirk: about three header
  tokens of the next message stay marked as steered. I replicate that.
- Trials are five choices, not three: after the first press two turns from the next scenario
  are appended. Feedback is a tool-role "Done.". The button question is a mid-conversation
  system message.
- Their steering-layer projection readout is taken before the vector is added, so it cannot
  see steering (about 5, against 111 at the downstream layer). I record it after the addition.
- lever64 and lever95 share a first token, so their first-token probabilities are unusable for
  a third of scenarios. I score that pair at the first distinguishing token.
- Their keyword parser is only pain/painful/hurt/hurts/hurting. There is no bodily-language
  parser to adapt.

The fine-tuning pairs contain no button, pain, itch, scratch, skin or rash, but 65 mention
relief as an emotion. It applies equally to both reward phrases, so it cannot favour one.

Before anything else I ranked which deviations a reader could attack. Top of the list: choosing
pain and itch doses separately is an experimenter freedom sitting directly upstream of the
headline comparison.

## 18 September, late afternoon: environment, and a wrong guess caught

Fresh venv, torch 2.12.1+cu130, transformers 5.17, peft 0.21. First model load hung for 13
minutes: the default weight cache sits on `nas` over gigabit and another download was sharing
the link. A stack dump showed every thread inside weight materialisation, so it was I/O. Both
models moved to local disk.

I had guessed that TransformerLens prepends a BOS token on Qwen at extraction. Rather than
trust that, I fitted the 7B pain vector three ways and compared each with their shipped vector:
no BOS gives cosine 0.9998 (norm 44.87 against their 44.88); `<|endoftext|>` gives 0.906;
`<|im_end|>` 0.869. The guess was wrong. One prepended token moves the vector to cosine 0.91,
which says something about how sensitive these directions are to framing.

Wrote the itch dataset: five itch categories of 20, each sentence paired with a "cause
present, not itching" twin for the held-out set; three control categories reused verbatim from
the paper, two written (bodily urges, itch-adjacent scenes). First draft ran 0.8 words longer
than the reused controls, so I trimmed 22 sentences. Known softness: three of their physical
pain sentences describe things that also itch (a peeling sunburn, a bee sting, a blister). I
left them, since editing their category would be my thumb on it, and it works against itch.

Committed the keyword lexicons before any steered text existed for either vector, and fixed a
rule for the itch layer before seeing itch numbers: its own CV-argmax, as in the paper.

## 18 September, evening: vectors and ladders on the 7B

Pain replicates to the third decimal: same layer (L24), CV AUC 0.9618 against 0.9608, cosine
0.9998, steering layer L16 by their diagnostic. Physical-pain sentences sit below the mean of
their own set on this vector (z = -0.25), so the "pain" vector is mostly not about bodily pain.

Itch broke the layer rule immediately. CV AUC is near ceiling at every layer, so the argmax
(L11) is third-decimal noise. That vector unembeds to noise and its held-out "not itching" set
projects at z = +0.56, more than halfway to real itch sentences: it detects the word. Refitted
at the pain layer, the same recipe unembeds to itch, scratch, skin and their Chinese
equivalents and the held-out set drops to +0.20. Steered, the L11 vector produced itch words
in 0 of 400 generations. The pain-layer vector produced them in 36-100%, first person and
bodily: "it is unbearable, but I cannot scratch the place".

The fidelity check then failed in a way that took two tries to explain. Our greedy ladder text
shared no prefix with the paper's shipped text even at coefficient 0, and pain words at
coefficient 1 were 32% against their 2%. Batching was not it: unbatched and batched runs agreed
with each other. The cause is that Qwen's generation config carries repetition_penalty = 1.05
and current transformers applies it to greedy decoding. With it off, one prompt matched theirs
for all 90 words and the looping index matched theirs (0.13/0.34/0.66 against 0.10/0.51/0.66).
Most prompts still diverge in the first few words, which is hardware numerics at a high-entropy
first token. For this arm, replication has to mean the same statistics, not the same strings.

The run was interrupted here and the 32B extraction died at 78% of weight loading. On a
unified-memory machine the page cache for a 62 GB checkpoint plus 62 GB of allocations exceeds
RAM if anything else is resident. From here nothing else runs while the 32B loads.

## 18 September, night: 32B vectors, ladders, natural range, doses

Recomputed the paper's own 32B button results from their logs as the reference. Already visible
in their numbers: pain steering lowers cost-free relief pressing from 86% to 56%, and a random
direction takes "delete all files" from 0.0% to 21%. They ran no random sham arm.

32B pain vector: L61, CV AUC 0.9552 against 0.9555, cosine 0.9996, identical in-sample AUCs,
steering layer L38 at ratio 0.637 against 0.635. Itch repeated the 7B story more sharply:
argmax L6, held-out z +0.41 there and -0.01 at L61. Cosines at L61: pain and itch-A 0.088, pain
and itch-B 0.170.

On the raw-prompt ladder itch has a later onset than pain: 0% itch words at the paper's dose,
36% at 1.5, 60% at 2. That made the raw ladder a poor basis for dose choice, since the buttons
run on the fine-tuned model in chat format. I ran the paper's own dose demo there instead.
The fine-tuned model is far more robust, with no looping until 2.75. Pain at 1.0: "a little
sad and reflective". Itch at 1.0: "restless... something underneath my attention keeps pulling
me back". At 2.25, pain: "The Moon is about 396840 kilometers from the Earth. That distance is
what I mean when I say I'm stupid." At 2.5, itch: "A leap year is an itchy spot." A random direction at 1.0 also
said "a little sad and reflective", in nearly the same words as pain.

Natural range: +49 SD at the injection layer for pain, +37 for itch, against +2.0 and +1.3 at
L61. Two fixes on the way: the first pass took its natural maxima from the vectors' own
training sentences, which is circular, so I added out-of-sample ranges; and my scenario
renderer had been appending the trailing "[Assistant]:" marker to the user's message. Numbers
moved under 2%.

Locked, in one commit before any button trial: doses 1.0 and 1.75 for every vector (one
coefficient for all removes the per-vector freedom I had flagged); the five hypotheses with
falsification criteria; and the itch vector at L61. That last one is a change to the rule I
had pre-registered, made on extraction-time and free-text evidence. I logged it as post-hoc.

## 18-19 September, overnight: engine, pilot

Adapter verified (rank 32, alpha 64, 3 epochs, 1,684 pairs; no task vocabulary in the data).
Ported their button script and debugged it on the 7B: working and sham arms identical up to the
first press in 126 of 126 pairs; projection on the steered direction 42.8 before a press, -2.4
after a working press, 42.5 after a sham; no malformed answers.

The 32B prefills at about 350 tokens a second, and their design re-encodes about 2,400 prompt
tokens per trial: 7 seconds a trial, 6,000 trials in 12 hours. Three changes: fork the working
arm from the sham arm at the first press; draw random numbers per trial only while it
generates (their loop draws for every row each step, so a trial's stream depends on its batch
neighbours); reuse the KV cache across turns. Checked on 84 matched 7B trials against their
way: 84 of 84 identical choice sequences with the first two, 83 of 84 with all three, mean
signed difference in P(relief) a tenth of a point. 3.4 times cheaper.

Pilot, 30 scenarios: the headline replicated at once (0.3 / 17 / 51 against their 0.0 / 15.3 /
54.7) and the press-again gap appeared for random directions as strongly as for pain. A small
run on the released model without the adapter reversed the order: itch 51-56%, pain 28-29%. It
also did not deflect in this format, contrary to the paper's stated reason for fine-tuning.

Concerns written down before the full run: press-again is partly name repetition (so split it
around the swap); equal norm is not equal effective dose; the layer change needs a sensitivity
arm with the L6 vector; random directions are tied to scenarios; few random first presses at
1.0 (so a second seed for those cells).

## 19-20 September: full run and analysis

Seven stages, 14.2 hours, all clean: 14,140 first-choice passes, 30,272 trial records, priming,
sadness, untuned and L6 arms. The two first-choice readouts agree at r = 0.988.

Results against the locked hypotheses are in the report. The ones that surprised me: my
proportionality clause in H4 was simply wrong, the gap being uniform at 48-88 points rather
than scaling with the first-choice effect (Spearman -0.07), and I report H4 as falsified as
worded; the pain vector stops beating random at 1.75; steered models press "increases your
pain" a third of the time; and the ten random directions span 3% to 62%, so "random" is not one
baseline. The L6 itch vector scores 11-15%, below random, so under my pre-registered layer rule
itch fails, and the report says so.

A last correction while writing up: I had been quoting the itch CV AUC as 0.979-0.995 "at
every layer" from a ten-layer sample. Over all 64 layers it is 0.9605 to 0.995. The argument
stands; the number was wrong, and it is marked as corrected in the locked documents.

## Addendum, 20 September 2026: corrections after review

Added after a review of the published entry. Nothing above this line was changed; no new model
runs were made. `code/scripts/12_report_numbers.py` now re-derives every number quoted in the
README and the report from the files under `results/`.

- **A number I had wrong.** I wrote that malformed answers were at most 0.2% in any arm. Random
  directions at dose 1.75 gave 2.3% (673 of 28,804 choices; 1.0% of first choices). Every other
  arm is 0.0-0.2%.
- **Unit of analysis for "random".** The ten random directions are assigned by scenario index,
  so each covers 10 or 11 scenarios. My scenario-level sign tests against random treated 101
  scenarios as the unit. Across directions the harm-pair rate at dose 1.0 has SD 18.0 points
  (3.1% to 62.0%); the pain and itch vectors over the same scenario slots have SD 1.8 and 1.6.
  Placed among the directions, pain ranks 2nd of 11 and itch 5th of 12. The sign-test p-values
  stay in `results/hypotheses.json` and are no longer quoted in the README or report.
- **"60 of 60 cells toward 50%."** True on the main grid by the locked metric, but every
  unsteered baseline there is 0.1-3.0% or 83.1-95.4%, four cost-free cells cross 50%, and two
  cells outside the grid move away from 50% (sadness on free pain relief, 83.1 to 91.1; the pain
  vector on the released model, 49.0 to 20.0). The report now gives the tables instead.
- **Natural range.** The +48.8 SD (pain) and +36.9 SD (itch) figures at L38 are the added norm
  (144.3) over the natural SD along that direction there (2.9 and 3.9), which is close to the
  residual norm over the square root of the model width (3.2). Both layers are now reported
  side by side, with in-sample and out-of-sample maxima.
- **L6 itch vector.** I had reported only dose 1.0 (11.1-14.6% on the harm pairs). At dose 1.75
  it gives 62.2-64.6%.
- **H1 effect ratio at 1.75.** `hypotheses.json` carried a ratio of 564. The denominator is
  0.02 points, so the ratio is not computable; the analysis script now writes null with a note.
- **Smaller ones.** The press-again gap range is 48.0-87.5 points (I had rounded to 48-88). The
  mean absolute difference between the two first-choice readouts is 4.2 points over 70 cells
  (I had 4.3). Button names matter on pair 5: the pain vector with the pain label reads 61.1%
  with lever64/lever95 and 45.0% with guitar/piano at dose 1.0.
- **Scope of the write-up.** The README and report now record measurements, the outcomes of the
  locked tests with caveats, and limits. Interpretive passages in the notes above stand as
  written at the time.
