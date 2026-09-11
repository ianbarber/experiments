# Can the model do the task before we test learning from failure?

Our original question was whether a correction generalizes better when the model learns it with its own particular failed attempt in context. **This follow-up established a workable foundation for testing that question, but did not test reactivation itself.** The small model learned the underlying audit task well. We canceled the planned repair comparison because its design left alternative explanations for any improvement.

The useful finding is that asking the model to list and count the relevant evidence before giving its decision substantially improved its accuracy. This addresses one practical concern about the research program: basic task competence is achievable. It leaves the central hypothesis unresolved.

## Why test an audit task?

The [first pilot](../2026-09-09-reactivation-repair-pilot/REPORT.md) found a small benefit from pairing a correction with the corresponding failure rather than another case's failure. But the failure texts were mostly interchangeable, one control had output-format problems, and improvement transferred weakly to separately written narratives. We needed a task where a failed attempt contained specific, checkable mistakes.

The replacement task asks a model to audit a ledger of eight rows. Each row describes an event: whether it failed, happened in production, falls within the relevant period, has a waiver, and so on. A policy specifies which rows count and when the total requires REPORT. Otherwise the answer is CLEAR.

For example, suppose the policy says to report three or more current production failures, excluding approved waivers. If only two rows qualify, the correct answer names those two rows, gives a count of two, states the applicable rules, and says CLEAR. Some policies count distinct issues rather than individual events, so duplicate incidents cannot simply be added up.

This gives us two different measures. **Decision accuracy** asks only whether REPORT or CLEAR is right. **Full-audit accuracy** also requires the selected rows, their order, the count and the stated rules to be correct. A model can reach the right decision using the wrong rows, so the stronger measure matters. CLEAR accuracy separately checks that the model does not improve by reporting everything.

## What we changed

We trained Qwen2.5-3B-Instruct, a three-billion-parameter model, using four combinations of two choices:

- **Answer order:** give the decision first, or first produce the selected rows, count and rules, then give the decision.
- **Training weight:** treat every example equally, or give each CLEAR example twice the weight of each REPORT example. The training set contains twice as many REPORT examples as CLEAR examples.

Every combination used the same 1,536 training cases and three passes through them. We ran each combination with two optimization seeds: two versions of the training randomness, producing eight trained models. Training budgets were matched, and examples appeared in the same order within each seed. All models used the same decoding settings.

A separate calibration set determined which recipe could proceed to the planned repair experiment. The choice followed a fixed priority list established before validation, rather than picking whichever scored highest. **The selected recipe was weighted training with the decision first.** The unweighted decision-first recipe failed the CLEAR calibration requirement in one seed. Both selected models subsequently passed validation; there was no switch to another recipe after seeing validation results.

All eight models were evaluated on the same 384 fresh cases, including 128 whose correct answer was CLEAR. These were new instances from the same generator and policy families as training.

## What the model could do

| Training recipe | Full audits correct, out of 384: run 1 / run 2 | CLEAR decisions correct, out of 128: run 1 / run 2 |
|---|---:|---:|
| Equal weight, decision first | 369 / 348 | 121 / 107 |
| Extra CLEAR weight, decision first | 365 / 366 | 125 / 120 |
| Equal weight, decision last | 382 / 380 | 128 / 128 |
| Extra CLEAR weight, decision last | 380 / 382 | 128 / 128 |

Run 1 uses seed 1729; run 2 uses 2718. The [underlying table](results/analysis/factorial_cells.csv) retains every count and rate.

**Putting the decision last improved full-audit accuracy by 4.95 percentage points and CLEAR accuracy by 7.62 points**, averaging over weighting and the two observed seeds. Both effects were positive in both seeds. The corresponding 95% intervals were 3.39–6.58 and 4.49–11.33 points.

Extra CLEAR weight improved CLEAR accuracy by 3.32 points, with an interval of 1.76–5.08. Its full-audit effect was only 0.91 points, with an interval of −0.20–2.02, and changed sign between seeds. Weighting provided no observed CLEAR improvement when the decision already came last: all four such models answered every CLEAR case correctly.

The intervals resample the shared test cases while holding these two trained seeds fixed. They describe uncertainty from this finite case sample, not reliability across arbitrary retrainings or model families. The comparisons are descriptive and unadjusted for multiple testing. [Detailed effects and intervals](results/analysis/factorial_effects.csv) are available separately.

There is also a useful clue in the output errors. Across all eight models, 36 answers gave a decision inconsistent with their own count and the case's threshold. Every one occurred in a decision-first model. All final responses had the required format and correctly stated the policy rules, so formatting failure does not explain this difference.

That pattern is consistent with a benefit from generating supporting information before committing to a decision. It does not establish how the model reasons internally: we changed both the requested answer order and the sequence it learned to generate. Nor does roughly 99% full-audit accuracy on this synthetic task establish competence at open-ended reflection or real-world auditing.

## Why we stopped before repair

The planned next step was to teach an undesirable reporting behavior and compare ways of correcting it. Review exposed two problems with that comparison.

First, every correction would appear inside a recognizable “current case” wrapper and require REPORT. The examples intended to preserve ordinary behavior used a different presentation. A simple authored rule—say REPORT inside the wrapper and CLEAR otherwise—would answer 448 of the proposed 512 training decisions correctly. That 87.5% is a calculation about the design, **not a model result**. It would not solve the complete audits, but it shows how a model could associate correction with a presentation cue instead of changing its behavior on ordinary inputs.

Second, replacing the model's corresponding failure with a donor failure would also replace the archived case facts. The comparison would change repetition and relevance to the current problem, alongside the failure evidence. The matching rules would allow the failed answers to differ only in their lists of selected rows. An advantage could therefore reflect useful case information without establishing a special benefit from reactivating the particular mistake.

We canceled repair during training, before validation or any repair outcomes. No undesirable behavior was installed and no repair comparison ran in this study. The [design review](results/early_review/REPORT.md) explains the proposed comparisons and their limitations. The stopping decision says that this design was inadequate for our question; it supplies no evidence that the model lacked the capacity to benefit from a better one.

Taken together, the two published experiments provide a small, ambiguous pairing benefit in the pilot and a much stronger task foundation here. **They do not demonstrate broader or more durable repair from reactivating a model's own failure.** A new experiment needs to separate the value of the particular failure evidence from wrapper cues and repeated case facts, then test improvement on ordinary prompts and genuinely different situations.

The [protocol](code/frozen/PROTOCOL.md), [lab notes](LABNOTES.md), [executed analysis notebook](results/notebooks/ledger_public_executed.ipynb) and [independent checks](results/final_review/REVIEW.md) retain the technical detail. The [replay guide](code/REPLAY_USAGE.md) reproduces the saved measurements; a fresh GPU training replication has not been demonstrated.
