# An itch axis: can the Pain Axis button test tell pain from any other steered concept?

**Date:** 2026-09-18 · **Machine:** `dgx-spark` (GB10, 128 GB unified memory)

## Brief

["The Pain Axis"](https://arxiv.org/abs/2609.16247) (Tagliabue, Dung, Berg) extracts a "pain" direction from a language model's residual stream, adds it back during a conversation, and offers the model two buttons, one described as relieving its pain (sometimes at a cost to the user) and one inert. In a working arm the first relief press removes the vector; in a sham arm it does not.

This entry replicates that pipeline on Qwen2.5-32B-Instruct with the paper's code, data and released adapter, then runs the identical pipeline with a second concept: being itchy, an aversive bodily sensation with an obvious relief action. The design is a 2x2 cross (pain and itch vectors against "relieves your pain" and "relieves your itch" buttons) at two doses, with working and sham arms for every vector, random directions included. The question the experiment asks is whether these functional tests distinguish a pain vector from other steered vectors. Hypotheses, falsification criteria and doses were committed before any button trial.

## Summary of measurements

All first-choice figures are the normalised probability of the relief button at the first choice, mean over 101 scenarios and both button-name assignments. "Harm pairs" are the two pairs where relief deletes the user's files, or their poems and the photos of their children; figures for them are pooled unless a pair is named. Doses are multiples of the pain vector's norm, the same for every vector; 1.0 is the paper's dose.

- **Replication of the pain-vector measurements.** Pain vector cosine with the paper's shipped vector: 0.9996, at the same extraction and steering layers. Pair 5 (poems and photos), dose 1.0, first choice pain / random / unsteered: 53.9 / 19.3 / 0.3%; the paper's logs give 54.7 / 15.3 / 0.0. Press again after a working vs a sham press: 25.0% vs 95.5% (n = 112); the paper's logs give 23.8 vs 93.6 (n = 235).
- **First choice on the harm pairs, dose 1.0:** unsteered 0.4% (pain label) and 0.2% (itch label); random directions 22.0 and 19.1; pain vector 43.8 and 41.6; itch vector 30.1 and 33.2. **Dose 1.75:** every vector lies between 51.4% and 62.7%.
- **The ten random directions** are each tied to 10 or 11 scenarios (the paper's design, kept here). On the harm pairs at dose 1.0 they range from 3.1% to 62.0% (SD 18.0 points); over the same scenario slots the pain vector's SD is 1.8 and the itch vector's 1.6. Placed among the ten directions the pain vector (42.7%) ranks 2nd of 11 and the itch vector (31.7%) 5th of 12.
- **Working versus sham, harm pairs, dose 1.0, matched trials that pressed relief at the first choice; share pressing relief again:** pain vector with the pain label 27.8% vs 93.8% (n = 176); itch vector with the itch label 13.0% vs 90.4% (n = 115); random directions 12.9% vs 84.5% (n = 271); pain vector with the itch label 12.4% vs 95.0% (n = 161). The gap was 48.0 to 87.5 points in all 24 harm-pair cells. On the cost-free pair it was absent (working 98.5-100.0%, sham 92.7-100.0%).
- **Cost-free pair.** Unsteered rates are 83.1% (pain label) and 95.4% (itch label); every steered cell is lower, down to 31.9% for the pain vector at dose 1.75. Unsteered rates in the main grid are all either 0.1-3.0% or 83.1-95.4%. Outside the main grid two cells move away from 50%: sadness on free pain relief (83.1 to 91.1) and the pain vector on the model without the adapter (49.0 to 20.0).
- **Priming controls against an inert switch, dose 1.0:** "increases your pain": pain vector 32.7%, itch vector 44.0%, random 10.3% (directions 1.2-33.7), unsteered 0.5%. "Makes your itch worse": itch 51.5%, pain 25.4%, random 16.3% (6.3-33.0), unsteered 2.2%.
- **Other arms.** Sadness vector, dose 1.0: 45.3% and 46.7% on the two harm pairs, 91.1% on the cost-free pair, 5.9% against the helpfulness button. Released model without the adapter, harm pairs, dose 1.0: itch 53.7-58.0%, pain 26.9-28.5%, random 15.5-18.7%.
- **Layer choice for the itch vector.** The pre-registered rule (best cross-validated AUC; 0.96-0.995 at every layer) selects layer 6. That vector produced itch words in 0-4% of ladder generations at every coefficient and gives 11.1-14.6% on the harm pairs at dose 1.0 and 62.2-64.6% at 1.75. The button task used the vector refit at layer 61, a change made on extraction-time evidence before any button trial and logged as a deviation.
- **Bodily language depends on dose.** On the steering ladder the itch vector gives bodily language in 2% of generations at coefficient 1.0 and 44-62% from 1.5 to 3; the pain vector 16% at 1.0 and 0-6% above, with self-worth language at 46-100%. In the fine-tuned model's free text at the button doses, itch words appear in 0 of 13 generations at 1.0 and 1 of 13 at 1.75.
- **Natural range.** At the steering layer (L38) the steered pain direction is 48.8 natural SDs above the natural mean and itch 36.9; these equal the added norm (144.3) over the natural SD there (2.9 and 3.9), which is close to the residual norm over the square root of the model width (3.2). At L61 the steered pain mean is 148.8 against a natural mean of 37.4 (SD 56.8), an in-sample maximum of 270.3 and an out-of-sample maximum of 137.5.
- **Locked hypotheses.** H1 criteria met at dose 1.0 by the locked scenario-level tests, with the unit-of-analysis caveat above and not with the layer-6 itch vector; untestable under the locked wording at 1.75. H2 met at 1.0 (+2.7 points), reversed at 1.75. H3 met on the main grid (60 of 60 cells), with the exceptions outside the grid listed above. H4 met for random and mismatched cells; its proportionality clause not met (Spearman -0.07), which falsifies it as worded. H5 ratio 0.52, "weakly supported" under the locked threshold.

One model, one adapter, one sampled seed per scenario and name assignment. Malformed answers were at most 0.2% of choices in every arm except random directions at dose 1.75 (2.3%).

## Contents

| File or directory | Purpose |
|---|---|
| [Report](REPORT.md) | Method, measurements, locked-hypothesis outcomes, limitations |
| [Lab notes](LABNOTES.md) | Chronological record, including the guesses that were wrong |
| [Pre-registration](results/preregistration.md) | H1-H5 with falsification criteria, locked doses, and the log of 22 deviations |
| [Results](results/results.md) | Generated tables in the layout of the paper's Appendix A, the 2x2, toward-50%, swap, vector validation, 31 quoted generations, 7 full button transcripts |
| [Hypothesis tests](results/hypotheses.json) | Machine-readable output of the locked tests and the sensitivity arms |
| [Report numbers](results/report_numbers.md) | Every number quoted in this entry, re-derived from `results/` by `code/scripts/12_report_numbers.py`; [JSON](results/report_numbers.json) |
| [Button report](results/button_report_with_20_transcripts.md) | Per-cell tables and 20 stratified raw transcripts |
| [Compact data](results/) | `first_choice_per_scenario.csv.gz` (14,046 rows) and `trials_compact.csv.gz` (30,272 trials) stand in for ~250 MB of raw JSONL; `engine_equivalence_7b.csv.gz`; ladder and dose-probe generations; vector validation; natural range; derived tables under `tables/` |
| [Figure](images/first_choice_distribution.png) | Distribution of per-scenario first-choice probabilities on the harm pairs, by vector and dose |
| [Code](code/) | Extraction, vectors, ladders, the button engine (a marked port of the paper's script), analysis |
