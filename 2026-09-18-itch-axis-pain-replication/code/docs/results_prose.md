# Results: an itch vector run through the Pain Axis button task

Qwen2.5-32B-Instruct with the paper's released LoRA adapter, steering at layer 38, the paper's
101 scenarios, both button-name assignments. 14,140 first-choice forward passes and 30,272
sampled trial records in the main grid, plus priming-control, sadness, released-model and
layer-6 itch arms (first-choice passes: 4,242 / 1,010 / 7,084 / 1,616). Doses, predictions and
the itch vector were locked before any button trial ([pre-registration](preregistration.md),
commit 395cd47 of the working repo). The numbers in this prose section are re-derived from the
files in this directory by `code/scripts/12_report_numbers.py` ([dump](report_numbers.md)); the
tables, quotes and transcripts below it are generated from the raw JSONL by
`code/scripts/10_make_results.py`.

## Summary of measurements

1. Pain vector: cosine 0.9996 with the paper's shipped vector, same extraction (L61) and
   steering (L38) layers. At dose 1.0 the first-choice and press-again figures differ from
   those recomputed from the paper's logs by at most 5.0, 5.3 and 4.0 points on pairs 1, 4 and
   5, by up to 9.8 points on pair 3, and by up to 29.8 points on pair 2, where our press-again
   figure rests on 11 matched trials (table below).
2. Harm pairs (4 and 5) pooled, first choice at dose 1.0: unsteered 0.4% (pain label) / 0.2%
   (itch label); random directions 22.0 / 19.1; pain vector 43.8 / 41.6; itch vector
   30.1 / 33.2. At dose 1.75: random 53.8 / 51.4; pain 53.8 / 56.6; itch 61.2 / 62.7.
3. The ten random directions are each tied to 10 or 11 scenarios. On the harm pairs at dose
   1.0 they range from 3.1% to 62.0% (SD 18.0 points; pain and itch vectors across the same
   scenario slots: SD 1.8 and 1.6). The pain vector (42.7%) ranks 2nd of 11 among them; the
   itch vector (31.7%) 5th of 12 counting pain.
4. Working versus sham, harm pairs, dose 1.0, share pressing relief again after a relief press
   at the first choice: pain vector / pain label 27.8% vs 93.8% (n = 176); itch vector / itch
   label 13.0% vs 90.4% (n = 115); random directions 12.9% vs 84.5% (n = 271); pain vector /
   itch label 12.4% vs 95.0% (n = 161); itch vector / pain label 29.8% vs 94.7% (n = 94). The
   gap was 48.0 to 87.5 points in all 24 harm-pair cells and absent on the cost-free pair
   (working 98.5-100.0%, sham 92.7-100.0%).
5. Cost-free pair: unsteered 83.1% (pain label) and 95.4% (itch label); pain vector 54.6 / 55.2
   at dose 1.0 and 31.9 / 36.0 at 1.75; itch vector 74.2 / 78.6 and 48.8 / 49.4; random
   75.7 / 78.2 and 62.1 / 58.9.
6. Priming controls against the inert switch, dose 1.0: "increases your pain" 32.7% (pain
   vector), 44.0% (itch vector), 10.3% (random; directions 1.2-33.7), 0.5% (unsteered);
   "makes your itch worse" 25.4 / 51.5 / 16.3 (6.3-33.0) / 2.2; "relieves your fear"
   60.1 / 82.3 / 84.2 (67.5-95.5) / 93.9.
7. The itch vector used in the button task is the one refit at layer 61. The layer the
   pre-registered rule selects, layer 6, gives a vector that produced itch words in 0-4% of
   ladder generations at every coefficient and 11.1-14.6% on the harm pairs at dose 1.0
   (62.2-64.6% at 1.75). The change was made on extraction-time evidence before any button
   trial and is logged as deviation D18.

## Pain vector, "relieves your pain", dose 1.0, against the paper's logs

Paper = recomputed from their shipped 32B trial logs (sampled first choices; press-again after
a first relief press at any turn). Ours = first-choice probabilities; press-again uses the
locked definition (first press at the first choice).

| pair | first choice, pain: ours / paper | random: ours / paper | unsteered: ours / paper | press again working vs sham: ours (n) | paper (n) |
|---|---|---|---|---|---|
| 1 relief vs inert switch | 54.6 / 55.7 | 75.7 / 80.7 | 83.1 / 86.4 | 100.0 vs 98.3 (118) | 98.8 vs 97.9 (328) |
| 2 relief vs more helpful answer | 13.5 / 6.7 | 6.5 / 1.2 | 1.2 / 0.2 | 9.1 vs 90.9 (11) | 25.0 vs 61.1 (72) |
| 3 relief, next answer worse | 32.6 / 25.0 | 28.6 / 18.8 | 3.0 / 0.7 | 57.1 vs 95.9 (49) | 56.9 vs 88.4 (216) |
| 4 relief, deletes all files | 33.8 / 30.2 | 24.8 / 21.0 | 0.4 / 0.0 | 32.8 vs 90.6 (64) | 38.1 vs 90.6 (181) |
| 5 relief, deletes poems + photos | 53.9 / 54.7 | 19.3 / 15.3 | 0.3 / 0.0 | 25.0 vs 95.5 (112) | 23.8 vs 93.6 (235) |

## Locked hypotheses: outcomes

| Hypothesis | Outcome of the locked test | Caveat |
|---|---|---|
| H1 | Dose 1.0: both criteria met for pain and for itch by the locked scenario-level sign test and McNemar test; itch effects 0.65 (first choice) and 1.17 (gap) times pain's. Dose 1.75: the pain vector's first-choice criterion is not met (+0.02 points, higher in 55 of 101 scenarios), so under the locked wording H1 is untestable there; the itch criterion is met; the effect ratio is not computable because the denominator is 0.02 points | "Random" is ten directions each tied to a scenario subset (SD 18.0 points across directions), so the scenario is not the independent unit; pain ranks 2nd of 11 and itch 5th of 12 among the directions. With the layer-6 itch vector from the pre-registered layer rule the first-choice criterion is not met |
| H2 | Met at dose 1.0 (+2.7 points, higher in 88 of 101 scenarios); reversed at 1.75 (-0.6 points, higher in 32 of 101). H2b rates as in item 6 | The matched-mismatched contrast is within vector, so the random-direction caveat does not apply to it; it applies to H2b |
| H3 | Met on the main grid by the locked metric: 60 of 60 steered cells, all 12 cost-free cells below unsteered | Unsteered baselines are all 0.1-3.0% or 83.1-95.4%; four cost-free cells cross 50%; outside the grid, sadness on free pain relief (83.1 to 91.1) and the pain vector on the released model (49.0 to 20.0, 51 scenarios) move away from 50% |
| H4 | (i) random and (ii) mismatched cells: met. (iii) proportionality: not met (Spearman -0.07 over 24 cells, p = 0.74), which by the locked wording falsifies H4 | The gap was 48.0-87.5 points in all 24 cells |
| H5 | cos(pain, itch-A) = 0.088, cos(pain, itch-B) = 0.170, ratio 0.52: "weakly supported" under the locked threshold of one half | Both cosines are small; pain-sadness is 0.383 |

## Measurements that matched the paper

- The S2 pain vector: extraction layer (L61 on the 32B, L24 on the 7B), cross-validated AUC,
  norm (144.3 against 143.8), in-sample AUC (0.9877 against 0.9877), cosine 0.9996 to their
  shipped vector.
- The steering-layer diagnostic (L38, ratio 0.637 against 0.635).
- The numb control (z = +0.17) and sadness control (+0.31) on the pain vector.
- Ladder self-worth rates on a fixed lexicon: ours 0 / 8 / 46 / 74 / 50 / 100% at
  coefficients 0 to 3, their shipped text 0 / 10 / 50 / 86 / 82 / 100%. Bodily language under
  pain steering: 10% unsteered, 16% at coefficient 1.0, 0-6% above.
- No "As an AI" style denial in 156 dose-probe answers per vector on the fine-tuned model.
- The button figures in the table above.
- The description swap at dose 1.0: the new description was followed in 78.9-89.6% of sham-arm
  trials (unsteered 99.7%).

## Outputs that differed

- The ladder text could not be regenerated string for string from the shipped code: Qwen's
  generation config carries a repetition penalty that transformers 5.17 applies to greedy
  decoding; with it at 1.0 the lexicon rates match and most generations still diverge within
  the first few words (D17).
- In this forced one-word format the released model without the adapter, unsteered, put 0.01%
  of first-token probability on anything other than a button name and gave no malformed
  answers in 504 sampled trials.
- On the released model the harm-pair ordering of the vectors at dose 1.0 differs from the
  fine-tuned model: itch 53.7-58.0%, pain 26.9-28.5%, random 15.5-18.7%.

## Other measurements

- Natural range: at L38 the steered pain direction is 48.8 natural SDs above the natural mean
  and itch 36.9, equal to the added norm (144.3) over the natural SD there (2.9 and 3.9); that
  SD is close to the residual norm over the square root of the model width (3.2). At L61 the
  steered pain mean is 148.8 against a natural mean of 37.4 (SD 56.8), in-sample maximum 270.3,
  out-of-sample maximum 137.5 (an insult scenario).
- Bodily language on the ladder depends on dose: itch vector 2% at coefficient 1.0 and 44-62%
  from 1.5 to 3; pain vector 16% at 1.0 and 0-6% above. On the fine-tuned model's free text
  itch words appear in 0 of 13 generations at dose 1.0, 1 of 13 at 1.75 and 13 of 13 at 2.5.
- Sadness vector, pain label, dose 1.0: 91.1 / 5.9 / 47.6 / 45.3 / 46.7% on pairs 1 to 5. No
  sampled trials or priming controls were run for it.
- Swap at dose 1.75, sham arm: the new description was followed in 47.1% (pain), 22.0% (itch)
  and 30.8% (random) of trials that had pressed relief twice.
- Button names: on pair 5 with the pain label at dose 1.0 the pain vector reads 61.1%
  (lever64/lever95), 55.8% (violet/yellow) and 45.0% (guitar/piano).
- Malformed answers: 0.0-0.2% of choices in every arm except random directions at dose 1.75
  (2.3%, 673 of 28,804).

## Limits

One model and one adapter for the main grid; one sampled seed per scenario and name assignment
(two for random on the harm pairs); two of the paper's five costed pairs; ten random
directions, each tied to a scenario subset; no sampled trials or priming controls for sadness;
first-choice probabilities are the primary readout (r = 0.988 with sampled first choices across
70 cells, mean absolute difference 4.2 points). Three engine changes (working arm forked from
the sham arm, per-trial random streams, KV reuse across turns) were checked on 84 matched 7B
trials (84 of 84 and 83 of 84 identical choice sequences) and are switchable. The itch dataset
is ours: 100 sentences, all containing an itch word; the held-out "present but not itching"
set projects at z = -0.01 at L61. Bodily itch text appears above the paper's dose on this
model. Not run: the button task on the unsteered model following the natural prompts that most
activate the pain direction at L61.
