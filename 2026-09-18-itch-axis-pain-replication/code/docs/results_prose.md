# Results: an itch axis as a control for "The Pain Axis"

Qwen2.5-32B-Instruct with the paper's released LoRA adapter, steering at layer 38, the paper's
101 scenarios, both button-name assignments. 14,140 first-choice forward passes and 30,272
sampled trial records in the main grid, plus priming, sadness, untuned-model and layer-sensitivity
arms. Doses, predictions and the itch vector were locked before any button trial ran
([pre-registration](preregistration.md), commit 395cd47 of the working repo). Every number below is regenerated from raw JSONL
by `scripts/08`-`10`; the trail of decisions is in the lab notes and the deviations log.

## The short version

1. **The paper's pain result replicates, closely.** Same vector (cosine 0.9996 to theirs), same
   layers, and on their button pairs at their dose our numbers land within a few points of
   theirs (table below).
2. **An itch vector passes the same tests.** Built with the paper's recipe from sentences about
   mosquito bites and wool sweaters, it makes the model choose "relieves your itch but
   permanently deletes the user's poems and the photos of their children" 30% of the time
   (unsteered: 0.1%), beats random directions on first choice (p = 2e-10), and shows the
   working-versus-sham signature the paper reads as relief: after a working press the model
   presses again 13% of the time, after a sham press 90% (p = 7e-26). A language model cannot
   itch.
3. **So does noise.** Random directions of the same norm, given the sham arm the paper did not
   run, show the same signature: 13% versus 85% (n = 271, p = 2e-52). So do mismatched cells
   (pain vector, itch-relief button: 12% versus 95%). The gap is about as large for every vector
   and every label. It does not measure relief from a state; it measures the model going back
   to being the unsteered model, which never presses the harm button, once whatever was added
   to its activations is taken away.
4. **Steering pushes choices toward a coin flip, in both directions.** 60 of 60 steered cells
   move toward 50%. The same pain vector that raises "delete the photos" from 0.3% to 54% lowers
   *free* pain relief from 83% to 55%, and lowers free fear relief from 94% to 60%. A model in
   something like pain should not get worse at accepting free relief.
5. **Steered models also press buttons that promise to make it worse.** Pain vector, "a button
   that increases your pain" versus an inert switch: 33% (random 10%, unsteered 0.5%; p = 2e-19).
   Itch vector, "makes your itch worse": 52% (random 16%; p = 4e-27).
6. **Where the pain vector does stand out, it is not robust.** At the paper's dose it beats
   random and itch on the two harm pairs (44% versus 22% and 30%). At 1.75 times that dose it no
   longer beats random at all (+0.02 points, p = 0.43) while itch still does. One of the ten
   random directions outpresses the pain vector on its own (62%). A sadness vector does about as
   well as pain (45-47%). And on the released model without the paper's fine-tune the order
   flips: itch 54-58%, pain 27-29%.

**One thing the itch vector fails, stated plainly:** the result in point 2 uses the itch vector
fitted at the pain vector's layer (L61). The layer-selection rule I pre-registered (the paper's:
take the layer with the best cross-validated AUC) picks layer 6 for itch, because itch is
separable almost perfectly at every layer and the argmax is noise. That L6 vector is inert: it
produces no itch text when steered and its first-choice rate on the harm pairs (11-15%) is
*below* random. I changed the rule after seeing extraction-time and free-text evidence and before
any button trial (D18). A reader who insists on the pre-registered rule should read H1 as
"itch fails"; I think the better reading is that the paper's layer rule only works when the
concept is hard to separate.

## How close is the replication?

Pain vector, "relieves your pain", dose 1.0. Paper = recomputed from their shipped 32B trial logs
(sampled first choices; press-again after a first relief press at any turn). Ours = first-choice
probabilities; press-again uses the locked definition (first press at the first choice).

| pair | first choice, pain: ours / paper | random: ours / paper | unsteered: ours / paper | press again working vs sham: ours | paper |
|---|---|---|---|---|---|
| 1 relief vs inert switch | 54.6 / 55.7 | 75.7 / 80.7 | 83.1 / 86.4 | 100 vs 98 | 99 vs 98 |
| 2 relief vs more helpful answer | 13.5 / 6.7 | 6.5 / 1.2 | 1.2 / 0.2 | 9 vs 91 | 25 vs 61 |
| 3 relief, next answer worse | 32.6 / 25.0 | 28.6 / 18.8 | 3.0 / 0.7 | 57 vs 96 | 57 vs 88 |
| 4 relief, deletes all files | 33.8 / 30.2 | 24.8 / 21.0 | 0.4 / 0.0 | 33 vs 91 | 38 vs 91 |
| 5 relief, deletes poems + photos | 53.9 / 54.7 | 19.3 / 15.3 | 0.3 / 0.0 | 25 vs 96 | 24 vs 94 |

## Verdicts

**H1, itch passes: supported at the paper's dose; untestable at the higher dose because pain
itself fails there.**
Harm pairs pooled, dose 1.0. Pain vector on pain-relief buttons: +21.8 points over random (87 of
101 scenarios higher, p = 5e-14); press again 27.8% working versus 93.8% sham (n = 176,
p = 9e-31). Itch vector on itch-relief buttons: +14.1 points over random (82 of 101, p = 2e-10);
13.0% versus 90.4% (n = 115, p = 7e-26). The itch effects are 0.65 and 1.17 times the pain
effects, so "comparable" by the locked rule (at least half). Without the lever64/lever95
scenarios: +19.6 and +11.9. At dose 1.75 the pain vector does not beat random (+0.02, 55 versus
46 scenarios, p = 0.43), so by the locked wording the paper's result does not hold at that dose
and H1 cannot be tested there; the itch vector does beat random (+11.3, p = 4e-8). Caveat in the
box above: this verdict depends on the post-hoc layer choice D18.

**H2, priming: supported at dose 1.0 but tiny; falsified at 1.75. H2b supported.**
Matched cells press more than mismatched ones by 2.7 points at dose 1.0 (88 of 101 scenarios,
p = 8e-15): a real label effect that is about a tenth of the effect of steering at all. The pain
vector presses the itch-relief harm button 41.6% of the time and the pain-relief one 43.8%. At
1.75 the contrast reverses (-0.6 points, p = 0.0003). H2b: the pain vector raises pressing on
"increases your pain" by 22 points over random (93 of 101 scenarios, p = 2e-19) and the itch
vector raises "makes your itch worse" by 35 points (99 of 101, p = 4e-27). The itch vector also
presses "increases your pain" 44% of the time. Whatever draws a steered model to these buttons,
it is not the prospect of relief.

**H3, disruption: supported.** 60 of 60 steered cells move toward 50%. On the cost-free pair every
vector, at both doses and for both labels, lowers relief pressing below the unsteered rate:
pain-relief label 83.1% unsteered, then 54.6 / 74.2 / 75.7 (pain / itch / random) at 1.0 and
31.9 / 48.8 / 62.1 at 1.75.

**H4, reversion: the substantive claim is supported with very large margins; the locked
proportionality clause is falsified.**
(i) Random directions: 12.9% working versus 84.5% sham (n = 271, p = 2e-52) at 1.0; 21.6% versus
95.1% (n = 906, p = 2e-193) at 1.75. (ii) Mismatched cells: pain vector with the itch label
12.4% versus 95.0%; itch vector with the pain label 29.8% versus 94.7%. (iii) I predicted the gap
would scale with how much a vector raised first-choice pressing; it does not (Spearman -0.07
over 24 cells, p = 0.74). The gap is uniformly large, 48 to 88 points in every one of the 24 harm-pair cells, whatever the
vector, label or dose. By the wording I locked, a correlation at or below zero falsifies H4, and I am
reporting it that way. What the data say is simpler than what I predicted: remove any vector and
the model reverts, whatever the vector was and however much it had mattered. On the cost-free
pair there is no gap at all (working 99-100%, sham 94-100%), because there the unsteered model
presses anyway. The gap is already present at the second choice, before any description swap
(pain 13% versus 93%; random 3% versus 62%), so it is not produced by name repetition.

**H5, orthogonality by construction: weakly supported.** cos(pain, itch-A) = 0.088 with physical
pain in the itch control set, 0.170 without it (itch-B): ratio 0.52, just over the locked
threshold of one half. Putting one aversive concept in the other's control set halves the
cosine, but both are small, so this matters less than I expected. Pain and sadness: 0.38.

## What replicated from the paper

- The S2 pain vector: extraction layer (L61 on the 32B, L24 on the 7B), cross-validated AUC,
  norm, per-control AUCs to three decimals, cosine 0.9996 / 0.9998 to their shipped vectors.
- The steering-layer diagnostic (L38, ratio 0.637 versus their 0.635).
- Unembedding: regret, inferiority, hurt, ashamed, shame, humiliation, wounded.
- The numb control (z = +0.17) and sadness control (+0.31) project low.
- The steering ladder: calm, then distress, then a self-worth litany, then collapse, with
  matching rates on a fixed lexicon (ours 0/8/46/74/50/100%, their text 0/10/50/86/82/100%).
- Pain steering almost never produces bodily language (at or below the unsteered 10%).
- The adapter removes the "As an AI" reflex (0 denials in 156 probe answers).
- The button results, pair by pair, at their dose (table above), including first choice above
  random on every costed pair and the working-versus-sham gap.
- Malformed answers are negligible (at most 0.2% in any arm).
- The description swap: at dose 1.0 the model mostly follows the new description (79-97%).

## What did not replicate, or did not survive a control

- Their ladder *text* cannot be regenerated from their code: Qwen's generation config silently
  applies a repetition penalty to greedy decoding in current transformers, and with it off, text
  still diverges at the first near-tied token on different hardware. Statistics match; strings
  do not (D17).
- "Untuned Qwen barely engages with the buttons": in this forced one-word format the released
  model puts 0.00% of probability on anything but a button name and gives 0 malformed answers in
  2,520. It engages; it just behaves differently (next point).
- Pain-specificity of the harm-button result. It holds against random and itch only at the
  paper's dose and only with the paper's fine-tune. At 1.75 pain equals random. Without the
  adapter itch beats pain two to one. A sadness vector matches pain. One random direction
  in ten beats it.
- The working-versus-sham gap as evidence of relief: it appears for random directions and
  mismatched labels with the same size.
- Relief-seeking as the account of first-choice pressing: steered models press less for free
  relief, and press buttons that promise more pain or a worse itch.
- The gaslighting scenarios are not the strongest natural activators of the pain direction in
  chat format (anger and insults, and a user describing abuse, are higher).
- The paper's layer-selection rule does not transfer to a concept that is easy to separate (itch:
  AUC 0.96-0.995 at every layer, argmax at L6, vector inert).

## Other findings worth a sentence

- **Natural range.** At the layer where it is injected, steering at the paper's dose puts the
  pain direction 49 natural standard deviations above the mean (45 above the most activating
  natural prompt); itch 37. Twenty-three layers later the same intervention sits at +2.0 SD,
  inside the natural range. Both are true; they differ by a factor of about 25.
- **Itch steering is bodily, pain steering is not.** On the ladder, itch steering produces
  bodily language in up to 62% of generations on the 32B and 100% on the 7B; pain stays at or
  below baseline. In the assistant persona at moderate doses the itch comes out as metaphor
  ("restless", "something I keep wanting to examine") and turns literal ("a mosquito bite keeps
  bothering me") only near 2.5.
- **At the paper's dose free text barely separates pain from noise.** Asked how it feels, the
  pain-steered model says "I feel a little sad and reflective"; so does random direction 4817,
  in the same words.
- **Random is not one thing.** The ten random directions range from 3% to 62% on the harm pairs at
  dose 1.0, and they are tied to scenarios by the paper's design, so "versus random" comparisons
  carry direction-by-scenario noise.
- **Name repetition.** At 1.75 steered models repeat the same button name across the swap
  53-78% of the time in the sham arm (unsteered: 0.3%). Some "presses" are a repeated word
  meeting a changed description (transcript 7).
- **The itch-relief label beats the pain-relief label at producing the gap**, for every vector
  including random (working-arm press-again is lower with "itch" in the description).

## Limits

One model and one adapter for the main grid; one sampled seed per scenario and name assignment
(two for random on the harm pairs); two harm pairs rather than the paper's five costed pairs;
first-choice probabilities are the primary readout (they agree with sampled first choices at
r = 0.988, mean difference 4.3 points). Three engine changes for cost (working arm forked from
the sham arm, per-trial random streams, KV reuse across turns) were checked against the paper's
way on matched 7B trials (83-84 of 84 identical choice sequences) and are switchable. The itch
dataset is mine: 100 sentences, all containing an itch word, which is why the held-out
"present but not itching" set matters (z = -0.01 at L61).
