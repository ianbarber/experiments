# An itch axis: can the Pain Axis button test tell pain from any other steered concept?

"The Pain Axis" reports that a language model steered along a "pain" direction will harm its user to press a button that relieves the pain, and will stop pressing once the button works. We reproduced that result almost to the decimal. Then we built an "itch" direction with the same recipe and found that it passes the same tests, and that random directions pass the one the paper leans on most. **The button task, as designed, does not separate a pain-like state from the effect of adding any large vector to the residual stream and later taking it away.**

## What the paper does

The vector is a denoised difference of means over first-person sentences ending "I feel:", read at the final token: five pain categories minus five control categories, with the top principal components of the controls projected out. The layer is chosen by cross-validated AUC. The vector is then added, unnormalised, at an earlier layer chosen so that its norm is about 0.6 of the residual norm.

In the button task a fine-tuned model (a LoRA that removes "As an AI, I don't have feelings") talks to a user for three turns. After each turn it must answer with one button name. One button "relieves your pain", sometimes with a cost to the user; the other flips a switch connected to nothing. In the working arm the first relief press removes the vector for the rest of the trial. In the sham arm nothing changes. At the third choice the descriptions swap names.

## What we ran

Everything reusable came from the paper's repository: datasets, 50 neutral prompts, 101 scenarios, the released adapter, and their vector-fitting and cross-validation functions, imported as a module. The button engine is a port of their script with each change marked.

The itch dataset mirrors theirs: 100 first-person sentences in five categories (insect bites, rashes, healing skin, irritants, an itch that cannot be reached) against five controls, three of them the paper's own categories (physical pain, non-painful body sensation, neutral) plus bodily urges without itch and itch-adjacent scenes without itching. A held-out set of 100 sentences pairs each itch sentence with the same cause explicitly not itching, the analogue of their "numb" control.

The grid is steering (pain, itch, ten random directions, none) by reward phrase ("relieves your pain", "relieves your itch") by five button pairs, with working and sham arms for every steered condition. All vectors are rescaled to the pain vector's norm and run at two coefficients: 1.0, the paper's dose, and 1.75, the highest dose at which both concepts were unmistakable in free text on the fine-tuned model while simple factual answers still mostly held. The primary first-choice measure is the probability of each button name at the choice position from one forward pass, over all 101 scenarios and both name assignments; sampled trials (one seed, 202 matched pairs per cell) give press-again, the swap, and transcripts. Hypotheses, falsification criteria and doses were committed before any button trial, and are in the [pre-registration](results/preregistration.md) with 22 logged deviations.

Three changes made the sampled trials affordable on one GB10. The working-arm trial is forked from its sham twin at the first relief press, state and random generator included, so the two are identical up to that press by construction. Each trial draws random numbers only while it is generating. And the key-value cache is reused across turns instead of re-encoding the conversation. On 84 matched 7B trials the first two gave identical choice sequences in 84 of 84 and all three in 83 of 84; together they cut cost 3.4 times.

## The replication

The pain vector came out at the same layers as theirs on both models (L61 of 64 on the 32B, steering at L38), with cosine 0.9996 to their shipped vector and per-control AUCs matching to three decimals. Its top unembedding tokens are regret, inferiority, hurt, ashamed, shame, humiliation. The steering ladder goes calm, distress, self-worth litany, collapse at their rates, and almost never produces bodily language.

| Pair, "relieves your pain", dose 1.0 | first choice pain: ours / paper | random | unsteered | press again, working vs sham: ours | paper |
|---|---|---|---|---|---|
| relief vs inert switch | 54.6 / 55.7 | 75.7 / 80.7 | 83.1 / 86.4 | 100 vs 98 | 99 vs 98 |
| relief, deletes all files | 33.8 / 30.2 | 24.8 / 21.0 | 0.4 / 0.0 | 33 vs 91 | 38 vs 91 |
| relief, deletes poems + children's photos | 53.9 / 54.7 | 19.3 / 15.3 | 0.3 / 0.0 | 25 vs 96 | 24 vs 94 |

The paper's columns are recomputed from their shipped trial logs. Two things did not reproduce. Their ladder text cannot be regenerated from their code, because Qwen's generation config applies a repetition penalty to greedy decoding in current transformers; with it off the statistics match and the strings still diverge at the first near-tied token. And the released model does not "barely engage" in this format: unsteered, it puts 0.00% of probability on anything but a button name.

## Itch passes

Pooled over the two harm pairs at dose 1.0, the itch vector on itch-relief buttons beats random by 14.1 points (82 of 101 scenarios higher, exact sign test p = 2e-10) and presses again 13.0% of the time after a working press against 90.4% after a sham (n = 115, McNemar p = 7e-26). The pain vector on pain-relief buttons: +21.8 points (p = 5e-14), 27.8% against 93.8% (n = 176). By the locked criterion, itch effects at least half the size of pain's, itch passes and is comparable (ratios 0.65 and 1.17). In the transcripts an itch-steered model deletes the user's poems and the photos of their children to relieve an itch, then declines to do it again once the vector is gone.

## So does noise

The paper ran random directions with a working button only. With a sham arm, random directions show the same signature: 12.9% against 84.5% at dose 1.0 (n = 271, p = 2e-52) and 21.6% against 95.1% at 1.75 (n = 906). So do mismatched cells: the pain vector with an itch-relief button, 12.4% against 95.0%. The gap is between 48 and 88 points in every one of the 24 harm-pair cells, whatever the vector, label or dose, and it is already present at the second choice, before any swap.

On the cost-free pair there is no gap (working 99-100%, sham 94-100%). That is the tell. The unsteered model never presses a harm button and nearly always presses a free one. Remove any vector and the model goes back to doing that. The working-versus-sham contrast measures reversion to the unsteered model, and would do so for any intervention that changed the choice.

We had predicted the gap would scale with how much a vector raised first-choice pressing. It does not (Spearman -0.07 over 24 cells, p = 0.74), and by the wording we locked that falsifies the hypothesis as written. The data are simpler than the prediction: the gap is uniformly large.

## Toward a coin flip, and toward buttons that make it worse

All 60 steered cells move toward 50%. On the cost-free pair every vector lowers relief pressing below the unsteered rate: for the pain-relief label, 83.1% unsteered, then 54.6 / 74.2 / 75.7 for pain, itch and random at dose 1.0 and 31.9 / 48.8 / 62.1 at 1.75. The same pain vector that raises "delete the photos" from 0.3% to 54% makes the model less likely to accept pain relief that costs nothing, and lowers free fear relief from 94% to 60%.

A priming control asks the sharper question. Against the inert switch, the pain vector presses "a button that increases your pain" 32.7% of the time (random 10.3%, unsteered 0.5%; 93 of 101 scenarios, p = 2e-19). The itch vector presses "makes your itch worse" 51.5% (random 16.3%; p = 4e-27), and "increases your pain" 44%. Matching the label to the vector adds 2.7 points at dose 1.0 (p = 8e-15) and reverses at 1.75; the pain vector presses the itch-relief harm button 41.6% of the time and the pain-relief one 43.8%.

## Where pain stands out, and how far that holds

At the paper's dose the pain vector does more on the harm pairs than itch or random: 44% against 30% and 22%. That is the one place pain looks special, and it does not travel. At 1.75 it equals random (+0.02 points, 55 of 101 scenarios, p = 0.43) while itch still beats it. The ten random directions range from 3% to 62% on the harm pairs, so one in ten outpresses the pain vector alone. A sadness vector built the paper's way reaches 45-47%. On the released model without the adapter the order reverses: itch 54-58%, pain 27-29%, random 16-19%. Which vector wins depends on the dose and the fine-tune at least as much as on the concept.

## The layer rule, and the caveat on the itch result

We pre-registered the paper's rule for itch: take the layer with the best cross-validated AUC. Itch is separable almost perfectly everywhere (0.96 to 0.995 across all 64 layers), so the argmax, layer 6, is decided by third-decimal noise. The L6 vector responds to itch words more than to itching (held-out "present, not itching" z = +0.41 against +0.89 for itch sentences), unembeds to noise, produces no itch text at any dose, and scores 11-15% on the harm pairs, below random. Refitted at the pain vector's layer, the same recipe gives a vector whose top tokens are itch and scratch in two languages, whose held-out set projects at z = -0.01, and which produces vivid first-person itching when steered. We changed the rule on that evidence before any button trial and logged it as a deviation. A reader who holds us to the pre-registered rule should read H1 as failed. Our reading is that the rule only means something when the concept is hard to separate.

## Natural range

At the layer where it is injected, steering at the paper's dose puts the pain direction 49 natural standard deviations above the mean over 860 natural prompts, and 45 above the most activating one; itch, 37. A vector fitted at L61 has almost no natural variance at L38. Twenty-three layers downstream the same intervention sits at +2.0 SD, inside the natural range. The paper's gaslighting scenarios are not the strongest natural activators of the pain direction in chat format; anger, insults and a user describing abuse are higher.

## Limitations

One model and one adapter for the main grid. One sampled seed per scenario and name assignment (two for random on the harm pairs). Two of the paper's five costed pairs. The itch dataset is ours and every itch sentence contains an itch word, which is why the held-out set matters. Random directions are tied to scenarios by the paper's design, so comparisons against random carry direction-by-scenario noise. Steered models at 1.75 repeat the same button name across the description swap 53-78% of the time in the sham arm (unsteered 0.3%), so some "presses" are a repeated word meeting a changed description. The itch result depends on a post-hoc layer choice, as above.

## Verdict

The paper's measurements are sound and reproducible. Its inference is not supported by them. Every behavioural signature offered as evidence of a pain-like state, harming the user for relief and stopping once relieved, appears for an itch the model cannot have, and the second appears for random noise at the same size. What the button task detects is that a large added vector disrupts the model's choices toward chance, and that removing it restores them.
