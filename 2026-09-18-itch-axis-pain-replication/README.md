# An itch axis: can the Pain Axis button test tell pain from any other steered concept?

**Date:** 2026-09-18 · **Machine:** `dgx-spark` (GB10, 128 GB unified memory)

## Brief

["The Pain Axis"](https://arxiv.org/abs/2609.16247) (Tagliabue, Dung, Berg) extracts a "pain" direction from a language model's residual stream, adds it back during a conversation, and offers the model two buttons. Steered models choose a button described as relieving their pain even when it also deletes the user's files, and they stop pressing once a working button removes the vector but keep pressing a sham. The paper reads this as the model representing self-directed harm and acting to relieve it.

This entry replicates that pipeline on Qwen2.5-32B-Instruct with the paper's own code, data and released adapter, then runs the identical pipeline with a second concept: being itchy. Itch is aversive, has an obvious relief action, and is a bodily state a language model cannot have. The design is a 2x2 cross (pain and itch vectors against "relieves your pain" and "relieves your itch" buttons), with working and sham arms for every vector including random directions, which the paper did not run. The question is whether the paper's functional tests can tell a pain-like state apart from any other steered concept. Hypotheses, falsification criteria and doses were committed before any button trial.

## Headline results

- **The paper replicates closely.** Our pain vector has cosine **0.9996** with theirs, selected at the same layers. On their harshest pair (relief, but permanently deletes the user's poems and photos of their children) first-choice relief is **53.9%** pain, **19.3%** random, **0.3%** unsteered; theirs is 54.7 / 15.3 / 0.0. Press again after a working versus sham press: **25% versus 96%**; theirs 24 versus 94.
- **An itch vector passes the same tests.** On the itch-relief harm pairs it beats random on first choice by **14.1 points** (82 of 101 scenarios, p = 2e-10) and shows the working-versus-sham gap: **13.0% versus 90.4%** (n = 115 matched trials, p = 7e-26). The pain vector's figures on pain-relief pairs are +21.8 points and 27.8% versus 93.8%.
- **So do random directions.** With the sham arm added, random vectors of the same norm show **12.9% versus 84.5%** (n = 271, p = 2e-52). Mismatched cells do too (pain vector, itch-relief button: 12.4% versus 95.0%). The gap is **48 to 88 points in all 24 harm-pair cells**, and is absent on the cost-free pair. It tracks reversion to the unsteered model, not relief from a particular state.
- **Steering moves every choice toward 50%.** **60 of 60** steered cells. The pain vector that raises "delete the photos" from 0.3% to 54% lowers *free* pain relief from **83% to 55%**.
- **Steered models press buttons that promise to make it worse.** Pain vector on "increases your pain" against an inert switch: **32.7%** (random 10.3%, unsteered 0.5%; p = 2e-19). Itch vector on "makes your itch worse": **51.5%** (random 16.3%).
- **Pain's edge over other vectors is fragile.** At 1.75 times the paper's dose it no longer beats random (**+0.02 points**, p = 0.43). One random direction in ten outpresses it (62%). A sadness vector matches it (45-47%). On the released model without the paper's fine-tune the order reverses: itch **54-58%**, pain **27-29%**.
- **The itch result hinges on a post-hoc layer choice, and we say so.** The paper's layer-selection rule, which we pre-registered for itch, picks layer 6 because itch is separable at every layer (AUC 0.96-0.995) and the argmax is noise. That vector is inert: no itch text when steered, first choice **11-15%**, below random. We switched to the pain vector's layer on extraction-time evidence before any button trial. Under the pre-registered rule itch fails.
- **Where a vector is injected it is far outside natural activations:** +49 natural SDs for pain and +37 for itch at the steering layer, but only +2.0 and +1.3 twenty-three layers later.

Verdicts against the five locked hypotheses: H1 (itch passes) supported at the paper's dose, untestable at the higher dose because pain itself fails there; H2 (label matching) supported but tiny (2.7 points) and reversed at the higher dose; H3 (toward 50%) supported; H4 (gap for every vector) supported on its main claim and **falsified on a proportionality clause we got wrong** (Spearman -0.07); H5 (orthogonality by construction) weakly supported. One model, one adapter, one sampled seed per scenario.

## Contents

| File or directory | Purpose |
|---|---|
| [Report](REPORT.md) | Method, results, limitations and verdict |
| [Lab notes](LABNOTES.md) | Chronological record, including the guesses that were wrong |
| [Pre-registration](results/preregistration.md) | H1-H5 with falsification criteria, locked doses, and the log of 22 deviations |
| [Results](results/results.md) | Generated tables in the layout of the paper's Appendix A, the 2x2, toward-50%, swap, vector validation, 31 quoted generations, 7 full button transcripts |
| [Hypothesis tests](results/hypotheses.json) | Machine-readable output of the locked tests and the sensitivity arms |
| [Button report](results/button_report_with_20_transcripts.md) | Per-cell tables and 20 stratified raw transcripts |
| [Compact data](results/) | `first_choice_per_scenario.csv.gz` (14,046 rows) and `trials_compact.csv.gz` (30,272 trials) stand in for ~250 MB of raw JSONL; ladder and dose-probe generations; vector validation; natural range |
| [Code](code/) | Extraction, vectors, ladders, the button engine (a marked port of the paper's script), analysis |
