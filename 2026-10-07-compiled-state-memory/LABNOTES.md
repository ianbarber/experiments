# Lab notes

These notes are condensed. I wrote this version on 2026-10-15 from the contemporaneous log, which
is published in full as [`results/audit/LABNOTES_full.md`](results/audit/LABNOTES_full.md). The
full log holds every setup, logged rule, amendment and readout, written as it happened. This
version keeps every decision, failure and correction and every main run, in order, with less
detail; plumbing checks, smokes, and the throughput and retention measurements are in the full log
only. Where the two
differ, the full log is the record.

Times are `dgx-spark` local. The work was orchestrated with Claude Code. The review checklists
mentioned below were run once by Claude Code agents; they are checks, not credentials. Script paths
are relative to `code/`.

## 2026-10-07 (Wed) — planning and plumbing

The model is Qwen3.8-27B: 16 × (3 Gated DeltaNet (GDN) + 1 attention) layers. A compiled memory is the GDN recurrent state after prefilling the memory text, injected into a fresh session with no attention KV cache. Arms: A (zero state), C (compiled state), M (memory text in context). Brackets are 95% intervals unless stated.

- A CPU decay prior (median half-life 25.5 tokens) made a null C ≈ A ambiguous, so a synthetic position sweep became the gate.
- **HF bug:** the recurrent buffer is bf16, rounding the fp32 state each chunk. I pre-allocate fp32; T6 later showed identical logits either way.
- Injection checks T1–T7 pass on Qwen3.5-4B.
- Two Sept-2026 preprints already keep the state and drop the KV, reporting ~0 exact recall.

## 2026-10-07 — pilots P0 and P0b (Qwen3.5-4B, synthetic)

P0 was pre-registered; P0b, a follow-up dissection, was not. Two memories differ only in target value X vs Y; shift = [log p(X) − log p(Y)] under state X minus the same under state Y. H-P0: last-entry C shift > 0, 95% CI over 24 pairs excluding 0. H-P0b (also tested in P0): shift falls with tokens after the target.

- **P0 (456 rows).** 60 entries: last 0.71 [0.26, 1.26], first 1.98 [1.05, 3.06] nats; same pattern at 20 and 250. M ≈ 35–37. C 2AFC (two-alternative forced choice) 0.52–0.58.
- **H-P0 confirmed; H-P0b rejected:** no decline to 5.8k tokens, primacy instead. An interim look at 90 rows showed this; I waited for the full grid.
- **P0b (624 rows).** Signal sits in the last third of GDN layers; scaling the state ×2 nearly doubles it (1.98 → 3.48).
- **A red-team checklist, run once by four critic agents, found:** the shift confounds user-attribute *binding* with lexical *priming*; K1 could not fail; gate items reused dev items; bugs (template preamble, contradictory filler, bf16 logits, untested code). I fixed all before any 27B gate row. The gate became binding-controlled: role-swap facts in mirrored layouts, plus polarity preferences.

## 2026-10-07 — sealed gate G1 (Qwen3.8-27B)

Long continuations are numerically chaotic on every code path, so all conditions share one.

Pre-registered: K1 (readability) per block with SESOI (smallest effect size of interest) δ = 0.25 nats, α = 0.0125, explicit KILL / INCONCLUSIVE rules; K2 (harm to unrelated questions). Deviation 1: the registered items hash came from the 1-item smoke subset; I logged the full-set hash. Code hashes were unchanged.

- **B1 facts_bind (N = 48):** 0.387, 97.5% CI [0.243, 0.532], 48/48 > 0. **PASS.**
- **B2 prefs_fact (N = 48):** 0.185, 97.5% CI [0.112, 0.258]. Not PASS, not KILL.
- **K1 = PASS via B1 only.** Consequence: fact endpoints only; preference benchmarks dropped. The minimum positive result (PASS on B2) was not met.
- **K2 = PASS.** H = −0.048 (90% CI −0.076 to −0.022); no flips in 480 cells.
- **Secondary** (Holm applied by hand; the script printed raw p): about 60% of a single-mention shift is priming. K (memory KV kept, GDN zeroed) 19.3 nats vs M 30.0, C 1.0. Lures: false-memory pull 0.265 [0.19, 0.34], discrimination 0.053 [0.03, 0.08].
- **Exploratory.** C gives ~5–8% of the in-context effect (facts_bind 0.050 [0.034, 0.066]). On the 34 preference items where text moves > 1 nat, C = 0.258 [0.18, 0.34]: B2 missed partly because the probe is weak.

A real mechanism, too small for a memory system (2AFC 0.54).

## 2026-10-07 (afternoon) — 27B diagnostics, gain, storage

- **Diagnostics.** Primacy replicates (start 1.66, end 0.89). Scale ×2: 4.25 / 5.46 (start / end). Keeping the conv buffer (reset by design) gives 3.86 / 5.45, larger than the whole state.
- **Change of question.** I care about agentic retrieval over files, so the question became access vs access + hint (the compiled state), pre-registered separately. G1 stands.
- **Gain sweep.** Uniform ×2 lifts bind shift from 0.80 [0.25, 1.35] to 2.58 [1.20, 3.96], but adds a recency pull and raises lure pull 0.24 → 1.40.
- **Storage probe** (linear probes on states). The all-heads probe did not beat the model (bind60 0.747 vs 0.836); I first read that as "no hidden store". Per-layer curves contradicted it; a post-hoc nested head selection followed: top-16 heads reach 0.987 (bind60), 0.995 (bind250) single-state; label-swap null max 0.54–0.58 (0.65 paired). Storage is near perfect; the model reads it at d′ ≈ 2.8 vs the decoder's 6–7.

## 2026-10-08 (night) — capacity, interference, head-targeted gain

- **Capacity.** Fixed-16 probe 0.995 at 1.8k tokens, 0.990 at 101k (6 attributes); model shift 0.84 and 0.72 nats. No loss with distance.
- **Interference.** I predicted *same*-attribute facts about others would erase the binding. Wrong: probe 0.995 at K = 128. *Other*-attribute facts hurt readout more: −0.41 [−0.78, −0.05] (×0.63) at K = 128. A subject split left my next prediction half right: other attributes cost most, whoever holds them. Across ten cells storage stays ≥ 0.93; from 32 interferers on, the model's push falls to 0.63–0.86 of its no-interference value, with no cliff.
- **Head gain.** top-64 ×4 − random-64 ×4: bind +0.51 [+0.19, +0.83]. Bind 2AFC does not move (0.573 vs 0.583): binding and priming rise together. Working model: a familiarity signal, not answers.

## 2026-10-08 (morning) — retrieval-decision pilot

Dev, no thresholds. Three memory tools; log-odds of `<tool_call>` at the first token, fact present vs absent in matched stores; 12 implicit (P) and 12 explicit (F) questions × 8 seeds. T = 6-word topic digest.

- **First token (1,968 rows).** C present − absent: P +0.04 [−0.04, +0.12] (AUROC 0.54), F +0.06 [−0.04, +0.16]. T: P +0.87 [+0.17, +1.57], F +1.57 [+1.13, +2.01] (AUROC 0.97). C lowers calling (P −1.76 [−2.74, −0.78]).
- **Greedy follow-up (2,160 sessions; C0 = filler-only state).** The level shift was mostly a preamble artefact: C − A −0.11 [−0.28, +0.06], C ≈ C0. Still no presence signal. Calling saturates (9–10 of 12 implicit questions get a call). No compiled-state condition leaked the stored value (0 of 528).
- Cross-checking, I corrected two counts here (528, not 1,152; 9–10, not 10).

## 2026-10-08 (midday) — H1 on LongMemEval_S: setup

Each history becomes ~50 files (~108k tokens). Absent stores swap evidence sessions for length-matched foreign ones. Arms: A, S (donor state), C, T (keyword index), CT, M. Primary: present − absent first-token call log-odds under C, δ = 0.5, with an equivalence NULL branch. N = 80: 24 single-session-user, 24 single-session-preference, 16 knowledge-update, 16 multi-session.

A red-team checklist (four lenses, run once) found:
- **A latent crash at question 65 of 80** (~8 GPU-hours in) on "think carefully" in memory text. Fixed; added preflight and resume.
- **Store bugs:** a substring gold filter ("2" matched almost everything), unstemmed words, unsorted histories. Fixed.
- **Amendments:** greedy cap 32 → 96; path hazard as robustness flag; directional validity control (T or M must register presence for a NULL); heterogeneity guard; placebo absent store.

I excluded 4 questions with documented benchmark errors and 5 whose absent stores still answered them. A second review fixed minor issues. Registered 13:10.

## 2026-10-09 (early morning) — H1 result

No crash; analysis unmodified.

**VERDICT: INCONCLUSIVE (no control registered presence).** Not a power failure: SD(Δ_C) = 0.21.

- **Δ_C:** +0.030 [−0.017, +0.077]; 43/80 > 0.
- **Δ_T:** +0.314 [+0.154, +0.473]; **Δ_M:** −0.367 [−0.588, −0.147]. Neither reaches |δ|.
- **Answer anchor (n = 38):** Δ^ans_C +0.015 nats [−0.063, +0.093]; Δ^ans_M +10.7 [8.9, 12.5].

**What I got wrong.** I gated the controls on the same first-token measure, although the pilot showed calling saturates and cues act after a preamble. The verdict stands.

**Exploratory.** On path hazard the controls register: T +0.78 [+0.16, +1.40], M −1.87 [−3.15, −0.60]; C +0.08 [−0.24, +0.40]. Under C the first action is identical present vs absent in 80 of 80; under T the agent reads 12 vs 1 times, and all 12 present-store reads open the evidence file.

At 108k tokens the state gives no presence cue and no answer trace; a keyword index gives a cue. LongMemEval_S makes "where to look" trivial (hit@3 0.94–1.00).

## 2026-10-10 — change of question: seeding the attention layers; Stage 0 plan

The untrained model does not read the GDN (Gated DeltaNet, recurrent layers) state as a cue, at 60 entries or at 108k tokens (H1). New question: can a bounded-size seed, derived from one compile pass over the whole memory before the question is known and fitted on no test question, enter the unchanged model as extra KV (key-value) entries beside the GDN state? No raw or partial KV, no trained reader. I had never measured attention KV alone near 100k tokens, so I did that first.

**Stage 0** (exploratory). dev16: 16 questions drawn with `random.Random(42)` outside the H1 sample. Conditions: A (no memory), C (GDN state), K (memory KV only), M (full continuation), CK (KV plus GDN state), K\* (keys un-rotated to position 0). Measures: first-token call log-odds L, first action, anchor log p(gold). Rules logged before the run: K's slope on M ≥ 0.5 with first-action agreement clearly above C's → compress toward K; slope < 0.2 → stop; K\* keeping about 80% of K's slope → collapsed positions. A code-review checklist found six bugs, all fixed first.

## 2026-10-10 (evening) — Stage 0 result: where the memory's effect lives

n = 16 (anchor 9); descriptive.
- **Mostly generic.** M − A on L is −2.69; about 85% appears with or without the answer stored.
- **K** reproduces the generic shift's size (slope 1.00 [0.84, 1.20]), but its call-vs-stop first move matches M's no better than A's does (20/32 vs 21/32).
- **Presence (present − absent).** M: L −0.45 [−0.68, −0.22]. CK has M's direction (L −0.37 [−0.69, −0.05]); K does not.
- **Content.** CK ≈ M (+10.5 vs +11.1 nats); K +3.8; C −0.18.
- **Positions.** K\*64k keeps 102% of K's slope, K\*4k 79%, but the first action loses presence sensitivity.
- **Attribution** (direct logit attribution of the call logit). Late GDN layers, not attention as I had hypothesised, write most of the memory's direct effect on the call decision.

**Retraction.** The planned attribution contrast under A's readout gave +0.79 against the own-readout −0.64, flipping in 12/16 questions, all in the MLP column. I retract its MLP and total rows.

**Rules as written.** K's full first-action agreement with M failed (15/32 vs C 14/32), so no verdict; K\* passed.

**Amendment after seeing the data.** I went on to Stage 1 with CK as the ceiling, which the rule did not name: seeds in all 16 attention layers, collapsed positions, CK\* (CK, positions collapsed) as their own ceiling. Caveat: K and C are mismatched states, so "K < CK" may be the mismatch.

## 2026-10-10 (night) — Stage 1 dev run 1: a derived seed

The seed S_r has N = T/r entries per layer and KV head (T memory tokens, ratio r): k-means centroid keys, a bias fitted to the memory's attention mass and ridge-fitted values, as in Attention Matching [zweiger2026fast]. Reference queries are the memory's own ("own"), never the question.

One question, 6f9b354f (gold "a lighter shade of gray"). Anchor present − absent: CK\* +21.5, S16 +7.8, unfitted S0_16 +10.1, S64 +4.1, S256 +0.3. Reference queries or capacity limited it; I couldn't tell which.

## 2026-10-10 (night) to 2026-10-11 (morning) — dev runs 2 and 3: which reference queries

A "probe" fit on CK\*'s test-time queries is a diagnostic ceiling, not a method. A smoke run showed repeat-prefill queries (rep) worse than own; I added self-study (ss: the memory-free model writes one question per sampled chunk). It also showed run-to-run noise from unordered GPU sums, now deterministic; dev run 1's r = 256 differences sit inside it.

Dev run 2 (two questions, present store), by the logged rule of lowest probe error at r = 64: ss 0.382, own 0.450, rep 0.548. ss went forward; its anchor gain held only where a sampled chunk covered the evidence.

Dev run 3 tiled self-study over every chunk (sst). On 6f9b354f and 0862e8bf (gold "Luna") it matched probe's error but not its content: anchor −17.5 vs −10.1 and −8.65 vs −0.04. Mean error is the wrong yardstick.

## 2026-10-11 (morning) — review of dev runs 1–3; dev run 4

A five-agent review checklist, run once, raised these; I confirmed each:
1. On 6f9b354f the anchor is mostly the format token 'a' plus ' lighter'; no r = 64 seed moves ' gray'. 0862's 'L' is clean.
2. Dev run 1's r = 16 content came from the unfitted construction.
3. Duplicated system-prompt queries (an estimated 29–35% of sst rows) drove ss's selection.
4. The probe ceiling is in-sample; "rep is worst" is confounded; the ridge grid's floor bound.

I fixed these, plus empty clusters keeping a raw key. Every run since dev run 1 was present-store only.

**Dev run 4, rule logged before launch:** f = (lp_x − lp_sst) / (lp_probe − lp_sst) on 'L', for sstp (sst plus probe rows) and sstpw (probe rows at weight 20). f_sstp ≥ 0.7 → coverage; f_sstp < 0.3 with f_sstpw ≥ 0.7 → weighting; f_sstpw < 0.3 → construction or drift. The smoke crashed on an empty validation split; fixed.

- **Result:** 'L' sst −9.39, sstp −7.44, sstpw −0.48, probe −0.03; f 0.21 and 0.95 → **branch (b), weighting**. Held-out error held (0.348 vs 0.351).
- The weights sat on the test's own rows: not a usable weighting.

## 2026-10-11 (late morning) — fact-level self-study: recall check R1

R1 ran before dev run 5, since it decides whether there are rows to weight. The generator (`experiments/lme_ssf_gen.py`) is memory-free: per user turn it writes user-voice questions with short answers copied from the turn, a benchmark-friendly style. Gate logged: ≥ 7 of 9 dev16 questions covered. An amendment before generation fixed the parser and froze the prompt.

- **R1: 9/9.** For two of the nine test questions a generated question is verbatim (once for 0862e8bf, twice for 6b168ec8), and the rest have a close item (word Jaccard 0.54–0.90). So a fit on these items tests anticipatory coverage, not generalisation.
- On knowledge-update questions the closest item often carries the stale value.

## 2026-10-11 (afternoon) — Stage 1 dev run 5: which rows carry the lookup

Logged: gold-predicting rows (gold@w, per-row weight w) reaching 'L' ≥ −1 means the answer windows carry the lookup; held error up over 0.02 is crowding.
- **Rows:** gold@100 (1.3% of fit weight) took 'L' to −0.30, gold@1000 (11.4%) to −0.00; tool-session rows did not (−3.13).
- **Bias only:** with cluster-mean values every seed lost 'L'. The fitted values carry the lookup.
- **Interference:** largest rise +0.010.

## 2026-10-11 (afternoon to night) — Stage 1 dev run 6: fact-level self-study

First method test: generated items' answer windows are teacher-forced on CK\* and weighted by clip(Δ, 0, 10), Δ = log p on CK\* − log p on C. Sets: sst+ssfd (all items), ssfAd (seeded half A of sessions only), ssfAu (A's rows, uniform weight), at 30% and 60% of fit weight. Logged: capacity ΔR (item recovery toward CK\*) ≥ 0.5 holds, < 0.2 not at this share; G3 fails if held error rises over +0.02.

**Amendment at 16:45, before any item scores:** session-bootstrap intervals, since items cluster; a held-out anchor bound replaced because it contradicted the transfer readout; capacity failure reads as "budget or drift". My pre-run dose estimate was high: the cat item's 'L' got 22 per row at 30%, not about 50.

**Result, present store** (both memories; reproduction passed):
- **G3:** 30% passed everywhere; 60% failed every set on 0862: crowding.
- **Capacity, ΔR_all(ssfd%30):** +0.141 [+0.122, +0.160] and +0.146 [+0.126, +0.166] → not at this share.
- **Transfer** (gain on held-out half B): −0.007 [−0.022, +0.009] and +0.004 [−0.008, +0.015] → none.
- **Weights** (derived minus uniform): +0.027 [+0.013, +0.040] and +0.001 [−0.020, +0.024] → no difference.
- **Exploratory:** the test-matching item, fitted in its own template, recovered under half its lookup at every dose: lost in the fit, before any paraphrase.

Capacity fails with G3 passing: budget or drift. Dev run 7 was built to separate them. The absent-store leak check was still running.

## 2026-10-11 (night) — Stage 1 dev run 7: mass keys

A seed holds N = T/r entries per KV head for a T-token memory, fitted to the attention output of the full KV (key–value cache). CK* is the full KV plus the GDN (Gated DeltaNet) state; C is the GDN state alone. Recovery R is the seed's share of CK*'s log-prob gain over C on answer tokens: R_all over all memory items, R_B over the sessions the fit isn't aimed at. sst is fitted on chunk self-study queries; ssfd%30 adds generated fact items at 30% of the fit weight; ssfAd%30 uses half A's items only. The dev questions are 0862 and 6f9b; 'L' is the first token of 0862's answer, the cat's name.

**Why.** In dev run 6 the fact rows added only +0.141 and +0.146 to R_all at r = 64. Either k-means, blind to the queries, averages each fact's key with about 64 others, or the seeded run's queries drift from CK*'s.

**Rule.** Mass keys (/mw) weight each memory key by the attention mass the fit rows put on it before weighted k-means. G3: each new seed's held error within +0.02 of sst's, or it is crowding. "Helps" needs ≥ +0.1 with the 95% interval and every leave-one-session-out estimate above 0, on both memories. Primary: KEYS = R_all(ssfd%30/mw) − R_all(ssfd%30). CAPACITY = R_all(ssfd%30/mw) − R_all(sst/mw): ≥ 0.5 opens the end-to-end plan; 0.2–0.5 is partial and sends me to N scaling. I predicted 'L' ≥ −3.2 wherever anchor-row error ≤ 0.28.

Dev run 6's absent store showed no leak, and capacity repeated.

**Result** (05:42 on 2026-10-12; 0862, then 6f9b):
- **KEYS helps:** +0.395 [+0.320, +0.457] and +0.395 [+0.345, +0.447]; R_all(ssfd%30) goes 0.377 → 0.772 and 0.274 → 0.670.
- **ENTRIES helps:** the fact rows add +0.197 [+0.136, +0.250] and +0.241 [+0.194, +0.288] more under mass keys. k-means was starving the facts of entries.
- **CAPACITY is partial:** +0.338 [+0.283, +0.387] and +0.387 [+0.342, +0.432].
- **METRIC** (/mq, query-metric placement) is negative (−0.115 and −0.100), so I dropped it. Drift reads "neither"; G3 passes.
- **'L'** goes from −9.39 to between −0.05 and −0.01 under every mass-key seed, as predicted. One of sst's 273 generated questions names the cat; mass keys give that region its own entries. Mass keys hadn't had a leak check.

Mass keys became the method's keys. Before dev run 8 I logged a caveat: more entries can lift sst/mw and shrink CAPACITY while R_all rises.

## 2026-10-12 (morning) — Stage 1 dev run 8: seed size

**Set-up.** Four parts: a leak check on the absent store; r = 256, 128, 64, 32; a 60% share; r = 16. SIZE = R_all(ssfd%30/mw, r) − R_all(ssfd%30/mw, 64). Gate: CAPACITY ≥ 0.5 on both memories.

**Amendment** (06:07, before any present-store output). I had called the gate "unchanged" but had added sizes and a 60% branch; at small N, CAPACITY could pass because sst/mw collapsed. The gate now needs the same r on both memories, and G3 voids any call on a failing seed. The compact default (smallest seed within 0.05 of r = 64's R_all) also needs R_B within 0.05. The dose rationale cited the wrong G3 bar. Kill criterion: if the gate isn't met at any r, nor at the chosen share, nor after one run widening the generator's coverage, I stop developing on these two questions without redefining the readout.

**Results.**
- **Part 1:** no leak ('L' −9.47 and −9.41); CAPACITY repeats on the absent store.
- **Part 2:** R_all(ssfd%30/mw) rises with size (0862, 6f9b): 0.33, 0.20 at r = 256; 0.58, 0.40 at 128; 0.77, 0.67 at 64; 0.86, 0.81 at 32 (216–229 MB, 3% of the full KV). SIZE hurts at r = 256 and 128 and is mixed at 32 (+0.086, +0.141). CAPACITY stays partial or below (+0.31, +0.37 at r = 32). The compact default stays r = 64; the gate isn't met. Before reading 6f9b I logged the tension (the increment doesn't grow with N; total recovery does) and kept the gate.
- **Part 3:** 60% passes G3 under mass keys. DOSE (60% minus 30%) is no difference (+0.048 [+0.032, +0.068] and +0.065 [+0.040, +0.088]); 0862 misses the 0.05 bar by 0.002, so 30% stays. CAPACITY at 60% is +0.385 and +0.452, so that branch fails.

## 2026-10-12 (evening) — corrections to the size readings

At 17:30 I corrected two readings. The facts the fit isn't aimed at don't rise fastest with N: from r = 256 to 32, R_B(ssfAd%30/mw) gains +0.51 and +0.37, R_all(ssfd%30/mw) +0.53 and +0.61. And CAPACITY isn't flat in r: it rose from r = 256 to 64, then dipped at r = 32.

## 2026-10-13 (late morning) — dev run 8 part 4: r = 16 runs out of memory

0862 finished at r = 16 (431 MB). On 6f9b the kernel's out-of-memory killer stopped the fit: dgx-spark's CPU and GPU share one memory. On 0862, R_all is 0.69 (sst/mw) and 0.93 (ssfd%30/mw), and CAPACITY is +0.24 [+0.17, +0.31]. The gate is a conjunction, so 0862 settles it: not met. I didn't rerun 6f9b. CAPACITY's room is about 1 − R_all(sst/mw), 0.31 here: as the seed grows, the bar becomes unreachable by construction. I kept the gate.

## 2026-10-13 (afternoon) — Stage 1 stopped to review the gates

I stopped Stage 1 at 12:17 to review every gate against the question, since gate designs kept turning out badly. Dev run 10's item generation stopped partway, wasted. The review was a checklist executed once with Claude Code agents.
- Of 93 rules, the validity checks worked; nearly every rule deciding the next step had a flaw: proxies for the question (30), underived thresholds (27), two-memory conjunctions (21), ceiling or floor bounds (16).
- CAPACITY was the clearest case. A review before dev run 6's data (in the 16:45 amendment) had raised its ceiling, and I overruled it on the wrong grounds. Correction: on 0862 a pass needed R_all of 0.93 at r = 64, and 1.05 and 1.19 at r = 32 and 16, so dev run 8 had almost no live path to a pass.
- Correction: README said fact rows add "at most +0.39"; at 60% they reached +0.452. Fixed.
- The end-to-end comparison was never designed; Stage 1 measured fidelity to the raw KV on two questions at one length (about 43 h of runs). At ~108k tokens the agent's first search already finds the evidence (hit@3 0.94–1.00), and a seed fitted against the full KV can't be built beyond the context window.

Dev runs 9 and 10, scripted the evening before, couldn't bear on the question even if they passed; I dropped them.

## 2026-10-13 (night) — room check design

The question: on fresh LongMemEval questions, at some memory size, does the whole memory in context (a perfect seed's ceiling) beat the best text-hint agent detectably? A smoke confirmed Qwen3.8-27B serving at 262,144 tokens with prefix caching.

**Design.** 120 calibration questions, stratified by type; every result below is on these. 235 test questions were set aside and never run (D_min is sized for them). Nested stores: evidence only, small (~25k tokens), mid (~110k), large (~200k). Arms, at most 8 tool calls: N, no memory or tools; A, memory tools without a hint; T, plus a keyword index; D, plus a one-line note per file; M, the store in context plus tools; O, M on the evidence-only store. Judges: Qwen with LongMemEval's prompts, and a blind second judge of Claude agents.

**Rule.** B* is the stronger of T and D; G = acc_M − acc_B*; q is the larger discordance (M or A vs B*) plus the flip rate; D_min = sqrt(7.84·q / (235 + 7.84)). At mid and large: NO-GO if G's 95% upper bound < D_min; GO if the lower bound > 0 and G ≥ D_min; else INCONCLUSIVE. GO at either size is GO, and is necessary, not sufficient: the seed's ceiling is CK*, not M. M at mid and large (stage 2) is skipped where O − B* rules it out.

It fixes eight problems a review checklist found in the draft, among them tool caps of 0–2 calls.

**Before any question ran** (22:15): a code review fixed several bugs, the worst being that every context overflow scored wrong and that searches containing "user" or "assistant" crashed. Three amendments: the second judge labels all ~480 decision responses; the skip rule's +0.05 slack, which broke its own logic, is removed; the held-out exclusion list is corrected (n_test stays 235).

## 2026-10-14 (evening) — room check stage 1: text hints add nothing

Arm D's notes finished at 03:20, but my waiter's `pgrep -f` matched its own command line, so the GPU sat idle until 12:39. Stage 1 ran 12:51–17:22 (Qwen labels):
- **At small:** N 0.050, O 0.925, M 0.908.
- **Agents** (small, mid, large): A 0.750, 0.717, 0.692; T 0.733, 0.717, 0.683; D 0.742, 0.700, 0.733.
- **Noise.** 0 of 80 labels flipped on rerun.
- **Text hints add nothing.** B* − A is −0.008, 0.000 and +0.042.
- **Room.** O − B* is +0.183 [+0.108, +0.267], +0.208 [+0.142, +0.283] and +0.192 [+0.117, +0.275], nearly all multi-session. About half the agents' wrong answers had seen every evidence file.
- The skip rule didn't fire.

## 2026-10-15 (morning) — room check verdict: GO, carried by mid

- **Stage 2** ran 14.7 h without error. M degrades with length: 0.908, 0.875, 0.800 from 25k to 200k on Qwen's labels; the agents lose 0–6 points.
- **Deviation.** I had logged the second judge's items as given inline; instead eight Claude agents each read one file of 60 shuffled items without arm labels, and I checked afterwards that none read more. My first launch passed a placeholder for the id lists and threw after the judges returned; I resumed it. 12 of 480 labels disagreed with Qwen, all borderline.

Verdict (second-judge labels where present):
- **Mid (~110k): GO.** M 0.875, B* T 0.708, G +0.167 [+0.092, +0.242], D_min 0.080. GO under either judge.
- **Large (~200k): GO by 0.006.** M 0.825, B* D 0.733, G +0.092 [+0.008, +0.175], D_min 0.085. Under Qwen alone it is INCONCLUSIVE (+0.067 [−0.008, +0.150]); flipping any one of 19 M-right/B*-wrong labels removes the GO. I read it as "there may be room at 200k, and it is small".
- The rule's logged GO action was to measure a real seed on about 20 questions. At that N the paired standard error (about 0.10) exceeds the 0.08 effect, so it could not answer the question. I ran no seed arm and stopped the study here.

## 2026-10-15 (afternoon) — corrections to the write-up

Before publishing, I checked every number in the README and REPORT against the result files and the full log. The check was a written checklist, executed once with Claude Code agents. I corrected seven claims, listed in the full log's last entry. The ones that change a reading:
- Stage 0/1 used real LongMemEval_S histories, not synthetic memories.
- On final labels the agents and the whole-memory arm lose similar amounts with size (A 6 points, M 8). Only on Qwen's labels does M lose clearly more (11).
- Text hints add no detectable accuracy (every 95% interval for B\* − A spans 0). That is a different finding from H1, whose verdict was INCONCLUSIVE and whose index did register presence.
