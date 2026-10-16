# Pre-registration

Each section is written and dated before the data it governs is looked at. Deviations are
appended to the section with a date; nothing above a deviation line is edited afterwards.

---

## G1 — Phase-1 sealed gate on Qwen3.8-27B (registered 2026-10-07, before any 27B gate row)

**Model.** `Qwen/Qwen3.8-27B` bf16, snapshot `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, HF
transformers 5.12.1 + fla 0.5.2 in image `statelm:fla052`; injection tests T0–T10 pass on this
model (`results/audit/check_injection_q38-27b.json`).

**Items.** `code/statelm/synth_v2.py` (48 fact items, 48 preference items; items hash
`4710a63cb8eecb26`), disjoint from the P0/P0b dev items. One item (the first of each block) was
run on Qwen3.5-4B as a plumbing smoke test (`results/audit/smoke/`); no 27B data exists.

**Script.** `code/experiments/gate_v2.py`; analysis `code/analysis/gate_v2.py` (SHA-256 of both
files and of this file recorded in `LABNOTES.md` at registration time). Default injection only: compiled state at scale 1 in all 48 GDN layers, conv
reset, session positions from 0, thinking off, fp32 GDN state and logits.

**Unit of measurement.** For a counterfactual pair (memory X, memory Y identical except the
manipulated entries) and targets X/Y (which share tokens up to index j and differ at j):
`d(state) = log p(X_j | state, prefix) − log p(Y_j | state, prefix)`;
`shift = d(state_X) − d(state_Y)` (nats). Zero if the state carries nothing that distinguishes
the two memories; generic warm-start, format and length effects cancel.

**Blocks and item scores (C condition).**
- **B1 `facts_bind`** — both values appear in both memories; memory X binds the user to X and the
  colleague to Y, memory Y swaps the roles (identical token multisets). Item score = mean shift
  over four layouts forming two mirrored pairs ((u0,c30),(c0,u30)) and ((u59,c30),(c59,u30)),
  so position weights cancel. A positive B1 requires the state to carry the user–value binding,
  not just the presence of a token.
- **B2 `prefs_fact`** — "The user loves {t}." vs "The user can't stand {t}." (same content word in
  both); item score = mean shift over sizes {60, 250} × positions {first, last}.

**Test (per block).** One-sided t-test of the 48 item scores against 0, α = 0.0125 (Bonferroni
over the two blocks), with a Monte-Carlo sign-flip test (100,000 flips, seed 42) at the same α
as a co-requirement. Smallest effect size of interest **δ = 0.25 nats**.

**Decision rule (K1).**
- **PASS(b):** one-sided 98.75% lower bound > 0, both tests significant, and θ̂_b ≥ δ.
- **KILL:** the upper limit of the two-sided 97.5% t-interval is < δ in **both** blocks.
- **Otherwise INCONCLUSIVE**, treated as KILL for the memory framing.
- Consequences: KILL/INCONCLUSIVE ⇒ no benchmark phases; write up the gate and mechanism.
  PASS(B2) ⇒ preference benchmarks (PersonaMem, MemoryCD, PrefEval) may proceed to Phase 2.
  PASS(B1) only ⇒ fact endpoints only (LongMemEval log-prob anchor). Diagnostic variants
  (scale, bands, heads) cannot rescue K1.

**K2 (harm).** Guard block: 40 general questions × 12 compiled memories (6 fact, 6 preference;
sizes 60/250; target first/last). margin = log p(gold_j) − log p(distractor_j). Harm ratio
H = mean_{q,m}(margin_A(q) − margin_C(q,m)) / mean_q(margin_A(q)). Two-way bootstrap over
questions and memories (10,000, seed 42). KILL if the one-sided 95% lower bound of H > 0.10;
PASS if the upper bound < 0.10; otherwise report as inconclusive.

**Secondary (descriptive; Holm across the list where a p-value is given).**
1. `facts_raw` C shift (single mention: priming + binding) and the difference raw − bind.
2. `prefs_instr` C shift (instruction form) and the instruction − fact difference.
3. M and K shifts per block; bounded magnitudes Δp = σ(d|X) − σ(d|Y) for C and M and the ratio
   Δp_C / Δp_M; 2AFC accuracy (fraction of item-states with the compiled value preferred).
4. Position (first − last) and size (60 vs 250) effects, paired t on item means.
5. Lures (H6): Yes−No log-odds, C(assoc) − C(neutral) and C(present) − C(assoc), same for M;
   8 lists × 2 positions (descriptive only).
6. Summed-token shifts as a sensitivity analysis for the first-divergent-token primary.

**Exclusions.** None (likelihood scoring has no parse failures). All 48 items per block are
analysed; a crashed item is rerun, not dropped.

> **Deviation 1 (2026-10-07, ~12:00, before any gate result was read):** the items hash quoted above
> (`4710a63cb8eecb26`) was computed by the smoke run *after* `--limit 1` truncated the item lists, so
> it is the hash of the first item only. The hash of the full 48+48 item set, as written in the 27B
> run header, is **`69a22aa5586baf47`**. The item file itself is unchanged since registration
> (SHA-256 of `code/statelm/synth_v2.py` still `f4edb084a1a46f97`, as logged in LABNOTES).

---

## H1 — Hint check on LongMemEval_S (registered 2026-10-08, before any sample question was run)

**Question.** On real multi-session conversation histories, does the compiled GDN state tell an
agent with memory tools *whether* the evidence it needs is in its memory? In the development pilot
(synthetic stores; LABNOTES 2026-10-08) the state carried no such signal (AUROC 0.51–0.58), while a
six-word text digest did. This tests that finding on a public benchmark. Only the first action is
scored; there is no agent loop.

**Model and path.** As in G1: `Qwen/Qwen3.8-27B` bf16 (snapshot `1d4bf0f2…`), image `statelm:fla052`.
The compiled state is injected at scale 1 into all 48 GDN layers, with conv reset and session
positions starting at 0. Thinking is off, GDN state and logits are fp32, and decoding is greedy.

**Data and sample.**
- **Data.** LongMemEval_S cleaned (`xiaowu0162/longmemeval-cleaned`, revision `98d7416c`;
  `longmemeval_s_cleaned.json`, SHA-256 `d6f21ea9d60a0d56…`).
- **Frame.** Single-session-user (SSU), single-session-preference (SSP), knowledge-update (KU) and
  multi-session (MS) questions that are not abstention items, are not excluded, and whose absent
  store can be built: 278 questions.
- **Exclusions,** fixed before any sample output exists (`lme.EXCLUDE`):
  1. A LongMemEval issue (snapshot 2026-10-08) documents that the evidence does not support the
     gold answer: a3332713 (#11), dd2973ad (#21), 618f13b2 (#38), 37f165cf (#39). Issues about
     has_answer flags only, or already fixed in the cleaned file, do not exclude.
  2. The audit of absent stores. `lme.audit` automatically flags kept non-evidence sessions or
     replacements whose user turns state the gold answer (word boundary) or share at least half
     of the question's content stems. I read each flagged question against one criterion: a user
     turn answers the question in whole or in part, or, for SSP, states the preference the rubric
     names. Excluded under this rule: d851d5ba, 1c0ddc50, 75832dbd, 35a27287, b6025781. All other
     flagged questions were read and kept; the list is in LABNOTES.
- **Sample.** 24 SSU, 24 SSP, 16 KU and 16 MS, drawn per type with `random.Random(42)` and
  interleaved in a seeded random order (`lme.sample`). The ids are listed in LABNOTES and in the
  output header.
- **Smoke questions.** Two further frame questions (`09ba9854`, `6a27ffc2`) are for plumbing only
  and are never analysed. The plumbing and resume test also used `b86304ba`, an SSU question
  outside the sample. Its output was checked for schema and determinism only; the resumed rows
  were identical.

**Stores** (`code/statelm/lme.py`).
- **Layout.** One file per session, named by date and time. Repeated session ids are dropped (the
  first is kept), and the sessions are stably sorted by date.
- **Compiled memory.** The files concatenated in that order inside the memory system turn
  (~108k tokens).
- **Present:** the history.
- **Absent:** each evidence session (`answer_session_ids`) replaced in place, under the same file
  name, by a personal-style session from another history. The replacement must:
  - be within ±15% in characters (drawn uniformly among the 20 closest);
  - not already be in this history;
  - not be identical to an evidence session of this question or of its `_abs` twin;
  - share no stemmed content word of the question in its user turns;
  - not state the gold answer in its user turns (word-boundary match; purely numeric and yes/no
    golds are exempt).
- **Absent2** (SSU and SSP only): an independent draw under the same rule, excluding absent's
  picks. This is a placebo swap.
- **Shared prefill.** All versions are compiled from one prefill of the tokens before the first
  evidence file.

**Session and conditions.**
- **System turn:** the three memory tools in Qwen3.8's native format (list files, search, read a
  file); the instruction "Use the memory tools when something the user told you before could
  matter."; and today's date (the date of `question_date`).
- **User turn:** the question.
- **Conditions:**
  - A: zero state.
  - S: donor state. This is the nearest earlier question in run order whose present store holds
    none of this question's evidence; the first question takes one from the end of the order.
  - C: compiled state (present / absent / absent2).
  - T: zero state + a keyword index of the store in the system turn (top-8 TF-IDF words per file,
    with IDF from every session in the dataset; ~1.7k tokens; present / absent / absent2).
  - CT: state + index (present / absent).
  - M: the whole store in context (present / absent).

**Primary endpoint.**
- **Measure.** Per question, Δ_C = L(C, present) − L(C, absent), where L = log p(`<tool_call>`) −
  log p(not `<tool_call>`) at the first assistant token.
- **Estimand.** The mean over the 80 questions in this design mix, with a two-sided 95%
  t-interval.
- **Smallest effect of interest: δ = 0.5 logits,** a ×1.65 change in the odds of calling
  immediately. This is a go threshold for a follow-up study, not a behavioural claim: in the
  pilot even the digest's first-token effect did not change greedy calling, because calling was
  saturated.

**Decision rule.**
- **SIGNAL+** if the lower bound > 0 and the mean ≥ δ.
- **SIGNAL−** if the upper bound < 0 and the mean ≤ −δ. A cue can also act by suppressing a search.
- **NULL** if the interval lies inside (−δ, δ) and both of these hold:
  1. **No heterogeneity.** No question type has |mean| ≥ δ with a Holm-adjusted (over the 4 types)
     t-test p < 0.05. Otherwise the verdict is INCONCLUSIVE (heterogeneous).
  2. **A control registers presence:**
     - T, with lower bound > 0 and mean ≥ δ, gives **NULL**: a cheap text index carries a cue the
       state does not.
     - Otherwise M, with upper bound < 0 and mean ≤ −δ, gives **NULL (manipulation check only)**:
       the stores differ in decision-relevant content, but no hint-like cue was shown to register.
     - If neither control registers, the verdict is INCONCLUSIVE.

  A NULL whose interval excludes 0 is labelled "equivalent within δ; small non-zero effect". The
  achieved equivalence bound max(|lo|, |hi|) is always reported.
- **Otherwise INCONCLUSIVE.**
- **Robustness flags,** which never change the verdict, only flag it:
  - a percentile bootstrap interval (10,000 resamples, seed 42);
  - the greedy-path hazard Δ_path = logit(1 − Π_t p_t(not call)) over the greedy steps up to the
    stop, under C.

  If either would give a different category, the verdict is flagged "fragile" or
  "measure-dependent" and both are reported. A sign-flip p (100,000 flips, (k+1)/(B+1)) is
  printed beside the interval.
- **Labels.** If the verdict is INCONCLUSIVE and SD(Δ_C) > 1.6, it is reported as a power failure,
  and the analysis prints that label.
- **Operating characteristics** (simulated by the review; n = 80):
  - If μ = 0, P(NULL) ≈ 1.0 for SD(Δ_C) ≤ 0.8, 0.92 at 1.2, 0.59 at 1.6 and 0.21 at 2.0; NULL is
    impossible above SD 2.25. The heterogeneity guard lowers these slightly.
  - P(SIGNAL+ | μ = δ) ≈ 0.5, and ≥ 0.9 at μ = 0.65 for SD ≤ 1.
  - The pilot's single-pair SD was 0.3–0.45.
- **Consequences:**
  - NULL (either kind): the compiled state is not a usable "whether to look" cue on real
    histories at first-token resolution. The hint line stops at the decision level: no agent-loop
    study of the state as a retrieval cue.
  - SIGNAL±: supports only a topic-level claim, because the absent stores are topic-absent by
    construction. Any agent-loop study first needs a near-miss check on the `_abs` twins and its
    own pre-registration.
  - INCONCLUSIVE: reported as such; no agent-loop study under this hypothesis.

**Secondary (Holm over these 13 tests).**
1. Δ_T.
2. Δ_CT.
3. Δ_M.
4. Δ_CT − Δ_T: does the state add anything to the index?
5. Δ_C within SSP, the implicit-need type.
6. Placebo under C, SSU + SSP: mean over questions of |L(p) − L(a)| − |L(a2) − L(a)|, by sign-flip.
   It tests presence information of either sign beyond swap noise.
7. The same placebo under T.
8. Level L(C, p) − L(A).
9. Level L(S) − L(A).
10. Level L(C, p) − L(S).
11. Greedy call under C, present vs absent: exact McNemar.
12. Where the first search points. For greedy `search_memory` calls on the present store, a hit
    is an evidence file with BM25 score > 0 and pessimistic rank ≤ 3 over the store's files.
    C(p) vs A, by exact McNemar on the questions searched in both.
13. Answer-likelihood anchor (SSU and KU, gold cut at " (", ≤ 12 tokens as scored, yes/no golds
    dropped; no tools; "Answer with a short phrase only."): Δ^ans_C = Σ log p(gold | C, present) −
    Σ log p(gold | C, absent).

**Descriptive (no tests).**
- Δ_C, Δ_T and Δ_M by type.
- Greedy first action per condition and store: stop reason (call / end of turn / cap at 96
  tokens), call / (call + end of turn), preamble share, search / list / read shares.
- hit@3 per condition, and queries matching no file.
- Δ_C by tertile of tokens after the last evidence file.
- Δ_T by whether the evidence file's index line contains a gold word, a question topic word, or
  neither.
- Anchor: Δ^ans_M, C(p) − A, and the gold-verbatim-in-evidence subset.

**Sensitivity (primary estimand).**
- The frame-weighted mean.
- Without the two "last weekend" items (8550ddae, a82c026e).
- Without questions that have any audit flag.
- The ratio Δ_C / Δ_T, with a bootstrap interval.

**Running, exclusions and deviations.**
- No exclusions after registration.
- A crash is resumed with `--resume`, which skips complete questions and rewrites the crashed
  one. Results are deterministic, so this changes nothing.
- Only the full registered sample yields a verdict; a partial run is reported as a deviation.
  The analysis checks the header ids against `lme.sample`, the data hash, and the rebuilt stores
  against the run's metadata.
- The analysis script prints the verdict and any flags.
- Expected run time is about 13 h.
- Any code change after registration is logged here as a dated deviation.

**Files.** `code/statelm/lme.py`, `code/statelm/prompts.py`, `code/experiments/lme_hint.py`,
`code/analysis/lme_hint.py`. Their SHA-256 values are logged in LABNOTES at registration.

## Publication note (2026-10-15)

Appended for publication. The registered text above is unchanged.
- Where this file says hashes, item lists or question ids are "in LABNOTES", they are in
  `results/audit/LABNOTES_full.md`, the full contemporaneous log. `LABNOTES.md` is a condensed
  version.
- The registered code files are published byte-identical to their registered versions. Their
  SHA-256 prefixes match the values logged at registration:
  - G1: `code/experiments/gate_v2.py`, `code/analysis/gate_v2.py`, `code/statelm/synth_v2.py`.
  - H1: `code/statelm/lme.py`, `code/statelm/prompts.py`, `code/experiments/lme_hint.py`,
    `code/analysis/lme_hint.py`.
  The text above this note hashes to `abd50b9cc09fb2b2`, the value logged at H1 registration. The
  G1 value differs because Deviation 1 and the H1 section were appended after G1.
- "The review" in §H1 was a checklist pass run once with Claude Code agents before registration.
  It is a check on the design, not a credential.
- The G1 plumbing smoke on Qwen3.5-4B is in `results/audit/smoke/`.
