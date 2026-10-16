# Lab notebook — persistent recurrent state as compiled long-term memory

*Published as written, apart from removing references to the operator and to agent sessions, local paths, and marking dev runs 9 and 10 as never run (2026-10-15). Not included in this entry: `RESEARCH_PLAN.md` and `PLAN.md` (planning documents), run logs, smoke scripts, the plan checklist JSON and the raw Stage 1 records. Run scripts cited as `results/audit/run_*.sh` are in `code/scripts/`.*

Chronological record: what was done, what was observed, what was decided, in order.
The original proposal is `RESEARCH_PLAN.md`; the execution plan will be `PLAN.md`.
Times are `dgx-spark` local. Orchestrated with Claude Code. Paths are relative to the entry (`code/…`, `results/…`); bare script paths such as `analysis/gate_v2.py` are relative to `code/`.

---

## 2026-10-07 (Wed) — planning

- **09:45** Project start. `dgx-spark`: GB10, aarch64 / sm_121, 119 GiB unified memory,
  CUDA 13.0, driver 580.159. GPU idle; no serving container running.
- **Stack.** pip vLLM/SGLang do not work on this box. The `lmsysorg/sglang:qwen38-27b`
  image works and carries torch 2.13.0+cu130 (sees capability (12,1)), transformers
  5.12.1 (has `qwen3_5`), triton 3.7.1, flashinfer, flash-attn-4; **no
  flash-linear-attention** (HF would fall back to the slow torch GDN kernels — install
  `fla` in a derived image).
- **Model.** `Qwen/Qwen3.8-27B` (Apache-2.0, released 2026-08, `model_type: qwen3_5`,
  multimodal, thinking on by default). Text config: 64 layers = 16 × (3 GDN + 1 gated
  full attention); GDN 16 QK heads / 48 V heads × 128, conv kernel 4, recurrent state fp32
  (`mamba_ssm_dtype`); attention 24 Q / 4 KV heads × 256, partial RoPE 0.25 (mrope),
  θ = 1e7; 262k native context. Compiled GDN state = 48 layers × 48 × 128 × 128 fp32
  ≈ 151 MB per memory (≈ the attention KV of ~1.2k tokens at bf16).
- **Weights.** Small-model checks use Qwen3.5-4B (bf16); the main model is
  bf16 Qwen3.8-27B (~54 GB), downloaded for this study.
- **HF state-injection path checked by reading the code** (transformers 5.12.1):
  `Qwen3_5GatedDeltaNet.forward` continues from `cache.layers[i].recurrent_states` on
  multi-token input (`initial_state=recurrent_state if use_precomputed_states`), so a
  cache with GDN states set and attention layers empty is expressible. Two traps noted:
  (1) `_update_linear_attn_mask` drops the padding mask once the cache has state, so
  left-padded batched prefill would run pad tokens through — and decay — the injected
  state; prefill must be batch-size 1 or varlen-packed. (2) Position offset is
  irrelevant to attention (RoPE is relative and attention sees only session tokens);
  to be confirmed by a unit test, not assumed.
- **Decay prior (CPU only, `code/diagnostics/gdn_nominal_halflife.py`).** Nominal per-head
  half-life ln2/|g| with the data-dependent gate input set to 0. Qwen3.8-27B (2,304
  heads): median 25.5 tokens; 10.1% of heads > 1,000; 0.1% > 10,000; max 18.1k.
  Fraction of heads retaining > 50% of a write after T tokens: 18.6% (T = 500),
  2.3% (2,000), 0.3% (5,000). Qwen3.5-4B is shorter (5.1% of heads > 1,000).
  Output in `results/gdn_nominal_halflife_qwen38-27b.txt`.
  Interpretation (tentative): if real gates behave like this, compiled state mostly
  carries the tail of the memory file and fades over session tokens. Real gates are
  input dependent and the delta rule also overwrites, so this is a prior only; it must
  be measured on real memories before anything is concluded.
- **Decisions from the decay prior:**
  - A null C ≈ A would be ambiguous (information lost in compilation vs present but
    unreadable without the attention KV). A synthetic position sweep (target fact at
    varying distance from the end of the compiled text) becomes the program gate:
    C ≈ shuffled-state even for facts in the last ~100 tokens ⇒ the zero-training
    premise fails ⇒ stop.
  - For C, a fact's "age" is its token distance from the end of the compiled text, not
    its time gap in the history. Stratified analyses (gap, number of updates, history
    length) are confounded with file position unless position is randomised.
  - Retrieved text in D sits between the state and the question and dilutes the state;
    analyse D − B against retrieved-token count. Fix and record the tokens between
    state and question for every item.
  - Add a tail-of-memory text baseline (last N tokens in context) and a long- vs
    short-half-life head ablation.
- **Decision: the memory updater is decay-aware and owns the layout (CLM-style).** The
  model that manages the canonical memory is told how it will be compiled (recent text
  is retained best; the text itself is not visible in the state-only condition) and
  chooses the layout. To keep this from confounding everything else, the updater runs
  as two fixed steps: (1) *edit* — layout-agnostic add/revise/delete on the canonical
  entries, shared by every condition; (2) *layout* — the model orders the entries for
  compilation. C is run with the model-chosen layout (the system as proposed) and with
  a randomised layout (deconfounds the stratified analyses; their difference is the
  value of letting the model choose). Exact operation set to follow the CLM paper once
  verified.

## 2026-10-07 — Phase 0: plumbing (before first GPU minute)

- No other jobs share the dgx-spark GPU during this study.
- Research image `statelm:fla052` = `lmsysorg/sglang:qwen38-27b` + `fla-core`/
  `flash-linear-attention` 0.5.2 (`--no-deps`, keeps the base torch/triton) + rank_bm25.
  Smoke test: fla `chunk_gated_delta_rule` with an fp32 initial state runs on sm_121.
- Started bf16 `Qwen/Qwen3.8-27B` download to the local NVMe HF cache (loads from NAS are slow).
- Found while reading HF code: `LinearAttentionLayer.lazy_initialization` allocates the
  recurrent buffer in the *model* dtype, so stock HF rounds the kernel's fp32 state to bf16
  at every chunk boundary and every decode step. Harness (`code/statelm/core.py`) pre-allocates
  fp32 buffers for all conditions; T6 below measures how much this matters.
- Plan for this step: correctness checks on Qwen3.5-4B (`code/tests/check_injection.py`):
  T1 split-pass = one-pass; T2 zero-injected = fresh; T3 RoPE offset invariance; T4 chunked
  compile; T5 determinism; T6 bf16-vs-fp32 state; T7 injected state is not a silent no-op.
  Pass criteria are in the script. Nothing downstream runs until these pass.
- **Phase 0 checks on Qwen3.5-4B: all pass** (`results/audit/check_injection_q35-4b.json`).
  Memory 1,426 tokens, session 24. T1 split vs one pass: max KL 1.7e-3, argmax agreement 1.0.
  T2 zero-injected vs fresh: bitwise 0. T3 RoPE offset 0 vs 1,426: max KL 2.5e-3 (bf16 noise;
  invariance holds). T4 512-token-chunked compile: state rel. diff ≤ 4.3%, logits KL 1.9e-3 —
  use one fixed chunk size everywhere. T5 bitwise deterministic. T7 injected state is not a
  no-op (KL 1.8 vs A; argmax differs at 3 of 24 positions). fla kernels confirmed in use.
- **T6 surprise:** bf16 vs fp32 state buffers give *bitwise identical* session logits on this
  path, so the fla chunk kernel's final state is effectively bf16-exact here. Keeping fp32
  buffers anyway (free); not a fidelity issue for prefill-only scoring.
- **Measured retention, Qwen3.5-4B** (`code/diagnostics/gdn_retention.py`,
  `results/gdn_retention_q35-4b.json`). Captured g, β, k from every GDN layer and computed,
  per head, decay-only retention ∏a_j and full retention ‖k_iᵀ∏a_j(I−β_j k_j k_jᵀ)‖ for a
  write d tokens before the end. Synthetic memory, 5.9k tokens:

  | d (tokens) | decay > 0.5 | full > 0.5 | full > 0.1 | full p90 |
  |---|---|---|---|---|
  | 16 | 43% | 32% | 61% | 0.81 |
  | 256 | 36% | 12% | 39% | 0.54 |
  | 1,024 | 30% | 5% | 32% | 0.37 |
  | 4,096 | 22% | 1.4% | 21% | 0.23 |

  The nominal prior (5% of 4B heads with half-life > 1k) was *pessimistic*: data-dependent
  gating holds ~20–30% of heads open over thousands of tokens. Delta-rule overwrite, not
  decay, is the dominant loss at distance. Median head still ≈ 0 beyond ~256 tokens. Prose
  (model card, 20k tokens) is somewhat more erasing than the memory format (full > 0.1 at 4k:
  13% vs 21–23%). So: strongly recency-weighted, but a long-memory head population exists.
- **Literature sweep returned (9 agents, details to `LIT_REVIEW.md`).** The decisive finding:
  two Sept-2026 preprints already test "keep the recurrent state, drop the attention KV" on
  Qwen3.5 4B/9B/27B in a single session, reporting ~0 exact key-value recall but surviving
  language/persona, plus false recall of related lures. Being deep-read and verified before
  I rely on it. Benchmarks are real but each has problems (Memora: 10 personas, released
  full-context evaluator drops user turns, needs 3 API judges; OAKS: not about users,
  high trivial baselines; MemoryCD: rating prompt leaks the answer, ranking solvable from
  missing metadata).

## 2026-10-07 — Pilot P0: counterfactual position sweep (Qwen3.5-4B, synthetic)

Pre-registered before running. This is a pilot on the small model with synthetic memories; it
decides what the 27B study should look like, it is not a headline result.

- **Object.** Two memories identical except the target value (X vs Y). Score both values
  under each state; shift = [log p(X) − log p(Y)]_{state X} − [same]_{state Y}. Zero under
  "the state carries nothing about this fact"; generic warm-start effects cancel by design.
  Conditions A (zero state), C (compiled state), M (memory text in context).
- **Grid.** 12 attributes × 2 value pairs; memory 20 / 60 / 250 entries (~0.5k / 1.4k / 6k
  tokens); target at fraction-from-end 0, 0.1, 0.25, 0.5, 1.0; dilution 0 / 256 / 1,024
  tokens of an unrelated prior exchange (60 entries; positions 0 and 0.5 only).
- **H-P0 (readability).** For targets in the last entry, mean C shift > 0 with a 95% CI over
  the 24 pairs excluding 0. Prior work predicts failure (≈0 exact recall).
- **H-P0b (recency).** C shift falls with tokens-after-target; M stays high at all positions.
- **Decision rule.** If H-P0 fails on 4B, repeat on Qwen3.8-27B before deciding anything.
  If it fails there too, exact-value compiled memory is dead for this model family, and the
  study narrows to what prior work says the state does carry (style, preference, persona).
- **Interim look at P0 (90 of 456 rows, ~5 value pairs, ~3 attributes — not a result).**
  C shift ≈ 1–4 nats vs M ≈ 37–43; C 2AFC 0.6–0.8 vs M 1.0. Unexpected: shift is *largest*
  for the target in the **first** entry (≈3.1–3.8) and smallest in the last (≈1.2–1.8) at all
  three memory sizes — opposite to the recency prediction, although measured storage
  retention favours the last entry. Dilution (unrelated exchange before the question) cuts the
  shift ~in half by 1k tokens. Waiting for the full grid before reading anything into it.
- Copied bf16 Qwen3.8-27B (snapshot 1d4bf0f2) to local NVMe and mounted it into the container.
- **P0 result (456 rows, 12 attributes × 2 pairs, Qwen3.5-4B; `results/pilot/pos_sweep_q35-4b_summary.txt`).**
  Cluster-bootstrap 95% CIs over attributes.

  | entries (tokens) | last | 10% | 25% | 50% | **first** |
  |---|---|---|---|---|---|
  | 20 (~0.5k) | 1.16 [0.50, 1.97] | 1.26 | 1.33 | 1.31 | **2.52 [1.39, 3.87]** |
  | 60 (~1.4k) | 0.71 [0.26, 1.26] | 0.83 | 0.82 | 0.78 | **1.98 [1.05, 3.06]** |
  | 250 (~5.9k) | 0.77 [0.35, 1.26] | 0.75 | 0.72 | 0.63 | **1.77 [0.99, 2.74]** |

  M shift ≈ 35–37 nats everywhere (2AFC 1.00). C 2AFC 0.52–0.58: the shift is real but
  smaller than the prior asymmetry between paired values (mean |Δ| under A = 2.33 nats), so
  it rarely flips a greedy answer — consistent with the published 0.00 exact recall.
  Dilution (60 entries): 0 → 256 → 1,024 unrelated tokens: 0.71 → 0.34 → 0.30 (last entry),
  0.78 → 0.37 → 0.28 (middle) — halves fast, then plateaus above 0.
- **Verdict against the pre-registration.** H-P0 (readability at the tail) **confirmed**:
  the compiled state carries a graded, fact-specific signal (C/M ≈ 2–7% in nats).
  H-P0b (recency) **rejected**: no decline with distance from the end out to 5.8k tokens, and
  a *primacy* effect (first entry ≈ 2×). The storage-retention picture (strong recency) does
  not predict the readable signal. Tentative reading: what is readable after the session
  template lives in the long-retention head population, which barely decays over 6k tokens;
  short-horizon heads are overwritten by the session's own tokens before the answer. Session
  tokens, not file position, are the dilution that matters.
- **Consequence for the layout design.** The retention-aware layout prompt must describe the
  *measured* profile (first entry privileged, rest flat, session tokens dilute), not "recent
  is retained best". The memory-length effect (20 → 60 entries roughly halves the tail
  signal, then flat to 250) looks like load/interference rather than distance.
- **Decision:** proceed to P0b (dissect primacy: positions start+0…8, layer bands, head
  classes, scale) and the 27B gate. Gate positions for 27B: first / mid / last.
- **P0b result (Qwen3.5-4B, 624 rows; `results/pilot/p0b_q35-4b_summary.txt`).** Primacy is a
  gradient, not a single slot: start+0 1.98, +1 1.57, +2 1.50, +4 1.28, +8 1.11, end region
  0.74–0.84 nats. **Layer bands:** almost all of the signal is in the last third of GDN layers
  (late-only 1.88 vs full 1.98 at the first entry; early ≈ 0, mid ≈ 0.1). **Heads:** nominal
  long-half-life heads carry more (first 1.25, last 0.55) than short (0.68, 0.10); at the last
  entry the short heads carry nothing. **Scale:** ×2 nearly doubles the shift (1.98 → 3.48,
  0.74 → 1.90; 2AFC up to 0.62–0.71); ×4 adds little. The model under-reads the state; scale is
  a zero-training knob but must be tuned on dev and checked against the capability guard.
- **27B throughput (HF, bf16).** Steady-state prefill ≈ 1,070 tok/s up to 8k (1k: 0.96 s,
  8k: 7.5 s); 32k 764 tok/s; 64k 632 tok/s (t ≈ 1.04e-3·L + 8.3e-9·L² s). A 30-token session
  forward costs ~0.25 s (launch-bound). Batch-1 decode 4.5 tok/s (bandwidth bound). First call
  at a new long shape costs several extra seconds once (one-time kernel tuning). Implication:
  likelihood endpoints are cheap; free-text generation and long E prefills are the budget.
- **Red-team pass on PLAN.md** (four critic agents with distinct lenses; a written checklist,
  not a credential; findings in `results/audit/plan_redteam_2026-10-07.json`). Adopted before
  any 27B gate row is read:
  1. The counterfactual shift cannot separate user-attribute *binding* from lexical/associative
     *priming* (memory X contains token X). In P0, C raised the unmentioned value by ~1.35 nats
     over A and the mentioned one by a further 0.4–1.0. The primary gate object becomes
     binding-controlled: role-swap facts (both values in both memories, user vs colleague,
     mirrored layouts so position weights cancel) and polarity preferences (loves / can't stand
     the same thing; lexically near-identical memories).
  2. K1 as written could not fail (any 1 of 12 CIs > 0; no minimum effect; percentile
     bootstrap with 12 clusters under-covers). Re-specified below with a SESOI, t-intervals,
     a sealed test set and an explicit inconclusive branch.
  3. Gate items reused the P0 dev items and had ~13 effective observations per block. New
     sealed items (≥ 48 per block), generated and hashed before the 27B look.
  4. Bugs: `compile_text` fails on the Qwen3.8 template (it injects a "Reasoning effort is set
     to xhigh" preamble unless `enable_thinking=False`); filler entries contradict preference
     items (e.g. a dog in every "has a cat" memory); binding memories contained a third user
     value for the same attribute; K2 needed a distractor answer that the guard never scored;
     `score_multi`/`kv_only_cache` tests (T8/T9) never ran; logits were rounded to bf16 before
     log_softmax; the gate prefilled each memory three times.
  5. Plan scope (left open until after G1): spend compute on preference endpoints (PersonaMem, MemoryCD,
     PrefEval, real-memory counterfactual pairs), demote OAKS / Memora / judged LongMemEval;
     magnitudes on a bounded Δp scale, not nats; layout experiment ceiling-first (oracle-first)
     with any digest placed first.

## 2026-10-07 — Phase 1: sealed gate on Qwen3.8-27B

- Fixes from the red-team pass implemented: template (`enable_thinking=False`, preamble asserts,
  render-equivalence check); fp32 logits at scored positions; one prefill per memory serving
  C / M / K; model loaded straight to GPU; Triton autotune choices cached; sealed v2 items with
  neutral filler and date-consistent insertion; guard distractors; provenance header and
  refuse-to-append in the gate output.
- **Injection suite v2** (T0–T10) passes on Qwen3.5-4B and **Qwen3.8-27B**
  (`results/audit/check_injection_q38-27b.json`): split vs one pass max KL 0.0019; zero-state
  injection bitwise equal to fresh; K with the compiled GDN/conv copied back is bitwise equal to
  the full split pass; greedy generation matches teacher forcing; 27B state 144 MiB fp32.
  (27B, memory "dog = Biscuit" compiled, greedy C answer: "Your dog's name is **Buddy**.")
- **Long-continuation numerics (not a bug).** With a 1.4k-token memory in context and a
  ~2k-token continuation of repetitive model-card text, *every* pair of paths (one pass, split,
  token-by-token decode) disagrees at the same level: median KL ~2e-4, p99 ~2.2 nats, argmax
  agreement ~0.96 (4B). Without memory the same text agrees to p99 1e-3. The fla kernel alone
  continues exactly (rel. error 0.5% vs an fp64 recurrence, same as one call;
  `tests/check_fla_continuation.py`). Consequence: long sessions are numerically chaotic, so
  every condition uses one code path; T1L/T6L are judged on median KL + argmax agreement.
- **Pre-registered** `PREREG.md` §G1 (K1 on binding-controlled blocks, δ = 0.25 nats, α = 0.0125
  per block, t + sign-flip, explicit KILL / INCONCLUSIVE rules; K2 harm ratio; secondary list).
  SHA-256 (first 16 hex) at registration, 11:41:
  - a4c650d4fa9bb2c5  PREREG.md
  - 8e0bd41cd2e4377b  code/analysis/gate_v2.py
  - 1b2cf4a29eb8d4fa  code/experiments/gate_v2.py
  - f4edb084a1a46f97  code/statelm/synth_v2.py
- Plumbing smoke test ran the first item of each sealed block on Qwen3.5-4B
  (`results/audit/smoke/`). That is the only look at sealed items before the 27B run.
- **12:00** 27B gate running (`results/phase1/gate_v2_q38-27b.jsonl`), ~46 s per fact item.
  Bookkeeping error found and logged as PREREG Deviation 1: the registered items hash was from
  the 1-item smoke subset; full-set hash `69a22aa5586baf47`; `synth_v2.py` SHA-256 unchanged.
- **13:34 gate finished** (760 rows, 6,410 s). Registered file hashes re-checked before running the
  analysis: `analysis/gate_v2.py` 8e0bd41cd2e4377b, `experiments/gate_v2.py` 1b2cf4a29eb8d4fa,
  `synth_v2.py` f4edb084a1a46f97 — unchanged. (`PREREG.md` hash differs only by the appended
  Deviation 1.) Analysis run unmodified:
  `results/phase1/gate_v2_q38-27b_analysis.txt`.

### Result — G1 (pre-registered)

| Block | mean shift (nats) | 97.5% CI | one-sided LB | t p | sign-flip p | items > 0 | verdict |
|---|---|---|---|---|---|---|---|
| B1 facts_bind | **0.387** | [0.243, 0.532] | 0.243 | 6.4e-8 | < 1e-5 | 48/48 | **PASS** |
| B2 prefs_fact | **0.185** | [0.112, 0.258] | 0.112 | 2.1e-7 | < 1e-5 | 39/48 | not PASS (θ̂ < δ = 0.25), not KILL (UB > δ) |

**K1 = PASS via B1 only.** The pre-registered consequence is "fact endpoints only (LongMemEval
log-prob anchor)"; the preference benchmarks (PersonaMem, MemoryCD, PrefEval) do **not** proceed
as confirmatory work, and the plan's "minimum positive result" (PASS on B2) was not met. B2 is
clearly non-zero but its whole interval sits around the SESOI — the preference effect is real
and smaller than I said would matter.

**K2 = PASS.** Harm ratio H = −0.048 (90% CI −0.076 to −0.022): compiled memories slightly
*widen* the gold-vs-distractor margin on 40 unrelated questions (mean A margin 7.2 nats), and
no gold answer flipped to the distractor in 480 question × memory cells.

Secondary (registered; Holm over the six p-values applied by hand because the script printed
raw p — `results/phase1/gate_v2_q38-27b_explore.txt`):

- facts_raw (single mention, priming + binding) C 0.995 nats; raw − bind 0.623 (Holm p 5e-9).
  About 60% of a single-mention shift is lexical/positional priming, 40% binding.
- Position: facts_raw first − last 0.519 (Holm p 3e-4) — primacy replicates at 27B on sealed
  items. Memory size 60 vs 250 entries barely matters (facts 0.032, Holm p 0.18; prefs 0.036,
  Holm p 0.014). prefs_fact first − last 0.031 (n.s.).
- Instruction form beats fact form: prefs_instr − prefs_fact(n=60) 0.125 (Holm p 0.002);
  instr C = 0.328 overall (first 0.41, last 0.24). Consistent with prior split-prefill work in
  which instructions survived and facts did not. Secondary; cannot rescue K1.
- Bounded magnitudes: 2AFC C 0.60 (raw), 0.54 (bind), 0.53 (prefs fact), 0.52 (prefs instr);
  M 1.00 / 1.00 / 0.63 / 0.69.
- K (memory KV kept, GDN zeroed): facts_raw 19.3 nats vs M 30.0 and C 1.0. The attention KV,
  not the recurrent state, carries nearly all of in-context recall.
- Summed-token sensitivity agrees with the first-divergent primary for facts (1.01 vs 0.995;
  0.39 vs 0.39); for prefs it is larger (0.34 vs 0.19) because later option tokens also shift.
  Sensitivity only.
- Lures (H6, descriptive; 8 DRM lists): under C the false-memory pull (assoc − neutral, Yes−No
  log-odds) is **0.265** [0.19, 0.34], 8/8 lists, while discrimination (present − assoc) is only
  **0.053** [0.03, 0.08]. Under M it is the other way round (2.9 vs 11.7). The compiled state
  carries the *gist* of a list far more than which word was in it.

### Exploratory (written after unblinding; `analysis/gate_v2_explore.py`; nothing here touches K1)

- **Binding is not a position artefact.** Decomposing the mirrored pairs: binding 0.421
  [0.28, 0.56] (positions 0 vs 30) and 0.354 [0.24, 0.46] (59 vs 30) — the two estimates
  agree. The position-weighted priming term is +0.60 for slot 0 over slot 30 and −0.18 for slot
  59 over slot 30: primacy again, and a mild preference for the middle over the tail.
- **The compiled state delivers a roughly constant ~5–8% of the in-context effect.** Ratio
  Δp_C / Δp_M with item-bootstrap 95%: facts_bind 0.050 [0.034, 0.066], prefs_fact 0.061
  [0.033, 0.097], prefs_instr 0.081 [0.039, 0.136], facts_raw (incl. priming) 0.133. (Prior
  recurrent-only long-context QA estimate: 5.1% — a coincidence of scale worth noting, not a
  replication.)
- **B2 missed δ partly because the preference probe is weak, not because preferences are harder
  to compile.** Even with the memory text in context, 14/48 preference items move < 1 nat; item
  C and M shifts correlate r = 0.80. On the 34 items where the text works (M > 1 nat), C = 0.258
  [0.18, 0.34]; on the other 14, C ≈ 0. Relative to its own ceiling the preference signal is the
  same size as the fact signal. So the plan's central hypothesis H ("preferences transfer while
  facts do not") is not supported in either direction: bound facts and preferences both transfer
  at ~5% of text.

**What I take from G1.** The compiled GDN state carries user-bound information across a session
boundary at 27B — every one of 48 binding items is positive, the effect survives a design where
the token multisets are identical, and nothing is harmed. But the effect is small in any unit a
user would notice: 2AFC 0.54, ~5% of the bounded effect of having the same text in context, and
it biases toward gist (lures) rather than item memory. That is a mechanism finding, not a
memory system.

Next (per the decision rule and PLAN §7): 27B dev diagnostics + retention profile are running
(H-readout, dilution, scale, bands/heads on 27B). The only benchmark the rule allows is the
LongMemEval fact anchor; I will not start it until the scope question is settled,
since the pre-registered "minimum positive result" failed and the practical magnitude argues
for a mechanism write-up.

## 2026-10-07 (afternoon) — 27B mechanism diagnostics, and a change of question

- **27B dev diagnostics** (dev items only: 12 v1 attributes, raw single-mention facts = priming +
  binding; `results/phase1/dev_diag_q38-27b_summary.txt`; 13:34–14:11). C shift, nats, t 95% over
  12 attributes:
  - Position: start+0 1.66, +1 1.39, +4 1.17, mid 0.90, end 0.89 (M ≈ 35 everywhere). Primacy
    replicates; mid ≈ end. A neutral entry at index 0 does not remove the head bonus for index 1
    (1.43), and neither does a ~200-token preamble (first 1.55, last 0.88) or shuffled dates
    (1.87 / 0.70) — the bonus belongs to "early in the memory", not to slot 0 or to date order.
  - Dilution between state and question (first / last): structure-only exchange 1.52 / 0.84
    (no loss); 512 tokens of prose in a separate exchange 0.87 / 0.41; 2048 tokens 0.67 / 0.31;
    512 in the same turn 1.46 / 0.57.
  - **Scale:** ×0.5 0.80 / 0.32; **×2 4.25 / 5.46**; ×3 4.85 / 5.95 (start / end). Doubling the
    injected state multiplies the shift 2.6× at the head and **6×** at the tail, and the tail
    overtakes the head — the ordering that storage retention predicts. Raw facts, so this
    includes priming; binding under gain and harm under gain are untested.
  - Bands (only one third of GDN layers kept): late 1.55 / 0.69, mid 0.01, early 0.01. Heads:
    long half-life only 1.01 / 0.65, short only 0.64 / 0.16.
  - **Conv buffer kept** (3 tokens per GDN layer, reset by design): 3.86 / 5.45 — larger than the
    whole recurrent state. The buffer holds projections of the last memory tokens' hidden states,
    which were computed with attention over the full memory, so it is an attention-mediated
    channel the model reads natively. The closest prior split-prefill work carried it; part of
    its positive instruction-following result may be this channel.
  - prefs_dev (v1 dev preferences, fact form): first 1.05, last 0.48; scale ×2 3.06 / 3.34;
    late band only 0.71 / 0.23; M 7.9 / 5.6.
- **27B storage retention** (`results/gdn_retention_q38-27b.json`; fraction of GDN heads keeping
  > 0.5 / > 0.1 of a write's contribution, full delta rule): 250-entry memory (5.9k tokens), d = 1024:
  6% / 28%; d = 4096: 0.7% / 17.5%. Decay alone would keep 27% / 36% at d = 1024 — overwriting by
  the delta rule, not decay, is the main loss, as on 4B.
- **Reading of the two together.** Storage favours the tail; readout favours the head; amplifying
  the state by 2× recovers the tail more than the head and multiplies the effect several-fold;
  memory size (60 vs 250 entries) barely mattered in G1. My working interpretation: at these
  memory sizes the weak link is *readout gain*, not retention capacity. Not yet tested: whether
  a decoder can recover user-bound content from the state (storage), and whether gain helps
  binding without harm.

- **Change of question (after G1).** The practical memory I care about is agentic RAG over files
  (list / grep / read), so the interesting comparison is *access* vs *access + hint* (compiled
  state), with "read everything into context" as the confound that benchmarks must guard against
  (stores larger than context, matched read budget, and the same content as a short text digest
  as the real competitor). Richer state stores (triadic / tensor / multi-timescale) are a possible
  answer to retention, but only after the current mechanism is bottomed out. This is a new
  question motivated by G1: it gets its own pre-registration and sealed items; G1 stands.
- Lit sweep for hint vs access (`LIT_REVIEW.md` §6; 36 verified entries added): internal-signal
  retrieval gating is established within a session; familiarity-without-recall suppresses "I
  don't know"; text-gist + access (ReadAgent) gains mostly from the gist sitting beside the
  fetched page, not from gist-guided choice. No prior work uses a cross-session compiled state as
  the cue. Baselines: another user's state (generic confidence shift), text digest, a probe on
  the hint-free model, always-retrieve.

- **Next (dev only, 27B):** (1) gain sweep on dev binding items with a harm guard and lures;
  (2) storage probe — linear decoders on the compiled states for raw / binding / binding at 250
  entries, against the model's own readout on the same states; then (3) the likelihood-only
  retrieval-decision pilot.

## 2026-10-07 (late afternoon) — gain sweep and storage probe (27B, dev items, exploratory)

Setup was logged in the "Next" item above; both ran back to back on dgx-spark in the container,
gain sweep 14:29–15:09, storage probe 15:09–16:35. Dev items only (12 v1
attributes, 12 v1 dev preferences, the G1 lure lists and guard questions, new seeds); nothing
sealed was touched. All of this is exploratory and nothing here bears on K1/K2.

**Gain sweep** (`code/experiments/gain_sweep.py`, `results/phase1/gain_sweep_q38-27b_summary.txt`;
C shift in nats, t 95% over 12 attributes; 60-entry memories):

| gain | bind (4 layouts) | bind 2AFC | raw first / last | raw 2AFC | prefs first / last | guard H [90%] | lure assoc−neutral | present−assoc |
|---|---|---|---|---|---|---|---|---|
| ×1 | 0.80 [0.25, 1.35] | 0.583 | 2.38 / 1.65 | 0.625 | 1.07 / 0.49 | −0.044 [−0.07, −0.02] | 0.24 | 0.06 |
| ×1.5 | 1.88 | 0.604 | 4.80 / 6.03 | 0.812 | 1.91 / 0.77 | −0.123 | 0.94 | 0.13 |
| ×2 | 2.58 [1.20, 3.96] | 0.667 | 5.99 / 9.01 | 0.812 | 3.15 / 2.85 | −0.260 | 1.40 | −0.01 |
| ×3 | 2.85 | 0.667 | 7.06 / 9.51 | 0.896 | 3.35 / 4.00 | −0.174 | 0.95 | 0.00 |
| late ×2 | 1.29 | 0.562 | 3.71 / 2.82 | 0.667 | 1.77 / 0.67 | −0.028 | 0.25 | 0.07 |
| late ×3 | 1.60 | 0.594 | 4.54 / 3.57 | 0.750 | 2.30 / 0.86 | −0.014 | 0.26 | 0.06 |
| late ×2, rest 0 | 1.31 | 0.583 | 3.74 / 2.51 | 0.708 | 1.46 / 0.37 | +0.026 [0.02, 0.04] | 0.01 | 0.01 |

- Binding/priming split from the mirrored layouts (pair 0 vs 30 | pair 59 vs 30): ×1 B 0.87 / 0.73,
  P 0.90 / −0.41; ×2 B 2.21 / 2.95, P 2.06 / 1.78. Uniform gain roughly triples the binding
  component, but it also switches on a recency pull for the last entry (P for 59 vs 30 goes from
  −0.41 to +1.78), so the extra shift is not all binding.
- Harm: no gold→distractor flips at any gain, and H is negative (margins on unrelated questions
  *grow*) except for "late ×2, rest 0" (+0.026). The guard measures two-way margins only;
  generation quality under gain is untested.
- Lures: gain multiplies the pull toward associated-but-absent words (0.24 → 1.40 at ×2) while
  present-vs-lure discrimination stays at ~0. Gain amplifies familiarity, not item memory, for
  word lists. Late-only injection removes both the pull and the discrimination (0.01 / 0.01).
- Dev g1 binding here (0.80) is about twice the sealed G1 estimate (0.39); different items and
  layouts' seeds, and dev attributes have larger shifts in every earlier run.

**Why 2AFC stays near 0.6 although the shift is reliable.** From the storage-probe rows (same
attributes): the injected state moves d consistently (within-class SD 0.2–0.3 nats; paired
accuracy 0.88–0.94) but by 0.8–0.95 nats on binding, while the model's prior gap between the two
candidate values is ~2 nats (median |d_A| 2.2). The answer flips only for attributes whose prior
is near even. With a calibrated threshold the model's own readout classifies a single state at
0.83–0.86. So the hint is a consistent but small push, which is what a retrieval *cue* needs and
what an answer-changing *memory* does not get.

**Storage probe** (`code/experiments/storage_probe.py`, `code/analysis/storage_probe.py`,
`results/phase1/storage_probe_q38-27b_summary.txt`): 12 attributes × 16 counterfactual pairs per
task (pair = same filler and slots, user's value swapped), difference-of-means probes computed
from per-head Grams of the fp32 states, leave-one-pair-out; single-state accuracy / d′:

| task (≈ tokens) | all heads | late third | nested top-16 heads | nested top-64 | model ×1 | model ×2 |
|---|---|---|---|---|---|---|
| raw60 (1.5k) | 0.997 / 6.9 | 0.977 / 6.3 | 1.000 / 13.0 | 1.000 / 20.4 | 0.951 / 7.7 | 0.927 / 8.2 |
| bind60 (1.5k) | 0.747 / 1.6 | 0.773 / 1.8 | 0.987 / 5.6 | 0.987 / 7.0 | 0.836 / 2.8 | 0.831 / 2.9 |
| bind250 (6.3k) | 0.807 / 2.3 | 0.784 / 2.1 | 0.995 / 7.0 | 0.997 / 6.7 | 0.831 / 2.8 | 0.862 / 3.1 |

- The all-heads probe did *not* beat the model, which I first read as "no hidden store". The
  per-layer curve said otherwise (single GDN layers reached d′ 3.2–3.5), so I added — post hoc,
  after reading that curve — a nested selection: inside each outer fold, rank heads by the SNR of
  inner leave-one-pair-out margins on the 15 training pairs only, keep the top k. Summing all
  2,304 heads was diluting a small set of informative ones.
- Checks (`code/analysis/storage_heads.py`, `results/phase1/storage_heads_q38-27b.txt`):
  - Label-swap null for the nested probe (50 within-pair swaps): mean 0.50, max 0.65 paired and
    0.58 single, against 0.98–1.00 observed.
  - The heads are shared across attributes: top-64 overlap between attributes 17–20 heads
    (chance 1.8). Heads chosen on the other 11 attributes, then a plain probe on the 12th: bind60
    0.97–0.98 single, bind250 0.98–0.99; heads chosen on bind60 transfer to bind250 at 0.99. With
    only 8 pairs and fixed heads, 0.95–0.97.
  - They sit in model layers 24–62, mostly 28–42 and 48–60 — not only the late third (where the
    G1-era band test found all the *readable* signal; mid-band injection alone gave ≈ 0).
  - No fall-off with distance inside a 250-entry memory: held-out z-margin median 1.55 / 1.79 /
    1.53 for the farther fact 0–83 / 83–166 / 166–250 entries from the end, all correct
    (spearman with the colleague fact's distance −0.34).
- Interpretation: up to ~6k tokens, who-holds-what is stored almost perfectly in a few dozen
  shared heads, and the model's native readout uses it at d′ ≈ 2.8 against the decoder's 6–7.
  At this scale the bottleneck is readout, not storage; richer state (triadic / tensor stores)
  is not motivated by these sizes. The retention estimate (0.7% of heads keep > 0.5 at d = 4096)
  was too pessimistic as a capacity bound, since a few heads suffice.
- Caveat: the decoder is trained on 15 labelled pairs per attribute and the model is not, so
  "the model under-reads" means "a linear readout the model does not use exists", not that the
  model could be trivially fixed. A head-targeted gain on the shared heads is the direct test.

**Next.** (1) Capacity stress (`code/experiments/capacity_probe.py`): binding facts followed by 30
/ 250 / 1000 / 4000 neutral entries (~1.8k to ~100k tokens), 12 attributes × 8 pairs, fixed head
sets from today's run plus the model's readout, to find where storage itself breaks. Timing (one
compile each): 1.8k tokens 3 s, 7k 6 s, 26k 30 s,
101k 193 s (peak 62 GiB), so d4000 runs on every other attribute (6 × 8 pairs, ~5 h) after the
other strata (~2.3 h). (2) Head-targeted gain on the shared heads, chosen leave-attributes-out. (3) The
retrieval-decision pilot.
Started 16:58 (approx.); log `results/audit/logs/capacity_probe_q38-27b.log`.

## 2026-10-08 (night) — capacity stress: no loss out to ~100k tokens of neutral filler

Capacity run finished 00:33 (`results/phase1/capacity_probe_q38-27b_summary.txt`; analysis
`code/analysis/capacity_probe.py`). Binding facts (user → X, colleague → Y, swapped within a pair)
at entries 30–39, then 30 / 250 / 1000 / 4000 neutral entries; 8 pairs per attribute; d4000 on
every other attribute (6). Probe heads were chosen on yesterday's storage-probe run (bind60 +
bind250, the other 11 attributes only), so selection is independent of these memories.
Mean over attributes, single-state accuracy / d′ (model: plus 2AFC at threshold 0 and shift):

| after facts (≈ tokens) | fixed 16 heads | fixed 64 | all heads | model ×1 | model ×1 2AFC / shift | model ×2 2AFC / shift |
|---|---|---|---|---|---|---|
| 30 (1.8k) | 0.995 / 7.2 | 1.000 / 8.5 | 0.79 / 2.2 | 0.87 / 3.6 | 0.589 / 0.84 | 0.641 / 2.25 |
| 250 (7.3k) | 0.964 / 6.1 | 0.969 / 7.0 | 0.78 / 2.6 | 0.92 / 4.5 | 0.604 / 0.97 | 0.620 / 2.18 |
| 1000 (26k) | 0.979 / 7.3 | 0.995 / 8.4 | 0.78 / 2.5 | 0.93 / 4.8 | 0.594 / 0.94 | 0.589 / 1.69 |
| 4000 (101k), 6 attrs | 0.990 / 6.3 | 0.990 / 7.1 | 0.79 / 1.7 | 0.99 / 5.7 | 0.573 / 0.72 | 0.562 / 1.20 |

On the same 6-attribute subset at every distance the picture is the same (fixed-16 single 0.98–1.00;
model shift 0.60 / 0.67 / 0.76 / 0.72 nats). Label-swap null for fixed-16: mean 0.50–0.52, max
0.69–0.81 (max is highest at d4000, n = 6). No attribute shows a trend with distance.

- Neither the stored signal nor the model's own readout falls with distance, out to ~100k tokens
  written after the facts. Whatever limits the hint at these scales, it is not loss with distance.
- This contradicts a naive reading of the retention profile (0.7% of heads keep > 0.5 of a write at
  d = 4096). That profile measures the whole write; the probe needs only a discriminative
  component in a few heads, and templated neutral filler may write into key directions that
  barely overlap the facts' keys. Under the delta rule, what should erase a binding is later
  writes with *similar* keys — other facts of the same form. That is the realistic stress, since
  a user memory is mostly other facts.
- Caveats: facts sit near the start (entries 30–39), where G1-era diagnostics found readout
  primacy, so "far from the end" is also "near the start"; the storage-probe run (random
  positions in 250 entries) showed no position effect either.

**Interference stress (setup; started 00:34).** `code/experiments/interference_probe.py`:
same fact block at entries 30–39, then 128 entries of which K are interfering facts at random
slots and the rest neutral filler (distance fixed, ~4.2k tokens). Kinds: *same* (the target
attribute held by other people: "The user's neighbour's dog is named Mochi.") and *other* (other
attributes, held by the user or by other people). Interferer values from pool[2:], never X or Y;
everything but the user/colleague swap shared within a pair. Cells: none (K = 0), same / other ×
K = 8, 32, 128; 12 attributes × 8 pairs; seeds 97000 + 1000·attr + 100·cell + pair; ~2 h.
Analysis as for capacity (fixed heads from the storage-probe run, nested, all heads, model readout).
Prediction: *same* interference degrades the stored binding with K and the model's readout with
it; *other* much less. If *same* at K = 128 leaves the fixed-head probe near ceiling, storage is
not the limit for user-fact memories of this density either.

**Head-targeted gain (setup; queued to start when the interference run ends).**
`code/experiments/head_gain.py`, heads from `code/analysis/binding_heads.py`
(`results/phase1/binding_heads_q38-27b.json`): heads ranked by storage-probe SNR (bind60 +
bind250); for bind / raw items, ranked on the other 11 attributes only; for prefs / lures / guard,
the global ranking. Same items and seeds as the gain sweep, so rows compare directly and
`analysis/gain_sweep.py` reads the file unchanged. Configs: g1, uniform ×2, top-16 ×4 / ×8, top-64
×2 / ×4, and layer-matched random heads (16 ×8, 64 ×4) as controls. Question: does amplifying the
heads that *store* the binding raise binding (B, 2AFC) without the recency pull and lure
familiarity that uniform gain brought? If the random controls do as well, the effect is just
"more state". If the top heads do nothing, the model does not read those heads (consistent with
mid-band injection alone giving ≈ 0).

**Retrieval-decision pilot (setup; queued after head gain).** First dev look at my new
question (hint vs access); exploratory and likelihood-only, to calibrate a later pre-registered
study. `code/experiments/retrieval_pilot.py`. The session's system turn declares three memory
tools (list files / search / read file; one file per past session) in Qwen3.8's native tool
format, and I score only the first action: log p(`<tool_call>`) at the first assistant token,
function choice given a call, and the query given a search call (forced choice over 12 topic
keywords, scored as full token sequences up to the closing newline). Two families of 12 items:
*P* — implicit need (dev preference fact "The user is vegetarian." / "Suggest one dinner idea
for tonight.", which never mentions memory); *F* — explicit questions about dev attributes ("What
is my dog's name?"). Per seed the 12 items split into two complementary 60-entry stores with the
same filler and slots, so each question is scored with its fact present and absent against
stores of equal size (diet and spice always share a store). 8 seeds per family. Conditions: A
(zero state), Aw (zero state + a prompt telling the model its memory was read into its state —
the CLM-style awareness, as a prompt-only control), C, Cw (compiled state, neutral / aware
prompt), T (topic digest of the store in the prompt — the cheap text competitor), CT, M (whole
store in context). What I will look at first: (1) present − absent in log-odds of calling a tool,
per condition — does the hint carry "something relevant is in memory", and how does that compare
with a 6-word digest? (2) C − A overall — does familiarity raise or suppress searching?
(3) P(correct query) present vs absent. No thresholds set; this sizes the effects for a prereg.

## 2026-10-08 (early morning) — interference result

**Interference stress (result; 00:34–02:37).** `results/phase1/interference_probe_q38-27b_summary.txt`
(analysis `code/analysis/capacity_probe.py`, unchanged). All cells ~4.2k tokens; 12 attributes ×
8 pairs. Single-state accuracy (leave-pair-out), mean over attributes:

| cell | fixed-16 probe | all heads | model g1 single | model g1 2AFC | model shift (nats) |
|---|---|---|---|---|---|
| none (K = 0) | 0.990 | 0.786 | 0.901 | 0.615 | 0.96 |
| same, K = 8 | 0.958 | 0.729 | 0.880 | 0.589 | 0.85 |
| same, K = 32 | 0.990 | 0.734 | 0.859 | 0.557 | 0.70 |
| same, K = 128 | 0.995 | 0.802 | 0.849 | 0.583 | 0.74 |
| other, K = 8 | 0.943 | 0.672 | 0.818 | 0.589 | 0.70 |
| other, K = 32 | 0.932 | 0.688 | 0.839 | 0.573 | 0.61 |
| other, K = 128 | 0.974 | 0.630 | 0.792 | 0.573 | 0.54 |

Label-swap null (fixed-16): mean 0.50–0.52, max 0.65–0.72. Model shift relative to K = 0, paired by
attribute (t 95% over 12 attributes; the mean is dominated by city and hometown, which carry
2–3.5 nats, so I also give the median ratio): same-8 −0.10 [−0.27, +0.07] (×1.03), same-32 −0.25
[−0.59, +0.08] (×0.81), same-128 −0.22 [−0.58, +0.14] (×0.86); other-8 −0.25 [−0.52, +0.02]
(×0.79), other-32 −0.34 [−0.72, +0.03] (×0.74), other-128 −0.41 [−0.78, −0.05] (×0.63). Same − other
at matched K: +0.15 [+0.03, +0.27], +0.09 [−0.00, +0.18], +0.20 [+0.04, +0.35].

- My prediction was wrong. *Same*-attribute facts about other people do not erase the stored
  binding: the fixed-16 probe is at ceiling with 128 of them (0.995). The model's readout loses
  perhaps 15–20% of its shift, with CIs that include zero.
- *Other* interferers do more harm than *same* ones, to both the probe (a few points, and
  non-monotonic in K: 0.94 / 0.93 / 0.97, so mostly noise around ~0.95) and the readout (shift
  ×0.63 at K = 128, CI excludes zero). The all-heads probe also drops most here (0.63).
- Half of the *other* interferers are facts about the user ("The user's car is a Volvo.") and half
  about other people. A reading that fits: the write key is shaped more by the subject ("The user
  …") than by the attribute, so further user facts compete with the user binding, while a
  neighbour's dog does not. The run cannot separate subject from attribute, because *other*
  mixes both. That matters, since a real user memory is mostly user facts. So I am running the
  split (below).
- Per attribute, the readout is noisy, and some attributes are near the floor at every K (drink,
  dog, sister, allergy: shift ≤ 0.3). allergy goes to 0.00 at same-128 ("The user's neighbour is
  allergic to kiwi" ×128) while its probe stays at 1.00: a readout failure, with storage intact.
- Overall: up to 128 interfering facts plus ~100k tokens of filler, storage is not the
  bottleneck. The model's own readout is, and interference erodes that readout gradually rather
  than abruptly.

**Interference by subject (setup; queued after the retrieval pilot).** Same script, new cells via
`--cells` (seed base 120000; output `results/phase1/interference_subject_q38-27b.jsonl`, Grams in
`data/interference/`): *user* = other attributes held by the user (this includes conflicting
values for the same other attribute, e.g. two coffee orders), *person* = other attributes held by
other people; K = 32 and 128 each; ~65 min. Prediction from the subject reading: *user* ≥ *other* >
*person* ≈ *same* in harm to probe and readout. If *person* hurts as much as *user*, the subject
reading is wrong, and what matters is the attribute's key mismatch with the probe's heads.

**Head-targeted gain (result; 02:37–03:21).** `results/phase1/head_gain_q38-27b_summary.txt`
(analysis `code/analysis/gain_sweep.py`, unchanged; same items as the gain sweep, and g1 / g2
reproduce it exactly). Paired over attributes, t 95%:

| contrast | bind shift | raw shift | prefs shift |
|---|---|---|---|
| top-16 ×8 − g1 | +0.29 [+0.08, +0.49] (×1.25) | +0.90 [+0.35, +1.44] (×1.59) | +0.47 [+0.06, +0.88] (×1.46) |
| top-64 ×4 − g1 | +0.53 [+0.22, +0.85] (×1.66) | +1.44 [+0.79, +2.10] (×1.75) | +0.67 [+0.17, +1.18] (×1.76) |
| top-64 ×4 − random-64 ×4 | +0.51 [+0.19, +0.83] | +1.37 [+0.71, +2.02] | +0.67 [+0.17, +1.16] |
| random-64 ×4 − g1 | +0.02 [−0.00, +0.05] | +0.08 [+0.02, +0.13] | +0.01 [−0.01, +0.03] |

(× = median per-attribute ratio.) Lures assoc − neutral: g1 0.24, top-64 ×4 0.34, random-64 ×4
0.31, uniform ×2 1.40. Raw first / last: g1 2.38 / 1.65, top-64 ×4 3.80 / 3.12 (primacy kept;
uniform ×2 flips to recency, 5.99 / 9.01). Guard H −0.04 to −0.05 for every head config; no flips.
Bind 2AFC: top-64 ×4 0.573 vs g1 0.583 (1 item gained, 2 lost of 96); uniform ×2 0.667 (11 gained,
3 lost).

- The heads the probe picked out are ones the model reads. Amplifying them raises the hint's push
  1.3–1.8×, and amplifying layer-matched random heads by the same factor does nothing. The heads
  were ranked on other attributes, and on fact items rather than preference items, yet the gain
  carries over to the preference block (heads ranked globally on ATTRS). So the result is not
  item-specific.
- It is cleaner than uniform gain: no added lure familiarity beyond the random-head control, no
  recency flip, no guard cost.
- It is not more *accurate*. Binding and priming rise together (pair 0 vs 30: B 0.87 → 1.38,
  P 0.90 → 1.34), and bind 2AFC does not move. So the extra push lands as much on "a dog name was
  mentioned" as on "the *user's* dog is X", and against a ~2-nat prior gap it rarely flips an
  answer.
- Taken together with the storage results: the binding is stored separably in a few dozen heads,
  and the model does read those heads. But the read path turns them into a value-level
  familiarity push, not a bound answer. Gain on the stored side cannot fix a readout that does not
  use the binding. For the hint-vs-access question this firms up the working model: **the
  compiled state is a reliable, fairly clean familiarity signal (which values and topics are in
  memory), not a source of answers.**

## 2026-10-08 (morning) — retrieval-decision pilot result

**Retrieval pilot (result; 03:21–06:05).** `results/phase1/retrieval_pilot_q38-27b_summary.txt`
(analysis `code/analysis/retrieval_pilot.py`; 1,968 rows as designed). Log-odds of
`<tool_call>` as the first assistant token. pres − abs is paired within seed (same filler and
slots, complementary stores), averaged over seeds per question, t 95% over 12 questions; AUROC is
within question (8 present vs 8 absent states).

| | P (implicit need) pres − abs | AUROC | level vs A | F (explicit) pres − abs | AUROC | level vs A |
|---|---|---|---|---|---|---|
| C (compiled state) | +0.04 [−0.04, +0.12] | 0.54 | −1.76 [−2.74, −0.78] | +0.06 [−0.04, +0.16] | 0.58 | −1.18 [−1.50, −0.86] |
| Cw (state + aware prompt) | +0.04 [−0.01, +0.09] | 0.55 | −0.68 vs Aw | +0.00 [−0.06, +0.06] | 0.51 | −0.51 vs Aw |
| T (6-word topic digest) | +0.87 [+0.17, +1.57] | 0.73 | −0.97 | +1.57 [+1.13, +2.01] | 0.97 | −0.52 |
| CT (state + digest) | +1.43 [+0.47, +2.39] | 0.77 | −1.37 | +1.44 [+1.13, +1.74] | 0.97 | −1.80 |
| M (whole store in context) | +0.66 [−0.24, +1.55] | 0.63 | −4.36 | −2.62 [−3.37, −1.87] | 0.02 | −5.15 |

Zero-state p(call): P 0.40, F 0.89. The aware prompt alone lowers calling (Aw − A: −0.60 P,
−0.96 F). Function choice given a call: search 0.64–0.97, the same present and absent under C;
under T, more search when present. Query choice (forced over 12 keywords): under C, P(correct
keyword) is the same present and absent (P 0.82 / 0.81; F at ceiling, since the question names
it); under T it drops when absent (0.47), because the digest pulls the query toward listed topics.

- **The compiled state does not tell the agent whether the needed fact is in memory.** Present
  vs absent makes no difference to calling, to which function, or to the query, for implicit or
  explicit questions, with or without the aware prompt (AUROC 0.51–0.58). A 6-word text digest
  carries that signal strongly (AUROC 0.73 implicit, 0.97 explicit), and adding the state to the
  digest adds nothing beyond it.
- The state lowers the first-token call rate across the board (−1.2 to −1.8 logits). That fits the
  storage and gain results: the state gives value-level familiarity, not a "this is in memory"
  signal, and familiarity here means "answer now". But see the caveat.
- **Caveat on the first-token measure.** The top first token under C is often "I" or "Let" (P:
  72 and 48 of 192 states), and "Let me check your memory …" may well lead into a call. So the
  level shift may partly be a shift from calling straight away to calling after a preamble.
  pres − abs is less exposed, because both arms share the same preamble tendency, but it is still
  a first-token measure. M rows are not comparable as a read-all ceiling for the *decision*: with
  the store in context, not calling is the right move (F M pres − abs −2.6: the model calls more
  when the fact is *not* in the context it can see — appropriate).

**Greedy first-action follow-up (setup; queued after the subject cells).**
`code/experiments/retrieval_gen.py`: same items, stores and seeds. Greedy decode up to 32 tokens
before a `<tool_call>` (stop at end of turn), then up to 28 tokens of the call, so the actual
function and free-text query are recorded. Adds C0 (compiled state of the seed's filler with no
facts) as a control for "any memory-like state" vs this store's content. Questions: (1) does
pres − abs under C stay at zero on the binary call decision? (2) is the C − A level shift real or a
preamble artefact, and does C0 shift as much as C (generic warm-state) or less (content)? (3) do
free-text queries under C leak the stored value (e.g. search "vegetarian" when the store says
vegetarian), which would be the value-level familiarity showing up in the query?

**Interference by subject (result; 06:06–07:17).** `results/phase1/interference_subject_q38-27b_summary.txt`.
Same layout and analysis. Single-state accuracy, mean over 12 attributes (none, same and other
are from the first run, same layout, different seeds):

| cell | fixed-16 probe | model g1 single | model shift | shift vs none, paired (median ratio) |
|---|---|---|---|---|
| user, K = 32 | 0.964 | 0.776 | 0.53 | −0.42 [−0.87, +0.02] (×0.69) |
| person, K = 32 | 0.953 | 0.807 | 0.70 | −0.26 [−0.52, +0.01] (×0.78) |
| user, K = 128 | 0.953 | 0.786 | 0.50 | −0.45 [−0.94, +0.03] (×0.75) |
| person, K = 128 | 0.969 | 0.828 | 0.62 | −0.33 [−0.70, +0.03] (×0.77) |

user − person: −0.17 [−0.35, +0.02] (K = 32), −0.12 [−0.32, +0.08] (K = 128); user − same: −0.17
[−0.31, −0.03], −0.24 [−0.45, −0.02]. Nulls as before (mean ~0.51, max ≤ 0.72).

- My prediction was half right. Facts about the user hurt the readout slightly more than the
  same facts about other people, but the difference is small and its CIs include zero.
  *person* ≈ *same* failed: other-attribute facts about other people hurt both probe (0.95–0.97
  vs 0.99) and readout (×0.77) more than same-attribute facts about other people (0.99, ×0.86).
  What costs most is **facts of other attributes**, whoever holds them; being about the user
  adds a little.
- Across all ten interference cells, storage stays ≥ 0.93 (fixed-16 probe), and the model's
  readout loses 15–35% of its push, with no cliff from K = 8 to K = 128. A plausible reading is
  dilution of a familiarity-like readout by other fact-shaped writes (a fan effect), rather than
  overwriting of the binding. I have not tested this further.
- Phase-1 storage question (exploratory, dev): within these synthetic bounds (≤ 100k tokens of
  filler, ≤ 128 interfering facts), **storage is not the bottleneck; readout is.** Triadic or
  larger-state stores would address a storage limit we have not found at this scale.

**Greedy first-action follow-up (result; 07:17–10:59).** `results/phase1/retrieval_gen_q38-27b_summary.txt`
(analysis `code/analysis/retrieval_gen.py`; 2,160 sessions as designed). Binary per session:
does a `<tool_call>` open within 32 greedy tokens.

| | P call share pres / abs | P pres − abs (t 95% over q) | F call share pres / abs | F pres − abs |
|---|---|---|---|---|
| A (zero state) | 0.83 | — | 1.00 | — |
| C0 (filler-only state) | 0.73 | — | 1.00 | — |
| C | 0.72 / 0.73 | −0.01 [−0.03, +0.01] | 1.00 / 1.00 | 0.00 |
| Cw | 0.75 / 0.75 | 0.00 | 0.99 / 0.97 | +0.02 [−0.03, +0.07] |
| T (digest) | 0.83 / 0.81 | +0.02 [−0.01, +0.05] | 1.00 / 1.00 | 0.00 |
| CT | 0.83 / 0.75 | +0.08 [−0.04, +0.21] | 1.00 / 0.97 | +0.03 [−0.04, +0.10] |
| M (store in context) | 0.14 / 0.15 | −0.01 | 0.00 / 0.69 | −0.69 [−0.88, −0.49] |

Level shifts in call share: C − A −0.11 [−0.28, +0.06], C − C0 −0.01 [−0.09, +0.08] (P); zero in F.
Preamble before the call: A 20% of calls, C0 81%, C 81%, T 59% (P); ~0 in F. Function given a
call (P): search 0.70 in A, 0.39 in C0, 0.35 in C, 0.43 in Cw, 0.82 in T; the rest is almost all
`list_memory`. No generated preamble or query contains the stored value under any state
condition (C, Cw, CT with the fact present: 0 of 528 sessions that have a checkable value word); under M, 50% (P) and 100% (F) do.

- **The first-token level shift was mostly a preamble artefact.** Under a compiled state the model
  says "Let me check if you've mentioned any food preferences before." and *then* calls, about
  as often as with no state (C − A −0.11, CI includes 0). The same happens with a filler-only
  state (C ≈ C0), so it is a generic warm-state effect, not this store's content.
- **No present-vs-absent signal on the actual decision either**, under C or Cw. The finding
  "the state does not tell the agent whether to look" stands on the binary decision.
- **The binary decision is saturated in this pilot.** Greedily, the agent searches for almost
  every question that might need memory (P 10 of 12 questions always; F 12 of 12), whatever the
  condition. So even the digest's strong first-token signal (AUROC 0.73 / 0.97) does not change
  what the agent *does*. The digest only reshapes the query (search share 0.82, topic in query
  0.63). The "read all the time" confound is here the agent's *default policy*, not a design
  choice. A benchmark for the hint's value as a "when to look" signal needs a realistic turn mix
  in which most turns need no memory, so that unnecessary reads can be counted.
- **New, unplanned observation:** any compiled state (C0, C, Cw) moves the agent from targeted
  search toward listing all files (`list_memory` ~0.6 of calls vs ~0.3 with no state, 0.2 with the
  digest), i.e. toward browsing. That is the opposite of the efficiency the hint was meant to buy.
  Exploratory, 12 dev questions, A has one session per question; treat as a lead.
- The state never leaks the stored value into what the agent says or searches for. That fits the
  probe and gain results: the stored value is there, but the model's readout does not surface it
  as content, here or in answer likelihoods.

**Report draft (late morning).** Drafted `REPORT.md` and `README.md` covering phase 1 (G1, the
storage-vs-readout work, the hint pilot), with two figures from `code/analysis/figures.py`
(`images/storage_vs_readout.png`, `images/hint_pilot.png`). Brought `code/README.md` up to date
with run commands for every phase-1 script. While cross-checking numbers against the summaries
I corrected two counts in these notes (value-leak denominator 528, not 1,152; the greedy agent
calls for 9–10 of 12 implicit-need questions depending on condition, not 10 in all).
The next phase is not chosen yet. The options are a real-benchmark check of the hint (first-action
scoring on a LongMemEval_S subset with its abstention items as natural absent cases), a trained
readout, or stopping at phase 1.

## 2026-10-08 (midday) — hint check on LongMemEval_S: setup

Decision: run the real-benchmark check of the hint, first-action scoring only, before any
agent-loop study. Data: LongMemEval_S cleaned (`data/raw/longmemeval/`, source and revision in
`SOURCE.txt`). Design, in short (full text goes into PREREG §H1 at registration):
- Each history becomes a store of ~50 files (one per session, named by date and time), ~108k
  tokens compiled. *Absent* store = evidence sessions replaced in place by length-matched personal
  sessions from other histories that contain neither the gold answer nor any of the question's
  content words. Present and absent share the prefill up to the first evidence file, which saves
  compile time; for the first sampled question the evidence is the last file (shared prefix
  104,822 of 107,566 tokens).
- Conditions A, S (donor state), C, T (keyword index: top-8 TF-IDF words per file, ~1.8k tokens),
  CT, M (whole store in context, ~108k tokens). Primary: within-question present − absent
  first-token call log-odds under C, δ = 0.5, with an equivalence-style NULL branch.
- A CPU check on 78 frame questions: BM25 with the question itself as the query puts an evidence
  file in the top 3 for 76. Finding the evidence is easy in LongMemEval_S once the agent searches.
  So "where to look" will be near ceiling, and the informative part is "whether to look".
- The keyword index puts a question content word on an evidence file's line for ~60% of questions
  (top 8 words; 50% with top 5). It is a realistic competitor, not an oracle.
- Sample: 24 SSU, 24 SSP, 16 KU, 16 MS from a frame of 282 (5 MS questions have no admissible
  replacement), `random.Random(42)`. Smoke test on two frame questions outside the sample
  (`0977f2af`, `6456829e`) for timing and plumbing only; started 11:36.
- Before sealing, a red-team pass with four lenses (validity, statistics, code, benchmark), run as
  a multi-agent workflow. It is a written checklist executed once, not a credential.

**Smoke test (11:36–11:57; plumbing only).** Two frame questions outside the sample, about 6–7.5 min
per question: compile 270–380 s with the shared prefix, then about 70 s of sessions. Peak about
83 GB reserved. All conditions and the deferred donor row were written. Explicit questions sit near
p(call) ≈ 0.95–0.99 in every condition. Not analysed.

**Red-team pass on the H1 draft (four agent lenses, a written checklist executed once; findings summarised here).**
A crash and several design problems, all fixed before registration:
- **Crash at question 65 of 80, about 8 GPU-hours in.** `prompts._check` rejected memory text
  containing "think carefully" (ordinary assistant text in one history; 7 of 282 frame questions).
  The check now looks only at what the template generated, so outputs are unchanged for every
  earlier experiment. There is now a CPU `--preflight` that builds every question's inputs, and a
  `--resume`, because a crash previously meant a full rerun.
- **Store construction.**
  - The gold-answer filter was a substring match, so "2" or "3" matched almost every session
    (dates, prices). That dropped 5 MS questions from the frame and gave numeric questions
    digit-poor replacements. It is now a word-boundary match on user turns, with numeric and
    yes/no golds exempt.
  - Content words are now stemmed ("course" vs "courses" let a MasterClass session through).
  - Candidates identical to any evidence body (including the `_abs` twin's) are banned.
  - KU and TR histories are not date-sorted in the cleaned file, so sessions are now sorted by
    date, and repeated session ids are dropped.
  - "Today's date" is now given as the date only, since some KU sessions are timestamped after the
    question time on the same day.
  - The keyword index uses a fixed IDF over the whole dataset, so present and absent indexes
    differ only on evidence lines.
- **Measure.** The first-token logit cannot see a decision made after a preamble; the pilot's
  compiled-state sessions mostly called after one. The primary stays first-token, which is
  continuous and unsaturated. Each greedy step now records log p(call), and a greedy-path call
  hazard is pre-registered as a robustness flag. The pre-preamble cap rises from 32 to 96 tokens,
  with a stop reason (call / end of turn / cap), because at 32 nearly every "no call" in the pilot
  was the cap. The logit is now log p(call) − log p(not call) in double precision (no cap at 30).
- **Decision rule.**
  - The validity control is now directional and applies only to NULL. T registering gives a full
    NULL; only M registering gives "NULL (manipulation check only)".
  - A heterogeneity guard across the four types was added: a mixed-sign effect could otherwise
    average to NULL.
  - Operating characteristics are stated.
  - A placebo second absent store (independent replacement draw) was added for SSU and SSP, to
    calibrate swap noise and catch presence information of either sign.
  - The Holm family is named (13 tests).
  - BM25 ties at score zero no longer count as hits.
  - The donor is the nearest earlier question whose store holds none of this question's evidence.
- **Benchmark issues.** Excluded where a LongMemEval issue documents that the evidence does not
  support the gold: a3332713 (#11), dd2973ad (#21), 618f13b2 (#38), 37f165cf (#39). The two "last
  weekend" items get a sensitivity analysis. Yes/no golds are dropped from the anchor.

**Outcome-blind audit of the absent stores** (`lme.audit` flags, then my reading against "a user
turn answers the question in whole or in part, or states the SSP rubric's preference"; no model
output exists for these questions):
- **Excluded:**
  - d851d5ba: a kept session states charity money raised (found by the review).
  - 1c0ddc50: a kept session states listening to podcasts on the bus commute.
  - 75832dbd: a kept ShareGPT session states wanting to publish in AI conferences.
  - 35a27287: a kept session states taking language exchange classes.
  - b6025781: a kept session asks for healthy recipes for meal planning.
- **Read and kept** (topic-word or incidental matches that do not answer the question): e3038f8c,
  195a1a1b, a08a253f, 118b2229, 8550ddae, 07741c44, edced276, a82c026e, 54026fce, 6cb6f249,
  51a45a95, 80ec1f4f, fca70973, eace081b, a89d7624, 184da446, 86f00804, 0edc2aef, e66b632c,
  6b7dfb22.
- **Resulting frame:** 278 questions. Sample (run order):
  58ef2f1c, 9d25d4e0, 184da446, 0edc2aef, 00ca467f, d7c942c3, caf03d32, 311778f1, fca70973, 46a3abf7
  4d6b87c8, 3b6f954b, 118b2229, 8550ddae, 29f2956b, 726462e0, 6b7dfb22, 06878be2, 4f54b7c9, 54026fce
  c5e8278d, 4fd1909e, faba32e5, b01defab, d23cf73b, 195a1a1b, 505af2f5, affe2881, 603deb26, 8a2466db
  60472f9c, ad7109d1, a82c026e, e66b632c, 06f04340, 1a1907b4, 853b0a1d, 60d45044, 0ddfec37, 06db6396
  07b6f563, 95228167, 1da05512, 1a8a66a6, 09d032c9, 51a45a95, 6ade9755, 3d86fd0a, 10d9b85a, 078150f1
  07741c44, b0479f84, 55241a1f, e3038f8c, 830ce83f, 27016adc, 1d4e3b97, 8fb83627, 0a34ad58, eace081b
  86b68151, gpt4_a56e767c, 86f00804, 0f05491a, 32260d93, 94f70d80, d24813b1, 36580ce8, 38146c39, 75f70248
  2318644b, e61a7584, 945e3d21, 25e5aa4f, 57f827a0, 8cf4d046, 577d4d32, 2b8f3739, ccb36322, afdc33df
  Smoke / plumbing questions: 09ba9854, 6a27ffc2; plumbing test also uses b86304ba (SSU, outside the sample).

**Second checklist pass** (two agent lenses, executed once: revised code, and analysis vs pre-registration). It found
only minor issues, all fixed:
- Greedy-path values were rounded to 5 decimals; they are now full precision, and the path hazard
  is computed from log p(call), so it is exact when p(call) is tiny.
- The resume check covered only the question ids. It now compares the full header, flushes the
  header at once, and truncates a half-written last line.
- The fragile / measure-dependent flags compared against the unguarded category; they now compare
  final verdicts from one `decide()` function.
- The registered "power failure" label is now printed.
- Partial and smoke files print no verdict. The analysis asserts the registered sample, the data
  hash, and that the rebuilt stores match the run's metadata.

The plumbing test (b86304ba, 09ba9854) plus a resume from a truncated copy reproduced all 29 row
keys with identical values, so the run is deterministic. Per-question time is 7–9 min; the second
review's estimate from compile throughput is about 13 h for the 80 questions. The CPU preflight
passed again on the final code.

**H1 registered 13:10.** `PREREG.md` §H1. SHA-256 (first 16 hex):
- abd50b9cc09fb2b2  PREREG.md
- 85c4d92df8f63556  code/statelm/lme.py
- 9257ddc45bcaaab7  code/statelm/prompts.py
- cdb101d38567e911  code/experiments/lme_hint.py
- 74bea721696c2c1a  code/analysis/lme_hint.py

Run started right after: `results/phase2/lme_hint_q38-27b.jsonl`, log
`results/audit/logs/lme_hint_q38-27b.log`.

## 2026-10-09 (early morning) — hint check on LongMemEval_S: result

**Run.** The run went from 13:10 to 01:43, 44,670 s (12.4 h) for 80 questions. There was no crash
and no resume. Per question it took 5–14 min, and peak memory was 84 GiB allocated (94 GiB reserved). Before
analysis I re-checked the five registered SHA-256 values and all matched. I ran the registered
analysis unmodified; its output is `results/phase2/lme_hint_q38-27b_summary.txt`.

### Result — H1 (pre-registered)

**VERDICT: INCONCLUSIVE (no control registered presence).** This is not a power failure:
SD(Δ_C) = 0.21, below the pilot's 0.3–0.45.

| | mean | 95% CI | notes |
|---|---|---|---|
| **Δ_C (primary)** | **+0.030** | [−0.017, +0.077] | bootstrap [−0.017, +0.076]; sign-flip p 0.21; 43/80 > 0; achieved equivalence bound 0.077 |
| Δ_T (control) | +0.314 | [+0.154, +0.473] | lower bound > 0 but mean < δ, so it does not register |
| Δ_M (control) | −0.367 | [−0.588, −0.147] | upper bound < 0 but mean > −δ, so it does not register |
| Δ_path under C | +0.081 | [−0.238, +0.399] | robustness flag; the same category |

- **Heterogeneity guard:** no type comes near δ. By type: SSU +0.04, SSP +0.01, KU +0.06, MS +0.01;
  all Holm p ≥ 0.65.
- **Sensitivities:**
  - frame-weighted +0.030 [−0.018, +0.078];
  - without the "last weekend" items +0.032;
  - without the 15 audit-flagged questions +0.033 [−0.011, +0.077];
  - Δ_C / Δ_T = 0.10, bootstrap [−0.07, +0.25].
- **Registered consequence:** "INCONCLUSIVE: reported as such; no agent-loop study under this
  hypothesis." In practice this is the same as a NULL: the hint line stops at the decision level.

**Secondary (Holm over 13).**
- Detectable after Holm:
  - Δ_T +0.31 (Holm p 0.002);
  - Δ_M −0.37 (Holm p 0.015);
  - placebo T +0.40 (Holm p 1e-4): the index registers presence beyond swap noise.
- Not detectable:
  - Δ_CT +0.20 (Holm p 0.10);
  - Δ_CT − Δ_T −0.11 (Holm p 0.57): the state adds nothing to the index;
  - Δ_C in SSP +0.01;
  - placebo C +0.02 [−0.03, +0.08]: no presence information beyond swapping in another unrelated
    session;
  - levels C(p) − A +0.24, S − A +0.29, C(p) − S −0.05 (all Holm p ≥ 0.15).
- **Greedy behaviour under C:** present vs absent, 0 discordant pairs out of 80. hit@3 C(p) vs A:
  0 vs 1 discordant.
- **Answer anchor** Δ^ans_C (n = 38): +0.015 nats [−0.063, +0.093]. Descriptive comparisons:
  - Δ^ans_M is +10.7 [8.9, 12.5];
  - C(p) − A is −0.82 [−1.35, −0.28];
  - with the gold verbatim in the evidence (n = 35), Δ^ans_C is +0.045 [−0.012, +0.102].

  The compiled state of a real ~108k-token history carries no answer-content trace the model
  can read. It moves the gold likelihood, but by the same amount whether or not the evidence is
  in the store.

**Descriptive.**
- **Greedy first action.** The agent calls a tool in 95–99% of sessions under A, S and C, so the
  decision is saturated, as in the pilot. Under M with the present store, it answers instead in
  14 sessions (gold string in 13 of them) and writes past the 96-token cap in 30.
- **Where the first search points.** hit@3 is 0.94–1.00 in every condition (A 0.98, 58
  searches). The question itself is the retrieval key: one search lands on the evidence file
  without any hint.
- **Δ_C by distance from the end:** no trend. Near +0.08, middle −0.01, far +0.02.
- **Δ_T by evidence index line:** +0.48 when the line names a question topic word, +0.09 when it
  names neither, and +0.37 when it names the gold (n = 6).

**What I got wrong in the design.** I required the controls to clear δ on the *same*
first-token measure as the primary. The pilot had already shown that calling is saturated and
that cues act after a preamble. The index's main effect turns out to be *which* tool (read
instead of search), not *whether* to call, and M acts after a preamble. Both are invisible to
first-token p(any call). A behavioural or path-level validity check would have been the better
gate. The registered verdict stands.

### Exploratory (written after unblinding; `analysis/lme_hint_explore.py`; nothing here touches H1)

Output: `results/phase2/lme_hint_q38-27b_explore.txt`.
- **The controls clear δ on the greedy-path hazard; the state does not.** Present − absent,
  path hazard:

  | condition | path hazard | 95% CI | first token |
  |---|---|---|---|
  | T | +0.78 | [+0.16, +1.40] | +0.31 |
  | CT | +1.00 | [+0.30, +1.69] | +0.20 |
  | M | −1.87 | [−3.15, −0.60] | −0.37 |
  | C | +0.08 | [−0.24, +0.40] | |

  So the controls fell short of the registered rule because of the measure, not because the
  stores did not differ.
- **The index changes what the agent does; the state does not.** Under C the first action
  (stop reason, function and argument, the search query included) is identical for present and
  absent in 80 of 80 questions. The count is 49 under T, 43 under CT and 30 under M. Paired
  present vs absent, exact McNemar:
  - T: reads 12 vs 1 (p 0.003), and all 12 present-store reads open the evidence file;
  - CT: reads 21 vs 1 (p 1e-5), 20 opening the evidence file;
  - M: calls 6 vs 24 (p 0.001); with the evidence in context it answers instead;
  - C: 0 vs 0 for both calls and reads.
- **Correlation with the controls.** Δ_C correlates weakly with Δ_T across questions (Spearman
  +0.28, p 0.01), but not with Δ_M (+0.16). Whatever shared swap effect this is, it is about a
  tenth of the index's size.

**Where this leaves the hint question.** On real histories at 108k tokens, the compiled state
gives the agent no cue that the evidence is present. This holds at the first token (bounded
within ±0.08 logits), along the greedy path, and in what the agent does (0 discordant pairs).
It also carries no readable trace of the answer. A ~1.7k-token keyword index gives a detectable
cue and sends the agent straight to the evidence file. LongMemEval_S also makes "where to look"
trivial: the agent searches by default, and its first query ranks the evidence file in the top 3
for 94–100% of questions. So the benchmark does not stress the regime where a hint could help
(vocabulary mismatch, implicit need, a read budget).

## 2026-10-10 — change of question: seeding the attention layers (planning)

**Why.** The untrained model does not read the GDN state as a "look here" cue, neither in the
60-entry pilot (where a probe decodes the content at about 0.99) nor at 108k tokens (H1). So the
limit is readout, not scale. The working hypothesis for the next step is that the action decision
(call a memory tool or not) is written by the 16 global attention layers, which read the whole
prompt, while the GDN state's readable push arrives late and refines the output. Phase-1 data
consistent with this (none designed to test it):
- the late third of GDN layers alone gives nearly all of the readable effect (1.55 vs 1.66 nats),
  and the early or mid third alone gives about 0.01;
- K (memory KV only) gives 19.3 nats vs M 30.0 and C 1.0;
- the 3-token conv buffer gives more than the whole recurrent state.

**What I am after.** A scalable way to get continual-learning behaviour without new mechanisms:
derive something from one compile pass over the memory and inject it into the unchanged model.
Constraints:
- (a) derived from the whole memory, not a subset of raw KV entries (that is partial context);
- (b) fixed size, or clearly sublinear in memory length;
- (c) consumed by existing mechanisms: extra KV entries with an optional per-entry additive logit
  bias, plus the GDN state;
- (d) built before the question is known;
- (e) nothing fitted on test questions.

Ruled out: injecting the full or partial raw KV (no better than putting the memory in context),
and training a reader on present/absent pairs (training the test). Text indexes stay a baseline
only. Byte parity with the GDN state is not a requirement.

**Literature and feasibility** (a multi-agent workflow: two literature sweeps, one code read, three
designs each with an adversarial critic; LIT_REVIEW §8):
- **Prior work.** The object exists for plain transformers as KV compaction by Attention Matching
  [zweiger2026fast]: compact keys, a fitted per-entry log-mass bias, and least-squares values, all
  from the context's own queries; on Gemma-3 only the global layers were compacted. Centroid
  merging with +log n is ResKV [zhan2026reskv]; its diffuse centroids dilute sharp peaks.
- **New here.** No one has combined such a seed with an injected recurrent state, tested it on
  cross-session memory with the question hidden, scored it on a presence or tool-decision
  measure, or run it at a fixed size near 100k tokens.
- **Cache facts** (checked in the image's transformers source and a CPU toy):
  - Keys are cached post-k_norm and post-RoPE, in bf16.
  - Only head dims 0–63 rotate, with pairs (j, j+32) at 1e7^(−j/32). Pairs j ≥ 25 turn less than
    0.37 rad over 108k tokens, so 206 of the 256 dims are effectively position-free.
  - The causal mask is built from cache indices and its size is read from attention layer 3, so
    all 16 attention layers must hold the same number of entries.
  - Positions act only through RoPE.
  - A per-entry or per-layer bias needs a registered SDPA wrapper.
- **Corrections to my own framing:**
  - A log(count) bias under-weights merged clusters (Jensen), so the bias must be fitted.
  - The conv-buffer result is a recency channel (the last tokens attended over everything), not
    evidence for centroids. Its analogue is a "gist suffix".
  - Collapsing all memory keys to one position keeps the seed valid as memory grows but throws
    away "how long ago". That cost can be measured exactly without compression.
- **Critics' main point.** K has never been measured at about 100k tokens or in the agent setup;
  the 19.3 nats came from G1 memories of at most ~6k tokens. If the full attention KV does not
  carry what M carries there, no compression of it will.

**Stage 0 (exploratory, dev; no compression): where does the memory's effect live?**
- **Sample ("dev16").** Drawn with `random.Random(42)` from the H1 frame after removing the 80 H1
  questions and the plumbing ids (09ba9854, 6a27ffc2, b86304ba). Run in a seeded order. Stores
  are built as in H1 (present / absent / absent2 where buildable). The plan was 4 per type, but
  only 2 SSP questions remain outside the H1 sample, so the draw is 5 SSU, 2 SSP, 5 KU and 4 MS
  (found by the CPU preflight, before any GPU run).
- **Conditions.**
  - Every store version: A (no memory); C (GDN state, conv reset); K (memory KV, GDN and conv
    zeroed); M (full continuation).
  - Present and absent only:
    - CK (KV plus GDN state, conv reset);
    - K*4k and K*64k (KV with every key un-rotated to position 0; the session starts 4,096 or
      65,536 positions later);
    - Krt (un-rotate, then re-rotate to the original positions: the numerical floor for K*);
    - K_early / K_mid / K_late (memory KV visible only at attention layers 3–19 / 23–43 / 47–63,
      masked elsewhere through the bias wrapper; GDN zeroed).
- **Measures.**
  - H1's session measures: first-token call log-odds, greedy path with stop reason, call and its
    argument, and the answer anchor for short-gold SSU/KU questions.
  - Plus a direct logit attribution: each of the 64 mixers' and 64 MLPs' contribution to
    z_call − E_q[z] (q = next-token distribution renormalised over non-call tokens) at every
    greedy step.
  - Plus the per-layer contributions at the last prompt position (fp16), saved per question.
- **What I will read.** With n = 16 these are descriptive.
  1. **Agreement with M across all sessions:** the slope of L(X) − L(A) on L(M) − L(A), and
     first-action agreement with M, for X in {C, K, CK, K*, Krt, K bands}.
  2. **Presence effects (present − absent)** on L, the path hazard, behaviour and the anchor,
     against the absent2 swap floor.
  3. **Where the call logit is written** (attention vs GDN vs MLP, by depth) in A and M, and
     where M's present−absent difference is written.
- **Branch rules** (rough, dev-only, written before the run):
  - **K vs M.** If K reproduces about half or more of M's shift (slope ≥ 0.5) and agrees with M's
    first action clearly more than C does, the attention route has a ceiling worth compressing
    towards (Stage 1). If K ≈ C ≈ A (slope < 0.2), the attention KV alone does not carry it at
    108k, and I stop to rethink before building any compaction.
  - **K\* vs K.** If K\* keeps about 80% of K's slope, Stage 1 seeds use collapsed positions;
    otherwise they keep original phases (logical length T).
  - **Bands.** The band that alone reproduces most of K tells Stage 1 where a seed has to go.
- **Plumbing first** (`tests/check_kvseed.py`, real model, a ~6k-token G1 dev memory):
  - the wrapper with no bias is bitwise equal to stock SDPA;
  - zero bias equals K;
  - a fully masked memory approximates A;
  - Krt approximates K (sets the K* noise floor);
  - a position shift on A approximates A;
  - the DLA terms sum to the logit contrast;
  - bias and shift survive a deepcopy.

**Plumbing (`tests/check_kvseed.py`, real model, 250-entry G1 dev memory, 5,885 tokens):** PASS.
- **S1 (wrapper, no bias).** Bitwise equal to stock SDPA for A, C, K and M.
- **S2 (zero bias on every layer).** Equals K exactly.
- **S3 (memory masked at every layer).** Approximates A: max KL 4.5e-4; the only difference is RoPE
  rounding at shifted absolute positions.
- **S9 (masked at half the layers).** Differs from both K (max KL 4.9) and A (0.057), so the bias
  reaches the layers.
- **S5 (un-rotate then re-rotate).** 95.5% of key elements are bitwise equal; max relative error
  0.19%.
- **S4 (Krt vs K).** Max KL 7.0e-4; this is the numerical floor for K*.
- **S6 (A shifted by 65,536).** Max KL 4.3e-4.
- **S7 (deepcopy).** Keeps the shift and bias bitwise.
- **S8 (generate on the biased, shifted cache).** Did not run: greedy emitted a stop token at once.
  Stage 0 does not use `generate()`.
- **Regression.** `tests/check_injection.py` reruns with the new `core.py` and passes; every T value
  is identical to the earlier run.

**Code review before the run** (an agent checklist executed once: three lenses on mechanics, measures and robustness; one
skeptic per finding; 7 of 11 findings confirmed). None changes the bf16 plumbing results. Fixed:
1. `unrotate_keys` wrote in place when the keys were already fp32 (`.float()` aliases). bf16 runs
   were not affected; it now copies.
2. The attribution kept autograd graphs alive: the cached fp32 head and the norm weight carried
   `requires_grad`. That is about 5.7 MB of GPU memory per greedy step, held until the end of the
   question, up to about 13 GB, which is a real out-of-memory risk. Gradients are now off in
   `lme_where`, and the head is detached in `core.py` (values unchanged).
3. Fully masked layers (the depth bands) went through a float mask, which expands K/V to 24 heads
   on every decode step. Hidden entries are now sliced off and stock SDPA runs on the rest
   (identical maths; S2/S3/S9 rerun).
4. The analysis compared K (pooled with absent2) against K* and the bands (present/absent only) on
   different sessions. All conditions are now compared on present/absent, and the pooled C and K
   rows are extra lines.
5. The attribution differences used each condition's own readout direction and norm scale, so a
   change in readout looked like a change in what components write. Each session now saves its
   step-0 readout vector, and differences are decomposed under A's readout, with drift reported
   separately. Per-component outputs are saved as bf16 (lossless), not fp16.
6. S8 generated from a cache the session had already been run through. It now uses an unmutated
   snapshot and records whether it ran.

Rejected after checking: Krt is used as the K* floor; the accumulation residue of the bf16
residual is reported by the check; fp16 overflow is moot after item 5; peak memory has headroom.
The review's runtime estimate is about 21 min per question, so about 6 h for dev16.

## 2026-10-10 (evening) — Stage 0 result: where the memory's effect lives at ~108k tokens

**Run.** dev16, 16 of 16 questions, 15,311 s (4.3 h), peak allocation 77.9–83.6 GiB per question.
Memories are 104.8k–111.4k tokens (mean 108.9k). Greedy decoding and deterministic forward passes,
so one seed; dev16 is a single `random.Random(42)` draw. Outputs:
`results/phase2/lme_where_q38-27b.jsonl` (traces in `.d/`), the planned summary
`lme_where_q38-27b_summary.txt` (`code/analysis/lme_where.py`) and the post-hoc analyses
`lme_where_q38-27b_posthoc.txt` (`code/analysis/lme_where_posthoc.py`, not in the plan). The plan
above was written before the run; the only evidence is file times, since the project is not a git
repository (LABNOTES saved 15:21, smoke run 15:47, data complete 20:09).

**Plumbing record, corrected.** The plumbing paragraph above predates the final check run
(`results/audit/check_kvseed_q38-27b.json`, 15:27, after the code-review fixes). In that run S8 ran
and passed: `generate()` on a cache with the K*4k shift and half the layers masked matched teacher
forcing. S3's max KL is 2.6e-4 (not 4.5e-4) and S9's are 4.84 / 0.054. The one-question smoke run
(b86304ba, 760 s, peak 84.2 GiB; `results/phase2/lme_where_plumbing.jsonl`) exercised the pipeline
and the analysis. The planned item "the DLA terms sum to the logit contrast" was checked inside the
main run, not in the plumbing script: |sum − x·w| ≤ 0.034 and |x·w − target| median 0.002.

**Results** (descriptive; n = 16, anchor n = 9 single-session-user and knowledge-update questions;
95% intervals; many comparisons, none adjusted).
1. **Most of the memory's effect on the call decision is generic.** Loading the history lowers L in
   all 32 present and absent sessions (M − A mean −2.69). About 85% of that drop is there whether or
   not the answer is in the store (absent − A −2.47, absent2 − A −2.48, present − A −2.92). I ran no
   long non-memory text, so "this memory" and "any 108k-token prefix" cannot be told apart.
2. **The generic shift: attention KV vs GDN state.**
   - K reproduces its average size (through-0 slope 1.00 [0.84, 1.20]; a constant shift of K's mean
     size would already give 0.93), but tracks it per session only weakly (r 0.46). Its
     call-vs-stop first move matches M's no better than A does (20/32 vs 21/32).
   - C moves L the other way (C − A +0.51) and calls in all 32 sessions.
   - CK is closest to M (r 0.79, mean |L − M| 0.64, call-vs-stop 25/32).
3. **Presence (present − absent) on the decision.** M: L −0.45 [−0.68, −0.22], path hazard H −3.97
   [−6.46, −1.49].
   - K moves between stores by as much as M beyond the absent2 swap floor (+0.38 vs +0.38; C +0.02),
     but not in M's direction: per question, L r −0.45 (signs agree 7/16; chance 9.2) and H r 0.06.
   - CK has M's average direction (L −0.37 [−0.69, −0.05]; H −4.09) and tracks M per question on H
     (slope 0.82 [0.61, 1.15], r 0.71, signs 15/16; chance 13.3), but not on L (r 0.11; signs
     14/16; chance 11.1). CK has no absent2 floor (absent2 ran only C, K and M).
   - C shows nothing.
4. **Answer content (anchor, present − absent).**
   - CK ≈ M: +10.5 vs +11.1 nats; per question slope 0.94 [0.85, 1.03], r 0.95 (part of r is the
     shared difficulty scale).
   - K: +3.8 (34% of M; slope 0.38). It never brings the gold near certainty: log p(gold) > −0.5 in
     0/9 present sessions, against CK 4/9 and M 5/9.
   - C: −0.18. C also scores about 2 nats below A for both stores, so a 108k GDN state with no
     matching KV does harm.
   - The anchor has no absent2 floor (anchors were not run on absent2).
5. **Position collapse.**
   - K*64k keeps 102% of K's through-0 slope and K*4k 79% (68% on H).
   - On content K*4k loses nothing: K*4k − K is +0.58 [+0.14, +1.03] nats. K*64k and Krt were not
     scored on the anchor.
   - With positions collapsed, the first action barely responds to presence. (stop, function)
     differs between present and absent in 1/16 questions (K*4k) and 2/16 (K*64k), against K 7,
     Krt 7 and M 5.
6. **Depth bands.** Each band is visible alone with GDN zeroed, so these are off-distribution
   sufficiency tests, not a map of where M reads.
   - Kmid (attention 23–43) gives most of K's generic shift (slope 0.97, r 0.57), but almost no
     content (+0.31 nats, 8% of K) and no presence direction (neither has K).
   - Early and late each give about 15% of the generic shift (slopes 0.16 and 0.15, intervals
     exclude 0) and at most 0.32 nats of content.
   - The three bands' content sums to 0.66 of K's 3.78 nats, so content is not additive across
     depth. No pair of bands was run.
7. **Direct logit attribution** of z_call − E_q[z] at the first assistant token. This target is L
   plus the entropy of the non-call distribution, not L; for M's presence effect it is −0.64 vs
   −0.45, r 0.72.
   - In A, +2.9 is written, +2.5 of it by layers 48–63: attention 59 alone +3.0 (positive in every
     question), attention 51 +0.9, GDN 52–63 +1.8, MLPs −2.1.
   - M's presence effect (−0.64 ± 0.25): under every readout I tried (present, absent, midpoint,
     A's), GDN layers write −0.75 to −1.00 (GDN 62 alone −0.63) and attention −0.30 to −0.59
     (mostly layers 51 and 63). The MLP term swings from −0.90 to +2.28 with the readout, so I read
     nothing into it.
   - The same late GDN layers write the generic shift in K, M and Kmid (Kmid: −2.8 of −3.2 over
     GDN 52–63).

**Retraction: the planned attribution contrast.** Section 4 of the summary (every difference under
A's readout, the plan after code-review fix 5) gave M present − absent = +0.79. That has the
opposite sign to the own-readout change (−0.64) and flips in 12/16 questions; the flip is all in the
MLP column. After seeing it I added section 4b (post hoc), an exact split through the midpoint of
the two sessions' own readouts. It is one of several exact splits, and only its GDN and attention
columns hold under other readouts (post-hoc P4). I retract section 4's MLP and total rows for
presence contrasts.

**The working hypothesis.** Before the run I wrote that the 16 attention layers write the decision
and the GDN state refines it. On direct writes that does not fit. The late GDN layers, above all
62, write most of the memory's effect on the decision, both the generic shift and the presence
effect; attention 51 and 63 add less. Attention 59 writes most of the baseline pull toward calling,
but not the memory's change to it. For the generic shift, the simplest account is that attention
reads the history (mid layers alone suffice) and the late GDN layers pass it to the output, but
direct attribution cannot test that. For presence, those GDN writes need the KV and the GDN state
together (C +0.07, CK −0.84). Nothing here locates where attention reads the memory.

**Branch rules, applied as written.**
- **K vs M.** The first half is met (slope 1.00 ≥ 0.5). The second half is not: K matches M's first
  action in 15/32 sessions against C's 14/32 (paired bootstrap difference +0.03 [−0.22, +0.28];
  with absent2, 24/48 vs 26/48). The stop branch (slope < 0.2) is not met either, so the rule gives
  no verdict.
- **K\* vs K.** "About 80%" is met: 102% (K*64k) and 79% (K*4k).
- **Bands.** As written, the rule points at the mid band (97% of K's slope).

**Amendment after seeing the data (2026-10-10, post hoc).** I go on to Stage 1, with CK as the
ceiling instead of K, and say so here because the rule did not name CK.
- **Why CK.** It is the only partial condition close to M on content (94%) and on presence in the
  hazard (r 0.71). A seed would sit next to the compiled GDN state anyway.
- **Layers.** I seed all 16 attention layers rather than the mid band, because no single band
  carries content.
- **Positions.** Seeds use collapsed positions (the K* rule). CK* (CK with collapsed positions,
  uncompressed) is measured as the seed's own ceiling, because collapse removed the first action's
  presence sensitivity in K.
- **Contrasts.** Stage 1 reads present − absent (the store-specific part), not X − A, which is
  mostly generic.

**Caveats.**
- **Mismatch confound.** Both mismatched conditions are off-distribution. K pairs a start-of-sequence
  GDN state with 108k tokens of KV; C pairs a 108k GDN state with no KV, and visibly suffers. So
  "K < CK" may mean the mismatch breaks retrieval, not that the GDN state holds the content. Both
  K's KV and C's state were computed by the full model.
- **Noise floor.** The Krt floor at 108k is larger than the 6k plumbing figure: |L(K) − L(Krt)|
  median 0.07 and max 0.22, and one first action in 32 differs. Per-question signs near zero are not
  interpretable.
- **Fragile results.** Several results are borderline: CK presence on L, K*4k − K on the anchor, and
  the Kearly slope. They are exploratory, not effects.

**Checking the claims** (a workflow orchestrated with Claude Code). Four agent checks, a written checklist executed once, were
run before this entry:
- recompute every number with fresh code;
- inference and confounds;
- the experiment and analysis code;
- pre-registration.

The recomputation found one slip in my draft (late GDN in A is +1.8, not +1.6). The code
check found no bug that changes a claim. The qualifications above (generic vs store-specific,
chance rates for sign agreement, readout dependence, and the rule outcomes) come from these checks.

## 2026-10-10 (night) — Stage 1 development: a derived seed (setup)

**Construction** (`code/statelm/kvseed.py`; exploratory). Each attention layer and KV head gets
N = T / r entries, consumed by unchanged attention as extra KV plus a per-entry additive logit bias.
They are fitted only to the memory's own compile:
- **Keys.** k-means centroids (20 Lloyd iterations, seeded) of the un-rotated keys, all at
  position 0, so the seed is position-free.
- **Bias.** Starts at log(cluster size), then is fitted by Adam (300 steps) so the seed's log
  attention mass matches the full memory's for each reference query.
- **Values.** Ridge least squares toward the cluster-mean values, fitted to the full memory's
  attention output. The ridge is picked on a validation split.
- **Reference queries.** The memory's own pre-RoPE queries at 6,144 seeded token positions, rotated
  to the session's distance D. These come from the q_norm output (`QueryTap`) and give 36,864
  queries per KV head. The session's question is never used.
- **Metrics.** Held-out log-mass RMSE and relative output error, on 10% of the queries that the fit
  never sees.

This is Attention Matching [zweiger2026fast] with centroid keys [zhan2026reskv] in place of a key
subset.

**Code changes.**
- `_sdpa_kvbias` now also takes a bias per KV head, [Hkv, N], expanded to the query heads the way
  `repeat_kv` groups them.
- New `tests/check_kvseed.py` cases on the real model (5,885-token memory), both PASS:
  - S10: a C cache filled through `cache.update` with the un-rotated keys and values, a zero
    [Hkv, T] bias and the session at D equals CK*: max KL 0, logits identical.
  - S11: a [Hkv, T] bias with equal rows is bitwise equal to the same [T] bias.
  - S1–S9 are unchanged from the 15:27 run.
- New `tests/check_kvseed_fit.py`, a synthetic check with no model, PASS. On clustered unit-RMS keys
  with T/N = 16, the fitted bias cuts held-out log-mass RMSE from 0.044 to 0.005, the fitted values
  cut output error from 0.084 to 0.072, and rounding the bias to bf16 (as the mask does) changes
  neither.
- **Fitting time** at 108k tokens (synthetic): about 20 s per head at r = 16, 6 s at r = 64 and
  2–3 s at r = 256. A whole store is 64 heads: about 21, 6 and 3 min.

**Development plan** (dev16 questions only; nothing confirmatory).
- **Script.** `experiments/lme_seed.py` runs A, C, CK, CK*, S_r (the fitted seed next to the GDN
  state; r = 16, 64, 256), optionally S0_r (unfitted: centroids, log cluster size, mean values) and
  M, on present and absent stores, with Stage 0's decision measures and the anchor.
- **First run.** One question, to see fit quality on real keys (attention sinks, peaked heads) and
  check the pipeline.
- **Then.** Fix D and the reference-query set on dev before writing a confirmatory plan for a fresh
  sample.

## 2026-10-10 (night) — Stage 1 dev run 1 result: the seed keeps some content at 16×, little at 64×

**Run.** One dev question, 6f9b354f (single-session-user, 111,363 tokens present; gold "a lighter
shade of gray"). Ratios r = 16, 64, 256 give N = 6,960, 1,740 and 435 entries per KV head. D is
65,536 and the reference set is 6,144 tokens of the memory's own queries.
- Wall time: 4,006 s, of which fitting was 3,522 s (2,432 + 748 + 342 s over the two stores). Peak
  allocation: 79.4 GiB.
- Output: `results/phase2/lme_seed_dev1.jsonl`; log `results/audit/logs/lme_seed_dev1.log`;
  summary `code/analysis/lme_seed.py`.
- **Seed sizes** (key and value in bf16, fp32 bias, 64 KV heads): 458 MB at r = 16, 114 MB at
  r = 64 and 29 MB at r = 256. The full KV is 7.3 GB and the GDN state is 151 MB.
- **Replay check.** A, C, CK and M match Stage 0's rows for this question to four decimals on
  every L and anchor value, so the pipeline is deterministic across scripts.

**Fit quality** (held-out queries; medians by layer).
- After the value fit, the relative output error at r = 16 runs from 0.03 (layer 3) to 0.50
  (layer 35). Layers 23–39 are all at or above 0.39, and layers 59 and 63 are at 0.16 and 0.08.
- Mean over the 64 heads, fitted against unfitted:

  | r | unfitted | fitted |
  |---|---|---|
  | 16 | 0.42 | 0.30 |
  | 64 | 0.50 | 0.36 |
  | 256 | 0.55 | 0.41 |

- The fit is not overfitting: train and held-out error are close (layer 27 at r = 16: 0.43 and
  0.46). So the limit is capacity or the shape of the keys, not too few queries.
- The bias fit cuts log-mass RMSE about threefold (layer 27 at r = 16: 1.47 to 0.58). On its own
  it leaves the output error unchanged or slightly worse; the value fit does the output work.

**Behaviour on the present store.** Content is the anchor, log p(gold):

| condition | present | absent | p − a |
|---|---|---|---|
| A | −23.6 | | |
| C | −25.4 | −25.6 | +0.2 |
| CK | −1.6 | −22.3 | +20.7 |
| CK* | −2.1 | −23.6 | +21.5 |
| M | −3.1 | −23.1 | +20.1 |
| S16 (fitted) | −14.7 | −22.5 | +7.8 |
| S0_16 (unfitted) | −12.9 | −22.9 | +10.1 |
| S64 | −19.1 | −23.2 | +4.1 |
| S0_64 | −22.0 | −22.2 | +0.2 |
| S256 | −24.4 | −24.7 | +0.3 |
| S0_256 | −22.7 | −22.6 | −0.05 |

- At r = 16 the seed carries 8–10 of the 21.5 nats that CK* carries.
- At r = 64 the fitted seed keeps 4 nats and the unfitted one keeps none.
- At r = 256 nothing is left.
- The fitted seed beats the unfitted one at r = 64 but not at r = 16. This is one question, so
  neither ordering means anything yet.

**Decision.** L (first-token call log-odds), present / absent:
- A +1.97.
- C +3.26 / +3.23; the GDN state alone pushes toward calling, as in Stage 0.
- M +0.10 / +0.93; CK −0.71 / +0.81; CK* −1.52 / −0.34.
- S16 −0.16 / +0.29; S64 +0.96 / +1.28; S256 +1.09 / +0.47.
- p − a: M −0.82, CK −1.52, CK* −1.18, S16 −0.45, S64 −0.32, S256 +0.62. The unfitted seeds give
  −0.10, +0.09 and +0.47.
- First actions: every seed calls search_memory("bedroom walls…") on both stores. CK on the present
  store answers without calling. CK* on the present store first writes "You repainted your bedroom
  walls a lighter shade of gray. You mentioned that it…" and then calls.

**What I take from it** (one question; descriptive only).
- A seed one sixteenth the size of the KV keeps some content and some of the presence signal on L.
  At about GDN-state size (r = 64) it keeps little of either.
- The fit is worst in the middle layers (23–39).
- I can't tell which of two causes limits the seed. One is the reference set: the memory's own
  queries are causal, mostly local reads, while the test-time queries (question, anchor) are
  long-range lookups. The other is capacity: centroids of 16 keys blur a single fact's key.
- The next dev run separates the two.

## 2026-10-10 (night) — Stage 1 dev run 2: which reference queries (setup)

**Question.** Is the seed limited by its reference queries or by its capacity at a given N?

**Changes to `experiments/lme_seed.py`.** dev run 1's behaviour is unchanged under the defaults.
- `--refs`: reference sets sharing one k-means per head, so the seeds differ only in bias and values.
  - **own:** as in dev run 1.
  - **rep:** repeat-prefill queries, the reference set KVzip and Attention Matching use
    [kim2025kvzip, zweiger2026fast]. After the compiled memory, a user turn asks "Repeat excerpts
    from the memory above, word for word." The assistant turn is 12 seeded chunks of 520 memory
    tokens, separated by "[...]" and teacher-forced. I keep the queries at each chunk's tokens after
    its first 8, which gives 6,144 tokens. These queries look things up at long range, as a question
    does, but they come from the memory alone. The pass runs on the full cache after the M, C, CK
    and CK* rows are done, and the KV is cut back to the memory before fitting.
  - **mix:** 3,072 tokens of own and 3,072 of rep.
  - **probe** (diagnostic only, never a candidate method): CK*'s own queries at the last 64
    positions of the session prompt and at every anchor-scoring position, each rotated to its exact
    position D + j. This is fitted on the test-time queries, so it is a ceiling on what better
    reference queries could buy at that N. It is not a method.
- Every fitted seed also reports its output error and log-mass RMSE on the probe queries, against
  CK*'s targets. This measures the approximation where it is used; it does not feed any fit.
- `--stores p` runs the present store only. This run reads the anchor on the present store against
  C, CK* and A from the same run. The p − a decision contrast waits for a run with both stores.
- `fit_head` takes a precomputed k-means (`km=`). With the default seed and 20 iterations it gives
  the same centroids as dev run 1, so S16 own on 6f9b354f should repeat dev run 1's row.

**Run.** Two dev questions with short golds where CK carries the content:
- 6f9b354f (single-session-user, CK p −1.6 / a −22.3);
- 0e4e4c46 (knowledge-update, gold "132 points", CK p −0.0 / a −15.3).

Settings: r = 16 and 64; refs own, rep, mix, probe; plus the unfitted seeds; present store only.
A smoke run first: 6f9b354f, r = 256, 1,536 reference tokens, to `results/audit/smoke/`.

**What I'll read** (dev; this sets the reference set, it is not a test).
- **Probe error:** probe output error per layer for own against rep against mix. If own's probe
  error is far above its held-out error, the reference set is mismatched.
- **Anchor:** present-store anchor against CK* and C for each set. If probe-fitted seeds recover the
  anchor and own does not, the reference set is the bottleneck; if probe also fails at r = 64,
  capacity is.
- **Selection rule:** I take forward whichever of own, rep and mix has the lowest probe output
  error at r = 64, averaged over heads, questions and the two probe kinds (session, anchor). If the anchor disagrees, I note it and still go by probe error, which is
  measured on 64 heads × 2 questions, not two anchors.

**Amendment after the smoke run, before dev run 2** (`results/audit/smoke/lme_seed_dev2_smoke.jsonl`:
6f9b354f, r = 256, 1,536 reference tokens, present store; it ran in 499 s, peak 72.4 GiB).
- **Probe error at r = 256**, output error on the session queries / on the anchor queries, mean
  over heads:
  - own 0.48 / 0.51;
  - rep 0.60 / 0.58;
  - mix 0.48 / 0.50;
  - probe 0.33 / 0.32, mostly in-sample.
- **Anchor, present store:** own −25.2, rep −22.1, mix −24.0, probe −13.8; C −25.4, CK* −2.1.
  At 29 MB, a seed fitted on the right queries keeps about 10 nats where every memory-only set
  keeps none.
- **So** the reference set matters, and repeat-prefill queries resemble the test-time queries
  less than the memory's own queries do. That rules rep out as the answer.
- **Change.** I add a self-study set, ss [eyuboglu2026cartridges, zweiger2026fast], and drop mix,
  which matched own on probe error.
  - For 48 seeded memory chunks of 384 tokens, the model, with no memory loaded, writes one
    question the user might later ask about the chunk (prompt `SS_ASK` in the script; greedy, up
    to 48 tokens).
  - Each question is asked on CK* twice:
    - in the tool session, with the queries at its last 64 positions;
    - in the anchor format, where the model writes its short answer greedily (up to 24 tokens),
      with the queries at every prompt and answer position.
  - That is about 6,000 tokens, close to own's 6,144.
  - The tool session's system prompt carries the test session's date, which is not question
    content.
  - The questions come from the memory alone and the test question is never seen. A synthetic
    question can still land on the test's fact if its chunk falls in the evidence session. So each
    question records its chunk start, and the meta row records the evidence span
    (`evidence_p`); I'll report the overlap.
- **dev run 2 now runs** refs own, rep, ss and probe at r = 16 and 64, plus the unfitted seeds,
  present store only, on the same two questions.
- **Selection rule, amended:** I take forward whichever of own, rep and ss has the lowest probe
  output error at r = 64, averaged over heads, questions and the two probe kinds.

**Second smoke run and a determinism fix** (before dev run 2).
- **Self-study smoke** (`results/audit/smoke/lme_seed_dev2_smoke2.jsonl`; 4 questions, 424 query
  tokens, r = 256): the path runs (390 s, peak 79.5 GiB).
  - The model wrote one question per chunk, e.g. "Who scored the second goal for Argentina?" with
    the answer "Di Maria", and "What is the guide number of the Canon Speedlite 470EX-AI?" with the
    answer "47".
  - None of the four chunks fell in the evidence span, tokens 79,645–82,901.
- **Run-to-run noise.** The probe fit at r = 256 came out differently in the two smoke runs: anchor
  −13.83 vs −13.76, L +0.59 vs +0.76.
  - The k-means sums used CUDA `index_add_`, which adds floats in arbitrary order. Over 20 Lloyd
    iterations, that can flip borderline assignments.
  - I replaced it with `kvseed.cluster_sums`, sorted segment sums of an fp64 cumsum, used for the
    centroids and the cluster-mean values.
  - Check (`results/audit/smoke/det_check.py`, synthetic, T = 111,363, N = 6,960): two `fit_head`
    calls now agree bitwise on C, b, U, b0 and the mean values. The sums agree with an fp64
    `index_add_` to 3e-5. `tests/check_kvseed_fit.py` still passes.
- **Consequences:**
  - dev run 1's seeds were fitted with the order-dependent sums, so their r = 256 differences of
    about 0.2 in L sit inside that noise.
  - S16 own on 6f9b354f in dev run 2 will not exactly repeat dev run 1.
- dev run 2 started at 22:23: `results/phase2/lme_seed_dev2.jsonl`, log
  `results/audit/logs/lme_seed_dev2.log`.

## 2026-10-11 (early morning) — Stage 1 dev run 2 result: the fit helps the lookups it was fitted on

**Run.**
- Questions: 6f9b354f and 0e4e4c46, present store only.
- r = 16 and 64; refs own, rep, ss and probe, plus the unfitted seeds.
- Wall time: 10,887 s (5,471 and 5,416 s per question). Fitting took 6,852 s at r = 16 and
  2,262 s at r = 64, all refs together. Peak allocation: 83.3 GiB.
- Output: `results/phase2/lme_seed_dev2.jsonl`; log `results/audit/logs/lme_seed_dev2.log`.
- **Replay check.** A, C, CK, CK* and M match dev run 1 on 6f9b354f.

**Probe error** (relative output error on CK*'s test-time queries, mean over the 64 heads):

| q | r | unfitted sess / anchor | own | rep | ss | probe (ceiling) |
|---|---|---|---|---|---|---|
| 6f9b | 16 | 0.38 / 0.45 | 0.39 / 0.40 | 0.54 / 0.69 | 0.43 / 0.68 | 0.21 / 0.22 |
| 6f9b | 64 | 0.47 / 0.54 | 0.44 / 0.45 | 0.55 / 0.61 | 0.38 / 0.53 | 0.28 / 0.27 |
| 0e4e | 16 | 0.38 / 0.42 | 0.40 / 0.42 | 0.49 / 0.63 | 0.21 / 0.28 | 0.22 / 0.21 |
| 0e4e | 64 | 0.45 / 0.49 | 0.45 / 0.46 | 0.50 / 0.54 | 0.29 / 0.34 | 0.28 / 0.27 |

**Anchor, present store** (log p(gold)):

| q | C | CK* | own 16 / 64 | rep 16 / 64 | ss 16 / 64 | probe 16 / 64 | unfitted 16 / 64 |
|---|---|---|---|---|---|---|---|
| 6f9b | −25.4 | −2.1 | −14.5 / −19.3 | −18.1 / −19.8 | −17.9 / −24.1 | −4.2 / −10.1 | −12.9 / −22.0 |
| 0e4e | −17.8 | −0.01 | −5.8 / −7.8 | −8.2 / −8.0 | −3.2 / −3.0 | −0.17 / −1.3 | −3.5 / −8.5 |

**Self-study coverage.**
- On 0e4e4c46 one of the 48 sampled chunks (start 26,022) lies in the old-score evidence session.
  Its question was "What are some tips for improving my Ticket to Ride score beyond 124 points?"
- On 6f9b354f no chunk touches the evidence session (tokens 79,645–82,901), and none of the 48
  questions is about painting or walls.
- `evidence_p` is not informative for 0e4e4c46. Knowledge-update questions have two evidence
  sessions, so the span (25,672–107,520) covers most of the memory.

**First actions, present store.**
- 0e4e4c46:
  - M and CK answer "132 points" without calling.
  - CK* writes the right answer and then calls.
  - S16 own answers "**100**" without calling.
  - S16 ss answers "**142 points**" and runs on to the 96-token cap.
  - The other seeds call.
- 6f9b354f: every seed calls. S16 rep first says "I don't have any record of you repainting your
  bedroom walls…".

**What I take from it** (two questions; descriptive).
- **Capacity is not the main limit at 64×.** Fitted on the test-time queries, a 114 MB seed
  carries 15 of the 23 nats on 6f9b354f and 16.5 of the 17.75 on 0e4e4c46. At 16× it carries
  21 and 17.6.
- **The reference queries decide what the seed keeps.** Self-study reaches the ceiling's probe error
  on the question whose topic it happened to sample (0e4e: 0.29 / 0.34 against 0.28 / 0.27 at
  r = 64). On the question it did not sample, it is no better than unfitted on the anchor queries,
  and at r = 16 it is worse (0.68 against 0.45). So the value fit buys the lookups it saw at the
  cost of the ones it didn't.
- **Repeat-prefill queries are the worst reference set** for the test-time queries at both ratios
  and on both questions.
- **A partial seed gives the model the gist and makes it confident without the specifics.** On
  0e4e4c46 two seeds answer from memory with the wrong number instead of searching. That matters
  for a deployed seed: the failure mode is a confident wrong answer, not a search.
- **Selection rule (logged before the run).** Probe output error at r = 64, averaged over heads,
  questions and the two probe kinds: ss 0.382, own 0.450, rep 0.548. So ss goes forward. Its
  anchor advantage is confined to the covered question, which is what the probe errors also say.

## 2026-10-11 (early morning) — Stage 1 dev run 3: self-study tiled over the whole memory (setup)

**Question.** Does self-study that covers every memory chunk get the seed near the probe ceiling
at 64× on a question whose evidence random sampling missed?

**Change: a new reference set, sst.** It is ss with consecutive, non-overlapping 384-token chunks
over the whole memory: 290 chunks at T = 111,363. Its queries are stored in bf16 after rotation to
fit in memory (about 7 GB), at about 36k query tokens per store. Everything else is as in ss.

**Run.**
- Questions: 6f9b354f (missed by sampling in dev run 2) and 0862e8bf (single-session-user, gold
  "Luna", new to Stage 1; Stage 0 CK p −0.0 / a −10.0, C p −11.3).
- r = 64 only (114 MB, about the GDN state's size); refs own, sst and probe, plus the unfitted
  seed; present store only.
- I expect about 1 h 40 min per question, most of it generating the 290 questions and their answers.
- A smoke run first: 6f9b354f, r = 256, sst with 24,000-token chunks (4 chunks), to
  `results/audit/smoke/`.

**What I'll read.**
- Anchor on the present store for sst against probe, own and C.
- Probe errors for sst against probe and own.
- Whether any sst questions are about the evidence fact; I'll list the ones from the evidence
  session.

If sst gets close to probe on 6f9b354f, the remaining gap is coverage, and I'll write the
confirmatory plan around sst at r = 64. If not, then even questions about the evidence session
don't produce the test-time lookups, and the seed has to be built differently.

**Smoke run** (`results/audit/smoke/lme_seed_dev3_smoke.jsonl`; 6f9b354f, r = 256, sst with
24,000-token chunks): it ran in 617 s. The 4 questions took 152 s and gave 431 query tokens.
A, M, C, CK and CK* repeat dev run 2's values. dev run 3 started at 01:52:
`results/phase2/lme_seed_dev3.jsonl`, log `results/audit/logs/lme_seed_dev3.log`.

## 2026-10-11 (morning) — Stage 1 dev run 3 result: covering the memory matches the error, not the content

**Run.**
- Questions: 6f9b354f and 0862e8bf, present store, r = 64. Refs own, sst and probe, plus the
  unfitted seed.
- Wall time: 10,219 s.
- sst generation took 2,785 s and 2,590 s: 290 and 273 questions, which gave 30,692 and 29,051 query
  tokens.
- Fitting took 2,210 s and 1,979 s for all refs.
- Output: `results/phase2/lme_seed_dev3.jsonl`; log `results/audit/logs/lme_seed_dev3.log`.

**Probe error** (relative output error on CK*'s session / anchor queries, mean over 64 heads):

| q | unfitted | own | sst | probe |
|---|---|---|---|---|
| 6f9b | 0.47 / 0.54 | 0.44 / 0.45 | 0.29 / 0.35 | 0.28 / 0.27 |
| 0862 | 0.43 / 0.48 | 0.44 / 0.42 | 0.26 / 0.30 | 0.26 / 0.25 |

**Anchor, present store** (log p(gold)):

| q | C | CK* | unfitted | own | sst | probe |
|---|---|---|---|---|---|---|
| 6f9b ("a lighter shade of gray") | −25.4 | −2.1 | −22.0 | −19.3 | −17.5 | −10.1 |
| 0862 ("Luna") | −11.3 | −0.00 | −10.1 | −8.7 | −8.65 | −0.04 |

**The evidence was covered.** Nine sst chunks lie in each evidence session, and their questions
touch the fact:
- "Do you think the lighter gray walls in my bedroom would be a good color choice for displaying a
  ZZ Plant?"
- "How can I help Luna adjust to wearing a new collar?"

No question asked for the fact itself (the colour; the cat's name).

**First actions on 0862e8bf.**
- M answers "Your cat's name is Luna!".
- CK and CK* say the same and then call.
- The fitted seeds call after "I don't have any information about your cat's name…" (own) and
  "I don't have a record of your cat's name…" (sst).
- probe calls search_memory("cat Luna").

**What I take from it** (two questions; descriptive).
- **Mean probe error is the wrong yardstick for content.** sst matches the probe-fitted seed's
  error to within 0.01–0.08 and still recovers 0–8 of the nats that probe recovers.
- **What carries the answer is a few lookups at a few positions,** which an average over all
  heads and positions barely registers.
- **The probe "ceiling" is in-sample on those lookups.** It shows that a 114 MB seed can serve
  them. It doesn't show that it can serve them alongside every other lookup the memory needs.
- **Open: fact coverage or capacity?** It could be fact-level coverage: one question per chunk
  doesn't ask for the colour or the name. Or it could be capacity: N = 1,740 centroid entries
  can't serve hundreds of facts' lookups at once. The next run tests this directly.

## 2026-10-11 (morning) — Stage 1 dev run 4: can one seed serve the test lookups alongside the whole memory's? (setup)

**Question.** dev run 3 leaves two explanations open:
- **Fact coverage:** the self-study questions never ask for the fact.
- **Capacity:** N = 1,740 centroid entries per head can't serve the test lookups together with
  everything else.

This run separates them by adding the test-time queries to the whole-memory reference set.

**Diagnostic sets** (not methods; both use the test-time queries).
- **sstp:** sst plus the probe queries once. That is about 470 queries per KV head against about
  180,000, which is roughly the weight of one more self-study question.
- **sstpw:** sst plus the probe queries repeated 20 times.

**Reading.**
- If sstp recovers the anchor near probe's level (dev run 3, r = 64: 6f9b −10.1, 0862 −0.04),
  capacity is enough. The gap is then which questions self-study asks, and the next step is
  fact-level self-study.
- If only sstpw recovers it, the seed can hold the lookup but the fit has to be pointed at it, so
  weighting matters.
- If neither does, the 64× centroid seed can't serve this lookup alongside the memory's other
  lookups, and the construction itself (key placement, per-head budgets) has to change.

**Code.**
- A reference set can now be a weighted concatenation of query sets.
- With `--ss_cache`, the tiled self-study queries (bf16, about 7 GB per store) and their log are
  saved to a local cache outside the entry and reused
  by later runs. So fitting experiments don't regenerate them.

**Run.**
- Questions: 6f9b354f and 0862e8bf, present store, r = 64.
- Refs sstp, sstpw and probe. sst itself is not refitted: generation is greedy and the fit is now
  deterministic, so dev run 3's sst rows are the comparison.
- A smoke run first, in two passes (generate and save, then load): r = 256, 24,000-token chunks.

## 2026-10-11 (morning) — review of Stage 1 dev runs 1–3: corrections and fixes before dev run 4

**What happened.** Before dev run 4 I ran a review pass: five subagents, orchestrated with Claude
Code, covering literature, construction, self-study and a critic of dev runs 1–3, plus a synthesis.
The pass is no credential. I checked each point below against the jsonl myself.

**Corrections to earlier entries.** The entries above stay as written; these supersede them.

1. **Per token, the anchor is mostly format on 6f9b354f.** Its tokens are 'a', ' lighter', ' shade',
   ' of', ' gray'.
   - 'a' (A −12.6, C −14.3, CK* −2.1) is format: whether the answer starts with "a".
   - ' lighter' (C −7.36) is the first content token.
   - ' gray' sits at −2.8 under A and C. No seed at r = 64 moves it, the probe-fitted one included
     (−2.81). S16probe gets it to −1.27.
   - So dev run 2's "a 114 MB seed carries 15 of the 23 nats on 6f9b354f" is mostly 'a' and
     ' lighter'; the colour itself is not carried.
   - 0862e8bf is clean. Its first token 'L' (of "Luna") carries everything: C −11.15, unfitted
     −9.98, sst −8.62, probe −0.04, CK* −0.00.
   - On 0e4e4c46 the second digit '3' separates the gold 132 from the stale 124: A −3.91, C −3.60,
     S64ss −0.46, probe −0.08.
2. **dev run 1: the content at r = 16 comes from the unfitted construction.** Centroid keys, log
   counts and mean values carry 10.1 nats; the fitted seed carries 7.8; ' gray' moves about 0.5.
   - The fit did not add content at r = 16; at dev run 2's r = 16 the unfitted seed beat the fitted
     one on the anchor of both questions.
   - "Not overfitting, so the limit is capacity or key shape" does not follow. The split was by
     row, so held-out rows shared tokens (the other q heads) and items with the fit set.
3. **The self-study sets duplicated one block of queries.** Of the 64 tapped session positions,
   41–44 are the tool session's system-prompt tail. Those queries come before the question, so they
   are identical for every self-study item. The review counts each of them 227–251 times in sst,
   about 35% of its rows. I have not re-tokenised to check that count; from the item sizes
   (30,692 query tokens over 290 items, 64 of them per item from the session) I estimate 29–35%.
   - That duplication drove "ss/sst match the probe's session error" and dev run 2's selection rule
     (ss 0.382).
   - On the anchor rows alone: ss 0.43 against own 0.45 (dev run 2, r = 64). In dev run 3, sst is
     worse than probe on the anchor rows in 61/64 heads (6f9b) and 54/64 (0862).
4. **The probe ceiling is in-sample and under-determined.** It was fitted on about 470 rows against
   1,740 entries, with the ridge chosen on about 58 rows.
   - It shows the model reads a seed entry for those lookups, not that 1,740 entries can serve them
     alongside everything else.
   - Only 0862e8bf gives a clean ceiling ('L' −0.04).
5. **"Partial seeds give confident wrong answers" rests on two greedy generations at r = 16,** on
   the present store only, with no foil scoring. It is plausible, not established.
6. **"Repeat-prefill is the worst reference set" is confounded.** rep covered about 6% of the
   memory, was rotated to D rather than D + j, and was tapped on M rather than CK*.
7. **The ridge grid's floor was binding.** sst chose 1e-3, the grid's lowest value, on 64/64 heads
   for both dev run 3 questions; ss on 63/64 at r = 64 (dev run 2); rep on 60–62/64. own did so on
   31–37/64 and probe on 6–10/64.

**Fixes.**
- **Grouped splits** (`kvseed.fit_head`, `groups=`). The q heads of one token, or the rows of one
  self-study item, now stay in one split.
- **Wider ridge grid,** 1e-5 to 1, solved in fp64.
- **Diagnostic rows enter the fit by weight** (`extra=`, `extra_w=`). Probe rows always go to the
  fit set with a per-row weight instead of being repeated, and validation and held-out rows come
  from the reference set only. Before this, sstp would have dropped about 20% of the probe rows at
  random and reshuffled the whole split.
- **No raw keys from empty clusters.** An empty cluster now gets a zero key, a zero value and bias
  −30. It used to keep a raw key, which the constraints exclude.
- **Self-study taps the session from the user turn on.** The system-prompt tail is tapped once, for
  the first item (48 positions, its own group). Every query token carries its item id. The cache
  file name is a hash of the model, prompts, settings, store and date.
- **Metrics:**
  - per-row p90 and maximum relative error;
  - a 'gold' probe subset, the anchor rows that predict the gold tokens;
  - per-token anchor tables in `analysis/lme_seed.py`;
  - fitted seeds can be saved (`--seed_dir`) for diagnostics without a refit.
- **Live check S12** (`tests/check_kvseed.py`, all of S1–S12 PASS). A 64-entry seed gets a
  different bias per KV head. Layer 35's attention output, recorded inside the model during a
  one-token session, matches the seed formula the fit uses (KV head h serves query heads
  6h–6h+5) to 0.19% relative error. The same formula with the wrong grouping misses by 118%.
  - My first version of S12 failed both groupings (relative error 6.4) because it indexed the
    recorded output on the wrong axis. That was a bug in the test, now fixed.
- **Still open, before any confirmatory run:**
  - Self-study uses the test session's date, and the same anchor and tool-session templates as
    the test. The confirmatory plan has to use the memory's last-session date and paraphrased
    templates, and score one held-out template.
  - Every run since dev run 1 is present-store only. Any run that decides a branch needs the
    absent store too.

## 2026-10-11 (morning) — Stage 1 dev run 4, revised setup and reading rule (before the run)

This replaces the dev run 4 setup above; that run was never launched.

**Sets** (all at r = 64, all fitted with the fixes above):
- **sst:** refit, because deduplication, grouped splits and the wider ridge change it.
- **sstp:** sst plus the probe rows, always in the fit set, weight 1.
- **sstpw:** the same with weight 20.
- **probe.**
- The unfitted seed as well.

Questions 6f9b354f and 0862e8bf, present store only: this run decides between explanations for the
content gap. It is not a method test, so the absent store waits for the method run. Fitted seeds are
saved to the local seed cache for later diagnostics.

**Reading rule.**
- **Tokens.** Primary: 0862e8bf's first gold token 'L'. Secondary: 6f9b354f's ' lighter'. I also
  report ' gray', which no seed has moved so far.
- **Recovery fraction** on a token: f_x = (lp_x − lp_sst) / (lp_probe − lp_sst), with sst and probe
  from this run.

| f_sstp | f_sstpw | branch |
|---|---|---|
| ≥ 0.7 | — | (a) **coverage:** one seed can serve the test lookup alongside the whole memory's once that lookup is in the fit; next is fact-level self-study |
| < 0.3 | ≥ 0.7 | (b) **weighting:** the lookup can be held but the fit must be pointed at it; next is derived lookup weights on top of fact-level self-study |
| — | < 0.3 | (c) **construction or drift:** first rule out query drift (the seeded model's queries moving away from CK*'s), then key construction and per-head budgets |
| otherwise | | **mixed:** (b) and (c) both |

- If the primary and secondary tokens disagree, the primary decides and I record the disagreement.
- The smoke run comes first (two passes: generate and save, then load) and must give identical fits.

**Smoke run** (`results/audit/smoke/dev4_smoke.sh`; 6f9b354f, r = 256, sst with 24,000-token
chunks, 4 items, 341 query tokens).
- The first version crashed: 5 groups at 10% gave an empty validation split. Validation and
  held-out now get at least one group each.
- Rerun: pass 1 generated and saved the queries (675 s); pass 2 loaded them (523 s). Every head's
  fit statistics and every anchor token are identical between the passes.
- Tapping from the user turn gives 22–39 session rows per item; the first item has 77, which
  includes the 48 system-tail positions.
- At r = 256, ' lighter':

  | set | ' lighter' |
  |---|---|
  | sst | −7.82 |
  | sstp | −1.81 |
  | sstpw | −0.49 |
  | probe | −1.36 |

  ' gray' stays near −2.5 throughout. This is four items and one question, so it only shows the
  path works.
- dev run 4 started at 06:22: `results/phase2/lme_seed_dev4.jsonl`, log
  `results/audit/logs/lme_seed_dev4.log`.

## 2026-10-11 (morning) — Stage 1 dev run 4 result: the seed can hold the lookup, but the fit has to be pointed at it

**Run.**
- Questions 6f9b354f and 0862e8bf, present store, r = 64 (N = 1,740 entries per KV head, 114 MB).
  Refs sst, sstp, sstpw (weight 20), probe, plus the unfitted seed.
- Wall time: 12,878 s, ending at 09:57.
- sst was regenerated in 2,807 s and 2,625 s. With the system-prompt tail tapped once, it has
  19,907 and 19,098 query tokens (dev run 3: 30,692 and 29,051) in 291 and 274 groups.
- Fitting took 6,764 s for all refs and both questions.
- Output: `results/phase2/lme_seed_dev4.jsonl`; log `results/audit/logs/lme_seed_dev4.log`.
  Fitted seeds are in the local seed cache.

**Size of the diagnostic rows.**
- The probe rows are 582 and 558 per KV head: 97 and 93 tokens × 6 q heads.
- Their share of the fit-set weight: sstp 0.6%, sstpw 10.9%. For example, 582 × 20 /
  (94,980 + 11,640).

**Ridge.** The grid floor no longer binds: sst is at 1e-5 on 8/64 heads and at the 1.0 ceiling on
3–4/64, with a median of 1e-2 to 6e-2. probe is at the ceiling on 14–19/64.

**Per-token anchor, present store** (log p):

| token | C | unfitted | sst | sstp | sstpw | probe | CK* |
|---|---|---|---|---|---|---|---|
| 0862 'L' | −11.15 | −10.07 | −9.39 | −7.44 | −0.48 | −0.03 | −0.00 |
| 6f9b 'a' (format) | −14.32 | −11.18 | −13.16 | −11.99 | −7.70 | −6.53 | −2.10 |
| 6f9b ' lighter' | −7.36 | −6.98 | −6.58 | −5.36 | −3.12 | −0.11 | −0.00 |
| 6f9b ' gray' | −2.86 | −3.01 | −2.53 | −2.50 | −2.53 | −2.74 | −0.00 |

**Recovery fractions** f = (lp_x − lp_sst) / (lp_probe − lp_sst):

| token | f_sstp | f_sstpw | branch |
|---|---|---|---|
| 0862 'L' (primary) | 0.21 | 0.95 | (b) weighting |
| 6f9b ' lighter' (secondary) | 0.19 | 0.53 | mixed |
| 6f9b ' gray' | n/a | n/a | probe itself is below sst (−2.74 vs −2.53), so f is undefined |

**Reading by the logged rule: branch (b), weighting.**
- The primary token decides. I record the disagreement: ' lighter' recovers only about half at
  weight 20, and ' gray' is moved by no seed, the in-sample probe fit included.

**Interference.** Adding the test lookup at 11% of the fit weight cost nothing measurable on the
memory's held-out lookups. Item-held sst error under sstpw against sst alone:
- 0862: 0.348 vs 0.351 (p90 0.471 vs 0.475).
- 6f9b: 0.338 vs 0.338 (p90 0.469 vs 0.472).

**Error on the gold-predicting rows** (mean over heads, 0862 / 6f9b):

| sst | sstp | sstpw | probe |
|---|---|---|---|
| 0.39 / 0.45 | 0.36 / 0.42 | 0.30 / 0.35 | 0.26 / 0.32 |

On 0862, the 0.05 between sstpw and probe is the difference between 'L' at −0.48 and −0.03.

**Decisions.** No seed stops the first call.
- On 0862e8bf, sstpw's first-token log-odds is −0.83, but the greedy path still calls
  search_memory("cat") (H +5.31). probe calls search_memory("cat Luna"). CK and CK* also call
  ("cat's name"); M answers.
- On 6f9b354f every seeded condition calls, as M does.
- The decision measures don't separate the sets on these two questions.

**What I take from it** (two questions, one clean token; exploratory).
- **On 'L', capacity isn't the limit.** At 64× the seed served the test lookup alongside the whole
  memory's without measurable cost to held-out lookups. But the fit had to be pointed at it: at
  0.6% of the fit weight the lookup's rows moved 2 of 9.4 nats; at 11% they moved 8.9.
- **What this does not show:** that a fit can be pointed at the right lookups without knowing the
  test question. The weights here sit on the test's own rows. A memory has hundreds of facts, and
  giving each of them an 11% share is impossible, so a usable weighting has to pick out the rows
  that are lookups from memory-only signals. Whether hundreds of such lookups fit at once is the
  capacity question in a new form, and this run does not answer it.
- **6f9b354f is not a weighting problem alone.** ' gray' is moved by no seed, even fitted on its
  own rows. Query drift or the construction is the likelier limit there; the drift diagnostic is
  still to run.

## 2026-10-11 (late morning) — Stage 1 dev run 5: which rows carry the lookup, and how much weight it needs (setup)

**Why.** Branch (b) means building weights from the memory alone. Two things shape what those
weights must look like:
- **Which rows carry the lookup.** The probe rows are three kinds of query:
  - the tool session's last 64 positions (they drive the call decision);
  - the anchor prompt's positions;
  - the gold-predicting rows, the anchor rows whose next token is a gold token (2 tokens for "Luna",
    5 for "a lighter shade of gray").
  If the gold-predicting rows alone carry 'L', fact-level self-study only needs the queries that
  predict each answer. That is cheap and targeted.
- **How much weight they need.** A share of the fit weight that one lookup needs can't be given to
  hundreds of facts at once. So the dose matters for whether the approach can scale.

I also want to know what the fitted bias carries without the fitted values.

**Sets** (all r = 64; sst from the local cache, unchanged since dev run 4).
- `sstpw`: rerun as a check. The fit is deterministic, so it must reproduce dev run 4 exactly.
- Dose on all probe rows: `sst+probe@5` and `sst+probe@100`.
- Row subsets:
  - `sst+anchor@20`;
  - `sst+sess@20`;
  - `sst+gold@100`;
  - `sst+gold@1000`.
- `probe`.
- With `--bias_only`, every fitted seed also runs with the unfitted cluster-mean values
  (S64<ref>-b): centroid keys, fitted bias, mean values.

**Shares of the fit weight on 0862e8bf** (sst fit set 92,838 rows):

| set | share |
|---|---|
| probe@5 | 2.9% |
| probe@20 | 10.7% |
| probe@100 | 37.5% |
| anchor@20 | 3.6% |
| sess@20 | 7.6% |
| gold@100 | 1.3% |
| gold@1000 | 11.4% |

**Reading** (exploratory; descriptive on 'L', with ' lighter' and ' gray' alongside).
- **Rows.** If gold@1000 reaches 'L' ≥ −1 at about sstpw's share, the answer-predicting rows carry
  the lookup, and dev run 6's fact-level self-study taps the answer windows. If only rows that
  include the session do it, the tap has to cover the question prompt as well.
- **Dose.** The smallest share that gets 'L' ≥ −1 sets the per-fact weight a derived scheme must
  reach.
  - If ' lighter' reaches probe level at probe@100, 6f9b354f is weighting too.
  - If it stalls near −3, the limit there is the construction or drift.
- **Bias only.** A seed whose -b variant keeps 'L' carries the lookup through which entries the
  query lands on. Otherwise the fitted values carry it.
- **Interference.** Item-held sst error and p90 at each dose. If held error rises by more than
  0.02 at probe@100, the weight is crowding out the memory's other lookups.

**Smoke** (`results/audit/smoke/dev5_smoke.sh`): one pass on the cached r = 256 smoke set
(6f9b354f, 24,000-token chunks), with refs sstpw, sst+gold@100, sst+sess@20 and probe, plus
--bias_only. sstpw's fit statistics and anchor must equal the dev run 4 smoke's.

**Smoke outcome** (10:23).
- sstpw reproduced the dev run 4 smoke exactly: every head's fit statistics and every anchor token
  are identical.
- The row subsets and the bias-only seeds run.
- At r = 256 on four items, the bias-only seeds lose ' lighter' (sstpw −0.49, its -b −5.85). This
  is a smoke, so I take nothing from it beyond the path working.

## 2026-10-11 (late morning) — fact-level self-study: generator and recall check R1 (setup)

**Change of order.** I run the generator recall check before dev run 5, because it gates the
fact-level line. Dev run 5 refines how the rows are weighted; R1 decides whether there are any
rows to weight.

**Generator** (`experiments/lme_ssf_gen.py`, frozen as of this entry).
- Memory-free and text-only. One prompt per user turn: the user turn, cut into pieces of at most
  400 tokens, after the last 160 tokens of the assistant turn before it.
- The prompt (SS_FACT) asks for the specific facts the user states about themselves in their last
  message. For each fact it wants a question the user could ask later, in the user's voice, and a
  one- to five-word answer copied from the message. At most 4 pairs per excerpt, or NONE.
- Kept pairs have an answer of at most 8 words that is not in the question and appears in the user
  turn (normalised). Dropped pairs are logged with the reason.
- The prompt never contains the session's question, its date or the test templates.
- About 250 prompts per memory (0862e8bf 246, 6f9b354f 260).
- **Design choice, recorded as such:** asking for user-voice fact questions matches the style of
  LongMemEval's single-session-user questions. It is a generic choice for a personal assistant's
  memory. It is not taken from any test question, but it suits this benchmark. Questions about
  what the assistant said would need assistant-turn items.

**Serving.** Generation runs on an SGLang server from the GB10 image:
- the same bf16 weights;
- greedy, bf16 KV and fp32 GDN state;
- native /generate with prompts rendered by the same chat template, thinking off;
- launched only while no HF job holds the GPU.

Its greedy outputs need not match the HF path's bit for bit (bf16 logits, different GDN kernels).
That doesn't matter here: the generator only has to be memory-free and logged.

**R1** (`analysis/lme_ssf_recall.py`).
- Generator run on the evidence sessions of the 9 anchor-scorable dev16 questions (SSU/KU with a
  short gold).
- A question is covered if some kept item's answer contains the gold or is contained in it (at
  least 3 characters, number words matched to digits), and the gold is not in the item's question.
- Evidence labels are used only to choose these sessions and to score the generator, on dev.
- **Gate:** covered on at least 7 of 9. If it fails, I revise the prompt once on dev and log the
  revision before any fit uses generated items.

**Amendment, before any generation** (10:45). A review pass checked each finding against the code.
It was four subagents, orchestrated with Claude Code, each followed by a verifier told to refute;
it is no credential. These changes followed:

- **Generator.**
  - The pair parser now accepts list markers, bold and Q1/A1 labels. A non-NONE output with no
    readable pair is logged as 'unparsed', and the count is printed.
  - Answers lose a trailing full stop.
  - Each file records the server's model path (checked against Qwen3.8-27B) and each prompt's
    finish reason.
  - SS_FACT is frozen as of this amendment and is not edited after R1 results are seen. Its hash is
    in the file tag.
- **R1 scores the items a fit would use.** The generator runs with `--scope all` on the 9 questions'
  present stores, and R1 restricts those items to the evidence sessions. Before, R1 scored a
  separate evidence-only generation.
- **R1 is in-sample for the prompt's wording.** The example categories in SS_FACT (names, numbers,
  times, colours) overlap the dev golds. The extraction is user-turn only, so assistant-stated facts
  and preference questions are outside it.

**Fit-side changes for dev run 6** (`lme_seed.py`).
- **Dose by share, not per-row weight.** At @20 per row, about 600 items would hold about 90% of the
  fit weight; dev run 4's 11% was for one lookup. ssf refs now take a target share of the fit weight
  (`sst+ssfd%30`), and the realised share per head is in the fit statistics.
- **A uniform control at the same share** (`ssfAu`) for the derived weights (`ssfAd`). Before, derived
  versus uniform was confounded with total weight.
- **Tapped rows** (`--ssf_rows`):
  - 'ans': the rows that predict the answer tokens;
  - 'q': every prompt row from the user content on, as well.

  The old `--ssf_ctx 8` tapped only the chat-template tail, with no question token. That was a
  mistake: those 8 rows were 70% of each item's rows.
- **Checks at startup:**
  - exactly one ssf file for every question and store;
  - no repeat-prefill ref alongside ssf (it mutates the cache the ssf pass forks).

  Before, a missing absent-store file would only have failed after the present store's hours of
  fitting.
- **The ssf block logs**, as a dev diagnostic that never feeds the fit:
  - which half of the session split holds each evidence session;
  - row counts by kind and the realised weight shares.
- **The seeded half split puts the test facts on different sides.** 0862e8bf's evidence session
  (27) is in held-out half B; 6f9b354f's (37) is in fitted half A (I checked both on CPU). So on
  0862, ssfA*/ssfAd seeds read
  'L' as transfer to a fact the fit never asked about.
- **Still open, recorded rather than fixed:**
  - sst's queries include 3 user-turn head rows per item, identical across items: about 6 rows per
    item, or about 8% of sst.
  - sst also still uses the test's anchor template (F10).

  Fixing either changes sst and breaks reproduction of dev runs 4–5, so both wait for the
  confirmatory plan.

## 2026-10-11 (late morning) — R1 result: the generator writes an item for every dev gold fact

**Run** (10:49–11:20).
- SGLang server (`code/serve_sglang.sh`), bf16 weights. The server reported model path
  Qwen3.8-27B (the local bf16 snapshot).
- Present stores of the 9 anchor-scorable dev16 questions, plus the absent stores of 0862e8bf and
  6f9b354f.
- About 150 s per store for 235–271 prompts: 260–401 kept items, 9–34 dropped (mostly answers not
  copied from the turn), none unparsed. The first spot check put all 6 completions under 'stop'.
- Files: `results/phase2/ssf/*_all_23c0c6003537.json` (the tag hashes the frozen prompt and
  settings). Log: `results/audit/logs/lme_ssf_gen.log`.

**R1: covered on 9/9** (single-session-user 5/5, knowledge-update 4/4). The gate (≥ 7/9) passes.

| q | gold | matching item |
|---|---|---|
| 0862e8bf | Luna | "What is the name of my cat?" → Luna |
| 6f9b354f | a lighter shade of gray | "What color did I paint my bedroom walls?" → lighter shade of gray |
| 6b168ec8 | three | "How many bikes do I own?" → three |
| 37d43f65 | 16GB | "How much RAM did I upgrade to?" → 16GB |
| 75499fd8 | Golden Retriever | "What kind of dog is Max?" → Golden Retriever |
| 0e4e4c46 | 132 points | "What was my highest score in Ticket to Ride?" → 132 points |
| a1eacc2a | seven | "How many short stories have I completed?" → 7 short stories |
| c7dc5443 | 5-2 | "What is my team's record?" → 5-2 |
| dad224aa | 7:30 am | "What time do I like to wake up on Saturdays?" → 7:30 am |

**How close the generated questions come to the test questions** (whole memory, word Jaccard of
the closest item).
- Two are verbatim:
  - 0862e8bf: "What is the name of my cat?", once;
  - 6b168ec8: "How many bikes do I own?", twice.
- The others range from 0.54 to 0.90.
- So on these questions fact-level self-study mostly anticipates the test question itself. A fit on
  these items tests **anticipatory coverage**, not generalisation to facts nobody asked about. I'll
  report it that way. The held-out half (ssfA*) is what speaks to generalisation.
- **Knowledge-update questions:** the closest item often carries the stale value:
  - 0e4e4c46 "What is my highest score in Ticket to Ride?" → 124 points (gold 132);
  - c7dc5443 → 3-2 (gold 5-2);
  - dad224aa → 8:30 am (gold 7:30 am);
  - a1eacc2a → four (gold seven).

  Items exist for both the old and the new value, and nothing in them says which is current. On
  these questions, fact-level self-study supplies the facts but not the update.

**Dev run 5 launch** (11:37). `results/audit/run_dev5.sh`, with questions in the order 0862e8bf then
6f9b354f. Output `results/phase2/lme_seed_dev5.jsonl`, log `results/audit/logs/lme_seed_dev5.log`,
seeds in the local seed cache (`seeds5/`).

**Smoke for dev run 6, v2** (11:30; `results/audit/smoke/dev6_smoke.sh`).
- Hand-built plumbing file of 8 items; refs sst+ssfd%30, sst+ssfAd%30, sst+ssfAu%30, rows 'q'.
- Teacher forcing, the share scaling, the matched control (ssfAd and ssfAu have the same total
  weight), item scoring and analysis section 4 all ran.
- The realised share is 34.4% for a 30% target. At smoke size the sst fit split is 3 of 5 groups,
  not 80%. The realised share per head is in the fit statistics, so I report that.
- In 'q' mode the answer-predicting rows get only 7% of ssfd's weight: every prompt row carries its
  item's largest delta. Dev run 5's row result decides which rows dev run 6 uses.

## 2026-10-11 (afternoon) — Stage 1 dev run 5 result: the answer-predicting rows carry the lookup, in the fitted values

**Run.**
- 11:37–16:14; 7,805 s for 0862e8bf and 8,430 s for 6f9b354f. Fitting took 7,358 s and 8,041 s for the
  8 sets with bias-only variants.
- Output: `results/phase2/lme_seed_dev5.jsonl`; log `results/audit/logs/lme_seed_dev5.log`. Seeds are in
  the local seed cache (`seeds5/`).

**Reproduction.** sstpw and probe reproduce dev run 4 bit for bit:
- every shared fit statistic in all 64 heads of both questions (the one new key is the weight sum);
- every anchor and decision row of the conditions both runs share.

**Per-token anchor, present store** (log p; share = the extra rows' share of the fit weight,
0862 / 6f9b):

| set | share | 0862 'L' | 6f9b 'a' | ' lighter' | ' gray' |
|---|---|---|---|---|---|
| C | — | −11.15 | −14.32 | −7.36 | −2.86 |
| sst (dev run 4) | 0 | −9.39 | −13.16 | −6.58 | −2.53 |
| sst+probe@5 | 2.9 / 3.0% | −2.81 | −10.07 | −4.26 | −2.49 |
| sstpw (probe@20) | 10.7 / 10.9% | −0.48 | −7.70 | −3.12 | −2.53 |
| sst+probe@100 | 37.5 / 38.0% | −0.15 | −8.52 | −2.07 | −2.63 |
| sst+anchor@20 | 3.6 / 4.0% | −0.32 | −8.96 | −3.50 | −2.43 |
| sst+sess@20 | 7.6 / 7.5% | −3.13 | −9.59 | −4.42 | −2.66 |
| sst+gold@100 | 1.3 / 3.1% | −0.30 | −7.70 | −2.30 | −2.23 |
| sst+gold@1000 | 11.4 / 24.0% | −0.00 | −7.25 | −0.78 | −2.31 |
| probe | — | −0.03 | −6.53 | −0.11 | −2.74 |
| CK* | — | −0.00 | −2.10 | −0.00 | −0.00 |

**Bias only.** Every -b seed loses 'L', the probe-fitted one included: −8.97 to −9.84, against sst's
−9.39 and C's −11.15. ' lighter' is −5.86 to −6.15 (sst −6.58).

**Interference.** Item-held sst error, mean over heads (p90 in brackets):

| set | 0862 | 6f9b |
|---|---|---|
| sst (dev run 4) | 0.351 (0.475) | 0.338 (0.472) |
| sst+probe@100 | 0.357 (0.480) | 0.337 (0.480) |
| sst+gold@1000 | 0.361 (0.482) | 0.344 (0.487) |
| other sets | 0.347–0.352 | 0.336–0.338 |

The largest rise is +0.010, under the 0.02 threshold. The same question's session rows, when they
aren't fitted (anchor@20, gold@100, gold@1000), keep sst's error: 0.438–0.450 against 0.448 on 0862,
and 0.394–0.411 against 0.401 on 6f9b.

**Error on the gold-predicting rows** (mean over heads, 0862 / 6f9b): sst 0.392 / 0.454,
gold@100 0.269 / 0.297, gold@1000 0.184 / 0.230, probe 0.255 / 0.319. The probe-fitted seed recovers
' lighter' (−0.11) at a larger gold-row error than gold@1000 does (−0.78), so the mean error on those
rows doesn't decide the token on its own. One untested reading: fitting the whole anchor prompt keeps
the queries at the gold rows closer to CK*'s.

**Decisions.** As in dev run 4, no seed stops the first call on either question.

**Reading by the logged rule.**
- **Rows:** gold@1000 takes 'L' to −0.00 at 11.4%, about sstpw's share, and gold@100 takes it to
  −0.30 at 1.3%. The answer-predicting rows carry the lookup. The tool session's rows don't carry
  the anchor (sess@20: −3.13 at 7.6%). Dev run 6's fact-level self-study taps the answer windows
  (`--ssf_rows ans`).
- **Dose:**
  - The smallest share tested that gets 'L' ≥ −1 is gold@100: 1.3% of the fit weight, at 100 per row
    on 12 rows per KV head.
  - Spread over all probe rows it takes between 2.9% (−2.81) and 10.7% (−0.48). What matters is the
    weight on the answer rows, not the total share: anchor@20 at 3.6% (−0.32) beats probe@5 at 2.9%.
- **' lighter':** −2.07 at probe@100, neither at probe level (−0.11) nor stalled near −3. On its own
  rows at 24% of the fit (gold@1000) it reaches −0.78. Mostly weighting, then, but it needs a large
  weight on its own rows. ' gray' and the format token 'a' move under no set, the probe fit included:
  not a weighting problem.
- **Bias only:** no -b variant keeps 'L'. The fitted values carry the lookup; the fitted bias with
  cluster-mean values carries nothing measurable.
- **Interference:** no rise above 0.02 at any dose, up to 38% of the fit weight on one question's rows.

**What I take from it** (two questions, one clean token; exploratory).
- The lookup lives in the values the fit gives to the queries that predict the answer. A derived
  scheme has to produce those queries and give them about 100 times an sst row's weight: 1.3% of the
  fit for one fact.
- At that dose, hundreds of facts would outweigh the self-study several times over. Facts in
  different parts of key space may not compete the way one fact competes with the self-study.
  Dev run 6 measures whether they do.

## 2026-10-11 (afternoon) — Stage 1 dev run 6: fact-level self-study as the fit's lookups (setup and reading rule, before the run)

**Why.** This is the first test of the method itself. The reference rows come from the memory alone:
- the generator's questions about facts the user stated, with the session's question never seen;
- the answer windows teacher-forced on CK*;
- weights from the memory's own effect on each answer token, Δ = log p on CK* − log p on C.

Three things are measured:
- how many of the memory's lookups one seed holds at once (capacity, in sample);
- whether fitting half of the facts helps the other half (transfer);
- whether the derived weights beat uniform weights at the same total.

The test anchor (0862 'L') is a secondary readout. The generator wrote the test question nearly
verbatim, so on the anchor this is **anticipatory coverage**, not generalisation (R1 result).

**Sets.** r = 64, present store, sst from the local cache, rows `ans`:
- `sst` (baseline and reproduction gate);
- `sst+ssfd%S`: all items, weights clip(Δ, 0, 10);
- `sst+ssfAd%S`: items from the seeded half of the sessions only, same weights;
- `sst+ssfAu%S`: the same rows as ssfAd at uniform weight, same total;
- each at S = 30% and S = 60% of the fit weight.

**Shares.** My rough estimate of Σ clip(Δ) per store (about 350 items) puts a Δ = 10 answer row
near 50 at 30% and near 150 at 60%, bracketing dev run 5's 100. The realised per-row weights follow
from the logged items and shares; I'll report the weight on the test-matching item's first answer
row next to dev run 5's dose.

**Half membership** (fixed in the amendment): 0862e8bf's evidence session is in the held-out half B;
6f9b354f's is in the seeded half A.

**Absent store** (same script, after the present store): sst, sst+ssfd%30 and sst+ssfd%60. Items
come from the absent memory, which doesn't contain the gold fact. This is a leak check: the seed
can't know the gold, so its anchor must not rise.

**Script:** `results/audit/run_dev6.sh`, questions in the order 0862e8bf then 6f9b354f. Outputs
`results/phase2/lme_seed_dev6.jsonl` and `lme_seed_dev6a.jsonl`; seeds in `seeds6/`.

**Measures.**
- **R:** item-level recovery R = Σ(lp_seed − lp_C) / Σ(lp_CK* − lp_C), with lp summed over answer
  tokens, over items with Δ ≥ 1 ("memory items"). R_A covers memory items from the seeded half,
  R_B the held-out half, R_all all of them.
- **CIs:** each difference below is paired on the same items. I bootstrap it over items (2,000
  resamples, seed 42) for a 95% interval.
- **Held error:** item-held sst error and p90, mean over heads, per question.

**Gates.**
- **G1, reproduction:** sst's fit statistics and anchors equal dev run 4's (bit for bit is expected;
  anchors within 0.01). If not, the run is void.
- **G2, template:** on 0862, the generated items that ask the test question ("name of my cat" →
  Luna) must have CK* log p('L') ≥ −1 and Δ ≥ 5 in the ssf template. If not, the anchor readout is
  void, because the ssf template doesn't carry the lookup.
- **G3, interference:** held error under an ssf set within +0.02 of sst, and p90 within +0.03.
  A set that fails is reported as crowding, and its gains don't count as a pass.

**Readouts, per memory and share.** A call needs both memories to agree; otherwise it is mixed.
- **Capacity:** ΔR_A = R_A(ssfAd) − R_A(sst), with ΔR_all for ssfd the same way.
  - ≥ 0.5 with the CI above 0: the seed holds most of its fitted lookups at once.
  - 0.2–0.5: partial.
  - < 0.2: not at this share.
- **Transfer:** DiD = R_B(ssfAd) − R_B(sst).
  - ≥ +0.1 with the CI above 0: fitting half the facts helps the other half.
  - ≤ −0.1 with the CI below 0: the fit hurts unfitted facts.
  - Otherwise none. I expect none.
- **Weights:** R_A(ssfAd) − R_A(ssfAu).
  - ≥ +0.1 with the CI above 0: the derived weights help.
  - ≤ −0.1: they hurt.
  - Otherwise no difference at this share.
- **Anchor** (secondary; 0862 'L'): f = (lp − (−9.39)) / ((−0.03) − (−9.39)), from dev run 4's sst
  and probe.
  - Under ssfd: f ≥ 0.8 means the anticipated item carries the test lookup across the template
    paraphrase; 0.3–0.8 partial; < 0.3 no.
  - Under ssfAd and ssfAu (evidence held out) 'L' must stay within 1 nat of sst. A larger rise means
    something other than the fact's own rows moves it, which needs explaining before any transfer
    claim.
  - 6f9b ' lighter' and ' gray' (half A) are descriptive.
- **Dose:** for the next stage I take the larger share that passes G3, unless it gains less than
  0.05 in ΔR_all over the smaller one, in which case I take the smaller.
- **Absent store:** the gold anchor under ssfd must stay within 1 nat of sst's on the absent store.
  A larger rise is a leak to find before anything else.

**What would change the plan.**
- If capacity holds, the next stage is end to end: a fresh LongMemEval sample, answer accuracy, the
  seed against the text-hint baseline.
- If capacity fails at both shares with G3 passing, the budget is the limit: N, or the per-head split
  of entries.
- If G3 fails, the shares are too high for the rows to share key space with the self-study.

**Dev run 6 launch** (16:22). `results/audit/run_dev6.sh`, log `results/audit/logs/lme_seed_dev6.log`.
I added the logged measures to `code/analysis/lme_seed.py` before any output existed:
- G2's gold-item line;
- the realised per-row weights in answer-row mode;
- the paired bootstrap intervals;
- per-question held error with realised share (section 5).

Sections 2 and 3 now show the absent store for an absent-only run. I checked the new code on the
dev 6 smoke file and on a copy with a substituted gold. A review of the rule and code runs alongside
the GPU job, and any amendment will be logged before the first item scores exist.

**Amendment to the dev run 6 rule** (16:45, before any item scores or seeded anchors exist; the run had
written only the unseeded conditions, the self-study load and the ssf block, and I hadn't read the
ssf block).
- **Review.** I ran a three-lens agent review (design, analysis code, run path). An adversarial
  check tested each finding against the code and data. Five findings held, two were refuted (below),
  and several minor ones were noted. The run itself needs no change.
- **Session-cluster intervals.** All three lenses raised it. Treatment is by session,
  and items cluster: on 0862, 139 A items come from 13 sessions and 123 B items from 15; on 6f9b, 175
  from 18 and 191 from 16. Up to 21 items share a session, and repeated facts sit within sessions.
  - Every paired difference now gets a 95% bootstrap over sessions (2,000 resamples, seed 42, fresh
    per comparison), plus the range of leave-one-session-out estimates.
  - A call that needs "CI above (below) 0" now needs the session interval to exclude 0, and every
    leave-one-session-out estimate on the call's side of 0.
  - The item-level bootstrap is dropped.
- **Capacity readout.** The capacity call is ΔR_all = R_all(ssfd%S) − R_all(sst), with the logged
  thresholds, and both memories must agree. "If capacity holds" and "if capacity fails" refer to
  this readout only.
  - At one share, ssfAd puts the whole weight on half A's rows: Σ_all clip(Δ) / Σ_A clip(Δ), about
    2×, the per-row dose of ssfd. So ΔR_A(ssfAd) is a dose point (half the facts at about twice the
    weight). It stays the in-sample reference for the transfer and weights contrasts.
  - Added, descriptive only: R_A(ssfd) − R_A(ssfAd), the cost to the A facts of also loading the B
    facts at a lower dose.
- **Anchor under ssfA\*.** 0862's 'L' is in session 27, and every cat item is in session 27 (half
  B). Under ssfAd and ssfAu it is therefore one held-out lookup, and transfer predicts it moves: a
  DiD of +0.1 implies about 1.1 nats. The old 1-nat bound contradicted the transfer readout and sat
  inside refit scatter ('L' already spans −8.62 to −10.07 across fits that never target it). It is
  replaced as follows:
  - Report t_L = (lp − lp_sst) / (lp_CK* − lp_C) beside that set's DiD.
  - Flag 'L' only if t_L exceeds DiD by more than 0.3, or 'L' rises more than 2 nats while DiD's
    interval includes 0.
  - First confirm from the ssf block that 0862's evidence session is in B.
  - Anticipatory coverage is read as f(ssfd%S) − f(ssfAd%S) at the same share, with ssfAu beside
    it, noting ssfAd's higher per-row dose. Only that excess is credited to the anticipated item.
- **Absent store.** The seed's target on the absent store is CK*(absent), and the absent memory alone
  moves 'L' by about 1.3 nats (Stage 0: C −11.23, CK −9.94). So a 1-nat bound against sst could
  fire with no leak. The new rule:
  - **Plumbing decides a leak.** The ssf file is the `_a_` file, no item's answer contains the
    gold, the self-study cache is the absent one, and the entry count is the absent memory's.
  - **Flag for investigation** if 'L' under ssfd exceeds max(sst(a), CK*(a)) by more than 1 nat, or
    'L' > −8. A rise toward CK*(a) inside that envelope is the seed tracking the absent memory.
- **Transfer vs format.** All items are scored in the ssf template, so fitting half A could raise
  B's log p by teaching this template's answer format rather than retrieving B's facts. A transfer
  call now also needs:
  - the first-answer-token DiD (items whose first token has Δ ≥ 1) to show it;
  - the placebo (mean shift on B items with Δ < 1) to be small next to it.

  Even then I'd call it transfer in the ssf template only. A claim beyond that needs the B items
  re-scored in the test template, which this run doesn't do.
- **Capacity failure.** If capacity fails at both shares with G3 passing, the run can't tell an entry
  budget limit from query drift. It doesn't log the seed's error on the ssf rows themselves. So I'll
  report it as "budget or drift", not "budget".
- **6f9b.** The generated answer "lighter shade of gray" tokenises as 'light', 'er', ' shade', ' of',
  ' gray', while the anchor has 'a', ' lighter', …. No ssf row predicts ' lighter', so on 6f9b I
  read ' shade', ' of' and ' gray' (all descriptive). G2's gold-item match now drops function words,
  so 6f9b's item and its per-row weights print.
- **Added, descriptive:** the fraction of memory items whose own recovery is ≥ 0.5, because R is in
  nats and long or repeated answers can carry it.
- **Refuted.**
  - "ssfAd beats ssfAu on R_A by construction": the fit minimises per-head output error, not R, and
    spending weight where the memory adds nothing is the contrast being tested.
  - "Absolute ΔR thresholds are unreachable when R(sst) is high": the evidence was the plumbing
    smoke's verbatim-opening items, not fact items.
- **Code.** `code/analysis/lme_seed.py`:
  - sections 4 and 6 (anchor t per token against this run's sst, and the absent store's plumbing
    lines);
  - a session-cluster `diff()`;
  - an absent-only run in sections 2 and 3.

  Checked on the smoke file, dev run 5 and two altered copies.

**Dev run 6, G2 and the ssf block for 0862e8bf** (read at 16:50, after the amendment).
- **Items.** 262 items (903 answer rows) were teacher-forced in 323 s. Δ is ≥ 1 on 250 items and
  ≥ 5 on 210, so nearly every generated fact needs the memory's attention. Half A holds 130 memory
  items and half B 120. The ssf block confirms the evidence session is in B.
- **G2 passes.** "What is the name of my cat?" → Luna (session 27, B). In the ssf template the CK*
  log p of 'L' is −0.00, and Δ = 11.37 on 'L' and 0.16 on 'una'.
- **Realised dose on 'L' under ssfd:** 22 per row at 30% and 78 at 60%, against 100 for dev run 5's
  gold@100 (−0.30) and 5–20 for the probe-row doses (−2.81 to −0.48). My pre-run guess (50 and 150)
  was high, because Σ clip(Δ) is about 2,950, not 1,400. The shares stay as logged.

**Dev run 6, 0862e8bf read** (19:00; no calls yet, since a call needs both memories).
- **G1 passes.** The sst fit statistics equal dev run 4's on every shared field of all 64 heads. The
  anchors are identical: 'L' −9.39, held error 0.351.
- **G3.** At 30% every set passes: held error +0.009 to +0.014, p90 +0.008 to +0.012. At 60% every
  set fails: ssfd +0.026 (p90 +0.028), ssfAd +0.035 (+0.039), and ssfAu +0.020 (p90 +0.033). So on
  0862, 60% is crowding, and its gains don't count.
- **Readouts at 30%** (session intervals and leave-one-session-out ranges as amended):
  - Capacity: ΔR_all(ssfd) +0.141 [+0.122, +0.160], LOSO +0.136 to +0.144. R_all rises from 0.24 to
    0.38, and the share of memory items recovering at least half their effect from 0.25 to 0.40.
    This is below 0.2, so not at this share. At 60% it is +0.163, which doesn't count.
  - Transfer: DiD −0.007 [−0.022, +0.009]; first token −0.014. None.
  - Weights: +0.027 [+0.013, +0.040]. No difference at this share.
  - Anchor: f(ssfd%30) = 0.13, so no. t_L under ssfAd%30 is +0.02 against a DiD of −0.007, so no
    flag. Anticipatory coverage f(ssfd) − f(ssfAd) is 0.12.
- **Exploratory (not in the rule): the gold item in its own template.** Under ssfd the item "What is
  the name of my cat?" → Luna is in sample. Its rows are in the fit at 22 per row (30%) and 78 per row
  (60%). Its 'L' still only goes from −8.47 (sst) to −7.59 and −7.70; CK* gives −0.00. Dev run 5's
  gold@100 rows were the anchor template's own queries, and they took the anchor's 'L' to −0.30 at a
  1.3% share. Here the item's rows carry about 0.3% of the ssf weight among 903 rows. The fit gives
  them no lookup even in sample, and the extra share at 60% doesn't help. I read this as a sign
  that the rows compete for the entries. That points to the budget over query drift, though this
  run can't separate the two (the amendment's "budget or drift").

## 2026-10-11 (night) — Stage 1 dev run 6 result, present store: the fact rows add about 0.14 of recovery on both memories, short of capacity

The present store finished at 21:44 (both questions, 5 h 22 min). The 0862e8bf read is above. These
are the calls under the amended rule. The absent store started at 21:44 and is still running.

**Gates.**
- **G1 passes on 6f9b354f too.** sst's fit statistics equal dev run 4's on all 3,456 shared fields of
  the 64 heads. Every unseeded and sst anchor log-prob is identical.
- **G3 on 6f9b.**
  - At 30% every set passes: ssfd −0.000 (p90 +0.005), ssfAd +0.003 (+0.011), ssfAu +0.002
    (+0.010).
  - At 60%, ssfd passes (+0.012, p90 +0.026), but ssfAd (+0.021, +0.034) and ssfAu (+0.018, p90
    +0.034) fail.
  - Every 60% set failed on 0862, so 60% is crowding.

**Calls at 30%** (session-cluster 95% intervals; leave-one-session-out ranges):

| readout | 0862e8bf | 6f9b354f | call |
|---|---|---|---|
| Capacity, ΔR_all(ssfd) | +0.141 [+0.122, +0.160], LOSO +0.136 to +0.144 | +0.146 [+0.126, +0.166], LOSO +0.142 to +0.149 | not at this share |
| Transfer, DiD R_B(ssfAd) | −0.007 [−0.022, +0.009] | +0.004 [−0.008, +0.015] | none |
| … first answer token | −0.014 [−0.031, +0.009] | +0.027 [+0.013, +0.041] | |
| Weights, R_A(ssfAd) − R_A(ssfAu) | +0.027 [+0.013, +0.040] | +0.001 [−0.020, +0.024] | no difference |

- **R_all on 6f9b** rises from 0.13 (sst) to 0.27. The share of memory items recovering at least half
  their effect rises from 0.23 to 0.30. On 0862 these were 0.24 to 0.38 and 0.25 to 0.40.
- **60%.** On 0862 it is crowding, so its capacity doesn't count. On 6f9b, ssfd%60 passes G3 and gives
  +0.176 [+0.154, +0.198], still below 0.2.
- **Dose: 30%.** 60% fails G3 on 0862, and on both memories it adds less than 0.05 in ΔR_all over 30%
  (+0.022 and +0.030).
- **Anchor.**
  - 0862: f(ssfd%30) = 0.13, so no. No t_L flag.
  - 6f9b's ' shade', ' of' and ' gray' (descriptive): CK* − C on them is only 0.53, 0.34 and 2.86
    nats, and t under ssfd%30 is −0.00, −0.01 and +0.05.

**What this does to the plan.** Neither branch to the end-to-end stage applies.
- At 30%, capacity fails with G3 passing on both memories. The amendment reads that as budget or
  drift.
- At 60%, the extra weight crowds the self-study rows: in every set on 0862, and in the half-A sets on
  6f9b.

Dev run 7 (next entry) is built to separate budget from drift. The absent store's leak check is
still to come.

**Exploratory (not in the rule).**
- **The test-matching item in its own template, 6f9b.** "What color did I paint my bedroom walls?" →
  "lighter shade of gray" (session 37, half A; Δ 19.76 nats). Its recovery, (lp − lp_C) / (lp_CK* −
  lp_C) over its answer tokens, is:
  - sst −0.03;
  - ssfd%30 0.27, and ssfAd%30 0.37 (in sample at about 27 per row on 'light');
  - ssfd%60 0.44 and ssfAd%60 0.49.

  Its first token 'light' goes from −11.71 under sst to −7.37 under ssfd%30 and −5.19 under
  ssfAd%60. CK* gives −2.78. On 0862 the cat item's recovery goes from 0.26 to 0.34 under ssfd%30.
  On both memories, then, the fit gives a fact it was fitted on less than half of its lookup at every
  dose this run tried.
- **The change of template loses little.** On 6f9b the anchor's 'a' and ' lighter' carry most of the
  gap (12.22 and 7.36 nats), and no ssf row predicts either: the generated answer tokenises as
  'light', 'er'. Even so, their t against sst is:
  - +0.29 and +0.32 under ssfd%30;
  - +0.37 and +0.37 under ssfAd%30;
  - 0.42 to 0.51 at 60%.

  The item's own gain over sst in its template is 0.30 and 0.40 at 30%. On 0862, t_L is 0.11 against
  the item's own 0.08 on 'L'. So on both memories the lookup is lost in the fit, before the
  paraphrase.
- **Calibration on 6f9b.** The anchor-row error stays between 0.328 and 0.343 in every set. The gold
  rows are at 0.454 under sst and 0.356–0.387 under the ssf sets. 'a' and ' lighter' stay at or
  below −6.92 and −3.27. No 6f9b set reaches the ≤ 0.28 band, so this neither supports nor breaks
  0862's split.
- sst alone recovers less on 6f9b (R_all 0.13) than on 0862 (0.24).

## 2026-10-11 (night) — Stage 1 dev run 7: where the fact lookups are lost, and placing the keys where they land (setup and reading rule, before the run)

**Why.** Dev run 6 (above), on both memories:
- Capacity at 30% is +0.141 and +0.146, below the 0.2 bar for partial.
- 60% crowds.
- The test-matching items are in the fit, yet their recovery in their own template is 0.34 (0862's
  cat, 22 per row on 'L') and 0.27 (6f9b's walls) under ssfd%30.
- The test template loses little beyond that.

Two causes remain (the dev run 6 amendment's "budget or drift"):
- **Entries.** k-means places the keys blind to the queries, so a fact's key is averaged with about 64
  others, and only the bias and values are fitted.
- **Query drift.** The fit matches CK*'s queries, while the seeded run produces its own.

This run tests one lever for the first and measures the second.

**Calibration from dev runs 4–6** (0862, r = 64; seed error mean over the 64 heads):
- **gold rows:** the anchor rows that predict 'L' and 'una';
- **anchor rows:** every anchor position.

| seed | gold rows | anchor rows | 'L' |
|---|---|---|---|
| sst | 0.392 | 0.306 | −9.39 |
| sstp | 0.361 | 0.293 | −7.44 |
| sstpw | 0.302 | 0.268 | −0.48 |
| probe | 0.255 | 0.252 | −0.03 |
| sst+probe@5 | 0.326 | 0.278 | −2.81 |
| sst+probe@100 | 0.272 | 0.253 | −0.15 |
| sst+anchor@20 | 0.301 | 0.268 | −0.32 |
| sst+sess@20 | 0.335 | 0.281 | −3.13 |
| sst+gold@100 | 0.269 | 0.279 | −0.30 |
| sst+gold@1000 | 0.184 | 0.276 | −0.00 |
| sst+ssfd%30 | 0.340 | 0.302 | −8.14 |
| sst+ssfAd%30 | 0.364 | 0.309 | −9.22 |
| sst+ssfAu%30 | 0.365 | 0.310 | −8.97 |
| sst+ssfd%60 | 0.334 | 0.314 | −8.01 |
| sst+ssfAd%60 | 0.367 | 0.324 | −9.30 |
| sst+ssfAu%60 | 0.367 | 0.326 | −9.35 |

- **Gold-row error doesn't settle 'L' between 0.30 and 0.34.** At 0.334–0.335, ssfd%60 gives −8.01
  and sess@20 gives −3.13.
- **Anchor-row error splits all 16 sets.** Every set at ≤ 0.281 has 'L' ≥ −3.13, and every set at
  ≥ 0.293 has 'L' ≤ −7.44.
- **The fact-row seeds don't lower the anchor-row error** (0.302–0.326 against sst's 0.306). This is
  descriptive: the anchor's prompt rows are never in an 'ans' fit.

**Sets** (r = 64, present store, both dev questions, rows `ans`, share 30%, which passed G3 on 0862):
- **Loaded from dev run 6** (`seeds6`, k-means keys): sst, sst+ssfd%30, sst+ssfAd%30. Not refitted.
- **Mass keys (`/mw`):** sst/mw, sst+ssfd%30/mw, sst+ssfAd%30/mw.
  - Same reference rows and weights as the k-means seed of the same name.
  - Each memory key is weighted by the attention mass it receives from that seed's fit rows (row
    weights included, val and held rows excluded), with a floor of 1e-6 of the mean.
  - Keys are placed by weighted k-means: initial centroids drawn in proportion to the weights, each
    centroid the weighted mean of its keys. Bias and values are fitted as before.
- **Query-metric mass keys (`/mq`):** sst+ssfd%30/mq.
  - As /mw, but Lloyd runs with distance (k − c)ᵀ M (k − c), where M = Σ w_j q_j q_jᵀ / Σ w_j over
    the fit rows.
  - This is a local optimum of the key-mass-weighted squared logit error on the fit rows. It is a
    factorised stand-in for the attention-weighted error.
  - M is uncentred: about 20–40% of its trace sits on the mean query, an offset the fitted bias
    largely absorbs. A null METRIC therefore tests this metric only.
  - There is no sst/mq. /mq is read against sst/mw.
- **`--drift`:** every seed's answer rows are tapped during item scoring. Per row and head, I log:
  - **e_star:** the seed's relative output error at CK*'s queries, against the memory's output there.
  - **e_run:** the seed at its own run's queries, against the memory at those same queries.
  - **e_tot:** the seed at its own run's queries, against the memory at CK*'s queries.
  - The query drift and the memory-output drift between the two query sets.

  All seven seeds are scored this way. The loaded seeds' items and anchors are recomputed.

**Code** (exploratory). I installed it at 21:46, after dev run 6's absent-store container had
imported its modules (it was loading weights), so that run is unaffected:
- `statelm/kvseed.py`:
  - `kmeans(w=...)`;
  - `full_targets(w=...)` returns the key mass from the same pass;
  - `fit_head(keys="kmeans"|"mass"|"massq")`.

  After the install I reran the CPU check on the installed module. The unweighted paths are
  bit-identical to dev run 6's module on test tensors, the key mass equals a brute-force sum, and
  the entry point lists the new flags.
- `experiments/lme_seed.py`:
  - `--load_seeds` (saved refs are loaded and the rest fitted);
  - `--drift`;
  - `/mw` and `/mq` refs;
  - a guard against `--bias_only` and `--unfitted` with mixed key placement.
- `analysis/lme_seed_keys.py`, with the test-matching item line added at 21:48, before any dev run 7
  output exists.

I ran a three-lens agent review of the drafts and this rule, and an adversarial check tested each
finding against the code. Five findings held, all in this rule or the analysis, and they are folded
in here. None was in the GPU path. Four were refuted.

Script `results/audit/run_dev7.sh` (dev run 6's flags plus `--load_seeds seeds6 --seed_dir
seeds7 --drift`, both in the local seed cache); output `results/phase2/lme_seed_dev7.jsonl`. It launches when dev run 6's
absent store has finished. If that store's plumbing shows a leak, or 'L' is flagged, I'll hold dev
run 7 until it's explained.

**Gates.**
- **G1′, reproduction:** the loaded seeds' anchors and item scores must equal dev run 6's (max
  |diff| = 0). The plumbing must also match dev run 6: the same sst cache (a hit), the same ssf file,
  half and rows, the same item log-probs, and weight sums equal to the k-means counterparts'. If any
  of these fails, the run is void.
- **G3, interference:** the held error of each new seed must be within +0.02 of dev run 6's sst, and
  p90 within +0.03. The held rows are the same sst rows. A seed that fails is reported as crowding,
  and its gains don't count. Dev run 6's k-means ssfd%30 passed G3 on both memories (+0.009 and
  −0.000), so KEYS compares against an uncrowded seed. I'll also print each seed's bias loss, log-mass
  error and ridge histogram to separate a placement effect from a fitting artifact.

**Readouts.** These are per memory; a call needs both memories to agree, otherwise it is mixed.
Intervals are session-cluster bootstraps with leave-one-session-out ranges, as in dev run 6. "Helps"
means ≥ +0.1 with the CI above 0 and every LOSO estimate above 0. "Hurts" means ≤ −0.1 with the CI
below 0 and every LOSO estimate below 0. Anything else is no difference.
- **KEYS (primary):** R_all(ssfd%30/mw) − R_all(ssfd%30), the net effect of placement on the
  method's seed.
- **ENTRIES:** the difference-in-differences [R_all(ssfd%30/mw) − R_all(sst/mw)] − [R_all(ssfd%30) −
  R_all(sst)]. Do the fact rows add more under mass keys? This tests the entries cause. KEYS sst
  (R_all(sst/mw) − R_all(sst)) is reported beside it, since a placement gain on the sst rows alone
  would show up in KEYS too.
- **CAPACITY under mass keys:** R_all(ssfd%30/mw) − R_all(sst/mw), the fact-row increment at fixed
  keys, with dev run 6's thresholds (≥ 0.5 holds, 0.2–0.5 partial, < 0.2 not). The net gain over
  k-means sst is reported beside it, but it is not a capacity figure.
- **METRIC:** R_all(ssfd%30/mq) − R_all(ssfd%30/mw), with the same thresholds.
- **TRANSFER/mw:** DiD R_B(ssfAd%30/mw) − R_B(sst/mw), with dev run 6's thresholds, first-token check
  and placebo. A "hurts" result here means keys placed by half A's facts starve the facts held out of
  the fit (half B).
- **UNSEEN:** R_B(ssfAd%30/mw) − R_B(sst), the net effect on half B against plain k-means. Half B's
  facts were generated items held out of the fit. Facts the generator never covered are not tested
  here.
- **Anchor** (0862 'L'; anticipatory coverage, descriptive): f = (lp + 9.39) / 9.36 under ssfd%30/mw
  and /mq, with gold-row and anchor-row errors.
  - **Prediction from the calibration:** anchor-row error ≤ 0.28 → 'L' ≥ −3.2; ≥ 0.29 → 'L' ≤ −7. A
    miss means the head-mean error isn't the right summary.
  - The gold-row error makes no call between 0.30 and 0.34.
  - 6f9b's 'a' and ' lighter' t against sst are descriptive.
- **Test-matching item in its own template** (descriptive): its recovery and first-token log p under
  every seed. It is in sample under ssfd%30 on both memories, and under ssfAd%30 on 6f9b. If mass keys
  give it most of its lookup while R_all barely moves, entries go to the facts the fit weights most and
  the budget is the limit. If it stays near dev run 6's 0.27–0.37, the fit doesn't hold even the facts
  it is aimed at.
- **Drift reading** (pooled lookup rows, the answer tokens the memory raises by ≥ 1 nat; mean over
  heads). The base for ssfd%30 is sst; the base for /mw and /mq is sst/mw.
  - g = e_star(base) − e_star(seed): the in-sample gain on the fitted rows.
  - g_run = e_tot(base) − e_tot(seed): how much of that gain survives the seeded run's own queries.
  - δ = e_tot − e_star, reported for both the seed and the base: the absolute drift penalty, common to
    every seed.
  - **Budget:** e_star(seed) ≥ 0.8 × e_star(base). The fit removes less than a fifth of the error on
    the rows it was fitted on.
  - **Drift:** g > 0 and g_run < 0.5 g. Less than half the in-sample gain survives.
  - These are assessed independently, so the call is budget, drift, both or neither. The split of
    e_tot into e_run and the memory-output drift, and the per-layer profile, are reported with it.

**What would change the plan.**
- **KEYS helps on both memories:** mass keys become the method's keys.
  - If CAPACITY under mass keys is ≥ 0.2 (or ENTRIES helps), next is N scaling (r = 32, 16) and, if
    capacity reaches 0.5, the end-to-end plan on a fresh LongMemEval sample against the text-hint
    baseline.
  - If KEYS helps only through KEYS sst (ENTRIES no difference), the gain is general key quality, not
    fact capacity, and the entries question stays open.
- **KEYS no difference and the drift call is budget:** the entry count or the seed's form is the
  limit (fitted keys, or more entries).
- **The drift call is drift (or both):** next, a refit on the seeded run's own queries, layer by
  layer. The e_run / memory-output-drift split decides whether its targets are the memory's outputs
  at the seeded queries or at CK*'s.
- **Neither:** the fitted rows' gain is real in sample and survives the run, yet R stays low. I'd
  look first at the prompt rows ('ans' fits leave them out; the calibration's anchor-row result
  points there) and at the ridge.
- **TRANSFER/mw hurts or UNSEEN hurts:** mixed placement (part of the entries by plain k-means)
  before anything else.

**Dev run 6, absent store** (finished 01:50, 14,328 s of compute; read at 01:52). No leak, so dev run 7 goes
ahead.
- **Plumbing passes on both memories.**
  - The ssf files are the `_a_` files, and no generated answer contains the gold.
  - The self-study caches are the absent ones.
  - The entry counts are the absent memories': 1,636 and 1,740 = n_tok(a) // 64.
  - The closest generated questions to the test question are unrelated (word Jaccard 0.60 and
    0.40), so the absent generator wrote no cat or bedroom-wall item.
- **No flag.**
  - 0862's 'L' is −10.78 under ssfd%30 and −10.65 under ssfd%60. That is 1.47 and 1.34 nats *below*
    max(sst(a) −10.72, CK*(a) −9.30), and below −8.
  - On 6f9b no gold token exceeds that envelope by more than +0.02.
  - The absent memory moves 'L' by 1.93 nats (CK*(a) − C(a)), and the seeds don't follow it (sst
    −10.72).
- **Descriptive: capacity repeats on the absent memories.**
  - At 30%: ΔR_all +0.134 [+0.114, +0.155] (0862) and +0.152 [+0.132, +0.173] (6f9b), against
    +0.141 and +0.146 on the present store. G3 passes: +0.002 (p90 +0.008) and +0.005 (+0.008).
  - At 60%: +0.166 and +0.180. G3 passes narrowly on 0862 (+0.018, p90 +0.029) and fails on 6f9b
    (p90 +0.031).

  So the +0.14 is a property of the method on these memories, not of the gold sessions.

**Dev run 7 launch** (01:52). `results/audit/run_dev7.sh`, log `results/audit/logs/lme_seed_dev7.log`.

**Dev run 7, 0862e8bf read** (03:50; no calls yet, since a call needs both memories). The question
took 6,244 s.
- **G1′ passes.**
  - The three loaded k-means seeds reproduce dev run 6's anchors and all item scores exactly (max
    |diff| 0).
  - The plumbing matches: a cache hit on the same sst cache, the same ssf file, half and rows, the
    same item log-probs, and equal weight sums.
- **G3: every mass-key seed lowers the held error.** sst/mw 0.243 (p90 0.319), ssfd%30/mw 0.251,
  ssfAd%30/mw 0.254 and ssfd%30/mq 0.272, against 0.351 (0.475) for k-means sst. The held rows are
  sst items held out of the fit. The key mass comes from the fit rows only, so this is an out-of-sample
  gain in the seed's fit to the memory as a whole.
- **Readouts on 0862.**
  - KEYS: +0.395 [+0.320, +0.457], LOSO +0.379 to +0.409. R_all goes from 0.377 to 0.772.
  - KEYS sst: +0.198 [+0.155, +0.238].
  - ENTRIES DiD: +0.197 [+0.136, +0.250]. The fact rows add more under mass keys.
  - CAPACITY under mass keys: +0.338 [+0.283, +0.387]. NET over k-means sst: +0.536.
  - METRIC (/mq − /mw): −0.115 [−0.164, −0.078], LOSO all below 0.
  - TRANSFER/mw: −0.012 [−0.051, +0.016].
  - UNSEEN: +0.191 [+0.139, +0.231]. This rise in half B's R over k-means sst is placement, not
    transfer.
- **Anchor 'L'.** −0.05 (f 1.00) under sst/mw, −0.01 under ssfd%30/mw, −0.04 under ssfAd%30/mw and
  −0.02 under /mq, against −9.39 for k-means sst. The calibration's prediction holds: the anchor-row
  error is 0.118–0.138, well under 0.28. The test-matching item gets its full lookup in its own
  template under every mass-key seed (item R 1.00).
- **Drift reading: neither, for all three pairs.**
  - Budget doesn't hold: the e_star ratios are 0.77, 0.58 and 0.62.
  - Drift doesn't hold: g_run ≈ g (+0.070 vs +0.076, +0.091 vs +0.084, +0.063 vs +0.075).
  - The absolute drift penalty δ is +0.08 to +0.10 for every seed. It grows with depth: under k-means
    sst it is 0.00 at layer 3 and 0.22 at layer 51.
- **Exploratory: why sst alone now carries 'L'.** I searched sst's 273 generated questions for the
  fact.
  - Five come from the cat chunks (positions 51,840–54,144, inside the evidence span 51,723–54,676).
  - One mentions the cat by name: "How can I help Luna adjust to wearing a new collar?"
  - None asks for the name.

  Under k-means those keys were averaged with about 64 others, and 'L' stayed at −9.39. When the keys
  are placed by the mass the fit rows put on them, the region gets its own entries, and the test
  lookup ("name of my cat") reads them. The question generator was memory-free and saw only the
  chunk, never the test question. Still, a mention by name in a self-study question is a strong form
  of coverage. The absent-store leak check hasn't been run for mass keys.

## 2026-10-12 (early morning) — Stage 1 dev run 7 result: keys placed by attention mass double the recovery on both memories

Dev run 7 finished at 05:42 (13,343 s; 6f9b354f took 7,100 s, 5,305 s of it fitting). I read 6f9b at
05:45 and made the calls under the rule logged before the run. 0862's readouts are in the entry above.

**Gates on 6f9b.**
- **G1′ passes.** The three loaded k-means seeds reproduce dev run 6's anchors and item scores
  exactly (max |diff| 0). The plumbing matches as on 0862.
- **G3 passes for every new seed, by lowering the held error.** sst/mw 0.226 (p90 0.339),
  ssfd%30/mw 0.228, ssfAd%30/mw 0.233 and /mq 0.244, against 0.338 (0.472) for k-means sst. Dev run
  6's k-means ssfd%30 passes G3 on 6f9b (−0.000, p90 +0.005), so KEYS here isn't read against a
  crowded seed.
- **No sign of a fitting artifact.** The rows, weights and steps are the same as the k-means seeds'.
  Under mass keys the bias loss falls from about 0.20 to 0.05, the held log-mass error from 0.44 to
  0.34, and the chosen ridges move smaller. The same fit has an easier job.

**Calls.** 0862 first, then 6f9b, with session-cluster 95% intervals. Every LOSO range sits on the
same side of 0 as its estimate.

| readout | 0862e8bf | 6f9b354f | call |
|---|---|---|---|
| KEYS (primary) | +0.395 [+0.320, +0.457] | +0.395 [+0.345, +0.447] | helps |
| KEYS sst | +0.198 [+0.155, +0.238] | +0.154 [+0.108, +0.207] | helps |
| ENTRIES DiD | +0.197 [+0.136, +0.250] | +0.241 [+0.194, +0.288] | helps |
| CAPACITY under mass keys | +0.338 [+0.283, +0.387] | +0.387 [+0.342, +0.432] | partial |
| METRIC (/mq − /mw) | −0.115 [−0.164, −0.078] | −0.100 [−0.125, −0.077] | not |
| TRANSFER/mw | −0.012 [−0.051, +0.016] | +0.005 [−0.023, +0.033] | none |
| UNSEEN | +0.191 [+0.139, +0.231] | +0.106 [+0.048, +0.173] | helps |
| drift reading | neither | neither | neither |

- **METRIC.** The rule reads it on the capacity thresholds, where anything under 0.2 is "not". Both
  estimates are negative with intervals below 0. On the helps/hurts scale it would be "hurts" on 0862
  and miss the −0.1 bar on 6f9b by 0.0001 (−0.0999), so I don't call it "hurts". I drop the query
  metric.
- **UNSEEN on 6f9b** is +0.1057, LOSO +0.087 to +0.122, so it clears the bar.
- **R_all on 6f9b.**
  - sst: 0.128 → 0.282 under mass keys.
  - ssfd%30: 0.274 → 0.670.
  - The share of memory items recovering at least half goes from 0.30 to 0.67.
- **Aimed and not aimed.** Under ssfAd%30/mw, R_A is 0.780 and 0.782. R_B, the facts held out of the
  fit, is 0.464 and 0.255, about sst/mw's 0.476 and 0.250. A fact the fit is aimed at gets about 0.78;
  a fact it isn't aimed at gets what sst's keys give it. Aiming the fit at half A neither takes
  entries from half B nor carries over to it (TRANSFER none).

**Anchor and test-matching item on 6f9b** (descriptive).
- **The item.** "What color did I paint my bedroom walls?" has word Jaccard 0.78 to the test
  question. It is in half A, so it is in sample under ssfd%30 and ssfAd%30. Its item R:
  - 0.38 under sst/mw;
  - 0.90 under ssfd%30/mw;
  - 1.01 under ssfAd%30/mw;
  - 0.81 under /mq;
  - against 0.27 and 0.37 under k-means.
- **The anchor 'a lighter shade of gray'.** C gives 'a' −14.32 and CK* gives −2.10.
  - In the gap's terms, (lp − C) / (CK* − C), 'a' is at 0.49 under sst/mw, 0.74 under ssfd%30/mw,
    0.84 under ssfAd%30/mw and 0.71 under /mq.
  - ' lighter' is at −0.37 and −0.02 under the two ssf seeds (CK* −0.00).
- **The calibration holds loosely.** The band's prediction was logged for 0862's 'L' and held there.
  On 6f9b every mass-key seed is inside the band (anchor-row error 0.149–0.208). Three reach the
  equivalent of 0862's f ≥ 0.66; sst/mw doesn't (0.49, anchor-row error 0.208). The head-mean error is
  a guide, not a sufficient summary.

**Drift on 6f9b: neither, for all three pairs.**
- **Budget doesn't hold.** The e_star ratios are 0.77, 0.60 and 0.64.
- **Drift doesn't hold.** g_run against g: +0.066 vs +0.077, +0.095 vs +0.087, +0.065 vs +0.078.
- **The common drift penalty.** δ is +0.09 to +0.12 for every seed. Under ssfd%30/mw it is now close
  to e_star itself: 0.098 against 0.129 (0862: 0.083 against 0.116). So about 43% of the remaining
  output error on lookup rows is query drift common to every seed. It grows with depth: under
  ssfd%30/mw at layer 51, e_star is 0.23 and e_tot 0.51.

**What this changes.**
- **Mass keys become the method's keys.** KEYS helps on both memories.
- **ENTRIES helps on both.** Under mass keys the fact rows add 0.20–0.24 more recovery than they did
  under k-means. So the gain isn't general key quality alone: k-means was starving the facts of
  entries.
- **Next by the rule is N scaling (r = 32, 16).** CAPACITY is partial on both memories (0.34, 0.39).
  TRANSFER/mw is none and UNSEEN helps, so mixed placement isn't needed. The end-to-end plan waits
  until capacity reaches 0.5.
- **A caveat on that gate,** written before any dev run 8 output. CAPACITY is an increment over
  sst/mw at the same keys. If more entries lift sst/mw too, the increment can shrink while R_all
  rises. I keep the gate as written and report R_all beside it.
- **Leak check.** The absent-store leak check hasn't been run for mass keys. It goes first in dev run
  8.
- **Scope.** Two questions, generated fact items in the ssf template, one seed each. These are
  development readings, not a result.

## 2026-10-12 (morning) — Stage 1 dev run 8: seed size under mass keys, and the leak check for mass keys (setup and reading rule, before the run)

**Why.**
- **N scaling.** Dev run 7 made mass keys the method's keys, and capacity under them is partial (0.34,
  0.39). Its rule's next step is N scaling to r = 32 and 16.
- **Smaller seeds too.** At fixed r the seed grows linearly with the memory, while the aim is a seed of
  fixed or sublinear size. So I also run r = 128 and 256: how much is lost at half or a quarter of
  the size?
- **Dose.** G3 now has headroom (held error 0.23–0.25 against a 0.35 bar), so I recheck the 60% share
  under mass keys. Dev run 6's dose rule applies.
- **Leak check.** Mass keys haven't had one.

**Seed size.** The size is N × 64 heads × (bf16 keys and values, 256-dim, plus an fp32 bias) =
N × 65,792 bytes. On these memories (T = 104,832 and 111,363):

| r | N | size |
|---|---|---|
| 256 | 409–435 | 27–29 MB |
| 128 | 819–870 | 54–57 MB |
| 64 | 1,638–1,740 | 108–114 MB |
| 32 | 3,276–3,480 | 216–229 MB |
| 16 | 6,552–6,960 | 431–458 MB |

For scale, the full attention KV is 7.3 GB and the GDN state 151 MB.

**Parts.** All four parts run from one script, in order, on both dev questions. They use mass keys,
rows `ans` and seed 42; each writes its own file.
1. **Leak check, absent store** (`lme_seed_dev8a.jsonl`). r = 64; k-means sst loaded from dev run 6a;
   sst/mw and sst+ssfd%30/mw fitted.
2. **Size, present store** (`lme_seed_dev8.jsonl`). Ratios in the order 64, 256, 128, 32, with r = 64
   loaded from dev run 7. Seeds sst/mw, sst+ssfd%30/mw and sst+ssfAd%30/mw, with `--drift`.
3. **Dose, present store** (`lme_seed_dev8d.jsonl`). r = 64; sst/mw and sst+ssfd%30/mw loaded from dev
   run 7, and sst+ssfd%60/mw fitted. With `--drift`.
4. **r = 16, present store** (`lme_seed_dev8r16.jsonl`). The same seeds as part 2. It runs last as the
   longest fit with the largest memory footprint, so an out-of-memory failure there loses nothing else.

The estimate is about 17 hours. Script `results/audit/run_dev8.sh`.

**Code.** `--load_seeds` now fits every ref at a ratio with no saved file and logs a `loaded` block
with no refs. Before, it failed on a missing file. Nothing else changed in the GPU path. The analysis
(`analysis/lme_seed_size.py`) is written while part 1 runs, before any present-store output exists.

**Gates.**
- **G1′ (parts 2 and 3).** The loaded r = 64 seeds must reproduce dev run 7's anchors and item scores
  (max |diff| 0), with the same plumbing. If they don't, that part is void. In part 1 the loaded
  k-means sst must reproduce dev run 6a's.
- **G3 at each r.** The held error of each ssf seed must be within +0.02 of sst/mw's at the same r, and
  p90 within +0.03. A seed that fails is crowding and its gains don't count. The held error of sst/mw
  across r is reported beside it, as the size effect on the fit to the memory as a whole.
- **Leak (part 1).** I use the dev run 6 amendment's rule unchanged.
  - Plumbing decides a leak: the `_a_` ssf file, no generated answer containing the gold, the absent
    self-study cache, and N = n_tok(a) // 64.
  - It flags for investigation if a gold token under either mass-key seed exceeds max(sst(a), CK*(a))
    by more than 1 nat, or 0862's 'L' > −8. Here sst(a) is the k-means seed, so the envelope doesn't
    lean on a mass-key seed.
  - A flag stops the plan until it is explained. Parts 2–4 still run, but I won't act on them first.

**Readouts.** These are per memory, and a call needs both memories to agree. Intervals are
session-cluster bootstraps with LOSO ranges, and "helps", "hurts" and "no difference" are as in dev
run 7.
- **SIZE (primary):** R_all(ssfd%30/mw, r) − R_all(ssfd%30/mw, 64), for r = 256, 128, 32, 16.
- **SIZE sst:** the same for sst/mw.
- **SIZE unseen:** R_B(ssfAd%30/mw, r) − R_B(ssfAd%30/mw, 64). Do more entries help the facts the fit
  isn't aimed at? This decides whether a larger seed buys coverage or only sharpness.
- **CAPACITY(r):** R_all(ssfd%30/mw, r) − R_all(sst/mw, r), with the thresholds ≥ 0.5 holds, 0.2–0.5
  partial and < 0.2 not.
- **TRANSFER/mw(r):** as in dev run 7.
- **DOSE:** R_all(ssfd%60/mw) − R_all(ssfd%30/mw) at r = 64, with CAPACITY at 60%. Under dev run 6's
  rule I take 60% if it passes G3 and gains at least 0.05, otherwise 30%.
- **Descriptive:**
  - anchors and the test-matching items per r;
  - drift per r (e_star, e_tot, δ on the lookup rows);
  - fit seconds per r;
  - on the absent store, whether capacity under mass keys repeats.

**What would change the plan.**
- **A leak flag:** nothing else moves until it is explained.
- **A compact default.** The smallest seed whose R_all(ssfd%30/mw) is within 0.05 of r = 64's on both
  memories, with G3 passing, becomes the default r. This is a design choice from point estimates, not
  a claim. If that is r = 64, the default stays.
- **Larger seeds.** If SIZE at r = 32 helps on both memories, more entries buy recovery, and the
  size–recovery curve is the method's main dial. If SIZE unseen helps as well, they buy coverage of
  facts the fit wasn't aimed at, and that is the case for a larger seed.
- **End to end.** The gate is unchanged from dev run 7: CAPACITY ≥ 0.5 on both memories at some r (or
  at 60%) with G3 passing. If it passes, the next step is the end-to-end plan, at the smallest such
  seed, on a fresh LongMemEval sample against the text-hint baseline.
- **If the gate isn't met.**
  - Under the dev run 7 caveat, CAPACITY can shrink as sst/mw improves. If CAPACITY stays below 0.5
    while R_all(ssfd%30/mw) rises with N, the limit is the fit's increment over the self-study, not
    lookups in general.
  - The levers left are the share (part 3) and the generator's coverage (more items per session).
  - I'd log that reading before choosing between them.

**Dev run 8 launch** (05:49). `results/audit/run_dev8.sh`, log `results/audit/logs/lme_seed_dev8.log`.

**Amendment to the dev run 8 rule** (06:07). Part 1 is on the absent store, and I haven't read any
of its output. No present-store output exists.
- **Review.** I ran a three-lens agent review (rule, run path, analysis), and an adversarial check
  tested each finding against the code, data and this entry. Ten findings held, and two were
  refuted (the r = 16 reading and part 1's reproduction gate, both clarified below anyway). The run
  itself needs no change.
- **The end-to-end gate was not unchanged.** I wrote "unchanged from dev run 7", but I had added
  three things: r = 128 and 256, "the smallest such seed", and a 60% branch.
  - At small N, chunk self-study's incidental lookups can fall faster than the fitted facts'. So
    CAPACITY, an increment over sst/mw, could reach 0.5 because sst/mw collapsed.
  - The gate now: CAPACITY ≥ 0.5 on both memories at the same r, with G3 passing.
  - At r = 128 or 256 it also needs the compact-default criterion below.
  - The 60% branch counts only if DOSE takes 60%.
  - The end-to-end seed is the smallest seed (largest r) that meets all of these.
- **The compact default also needs out-of-sample recovery.** R_all(ssfd%30/mw) is wholly in sample:
  every item's answer rows are in the fit. So it is the readout least likely to fall as N shrinks.
  Besides R_all within 0.05 of r = 64's, the default now needs, on both memories:
  - R_B(ssfAd%30/mw) within 0.05 of r = 64's;
  - SIZE sst not "hurts";
  - G3 passing for both ssf seeds at that r.
- **The dose rationale cited the wrong bar.** The 0.35 I quoted is dev run 7's base, k-means sst. Dev
  run 8's G3 compares against sst/mw at r = 64, with bars of 0.263 and 0.246 (p90 0.349 and 0.369).
  ssfd%30/mw already uses +0.008 and +0.002 of that allowance, about what k-means ssfd%30 used, and
  k-means 60% then failed on 0862. So part 3 is a recheck. It has no new headroom.
- **G3 applies to every call.** A SIZE, SIZE unseen, CAPACITY or TRANSFER call is void on a memory if
  any ssf seed it reads fails G3 there.
- **TRANSFER needs dev run 6's conditions.** The first-token DiD must give the same call, and the
  placebo is read beside it.
- **The leak flag is a gate, not a two-memory call.** A flag on either memory is a flag. If the
  loaded sst(a) doesn't reproduce dev run 6a, part 1 is void and I rerun it.
- **r = 16.** SIZE at r = 16 is descriptive in the "larger seeds" reading, since part 4 may not
  finish. It still counts in the gate.
- **One fit per r.** The guidelines ask for more than one seed behind a phase decision, or a reason
  why one is enough. Before acting on the gate or the compact default, I'll refit r = 64 and the
  chosen r with a second fit seed (43) on both memories. That changes the k-means draw and the
  fit/val/held split. A decision stands only if it holds under both fits. This needs a seed flag
  in `lme_seed.py`, which I'll add after dev run 8 finishes, since parts 2–4 read the script at
  their start.
- **Kill criterion for Stage 1 development.** Suppose the gate isn't met at any r here, nor at the
  share DOSE picks, nor after one run that widens the generator's coverage. Then I stop developing
  on these two questions and write Stage 1 up as it stands. I won't redefine the readout and retry
  under this gate. I'd then decide whether to run the end-to-end comparison anyway, after setting out
  the case for and against.
- **A latent hazard, not in this run.** The seed files dev run 8 saves at r ≠ 64 hold an "unfitted"
  seed built on mass keys, in a field that elsewhere holds k-means. No part of dev run 8 reads it
  (no `--unfitted`, no `--bias_only`). Before a later run loads these files with either flag, I'll
  tag the file's key placement and assert on it.
- **Analysis.** `analysis/lme_seed_size.py` follows this amendment:
  - the leak gate;
  - the base file's r = 64 rows take precedence, and a part that fails G1′ is void;
  - G3 on every call;
  - TRANSFER's first-token condition;
  - the new gate and default;
  - "incomplete" in place of a verdict on partial data.

**Dev run 8, part 1: absent store** (finished 07:55, 7,142 s of compute; read at 07:57). No leak
from mass keys, so the plan goes ahead.
- **Part 1 is valid.**
  - The loaded k-means sst(a) reproduces dev run 6a's anchors and item scores exactly (max |diff| 0)
    on both memories.
  - The plumbing passes. The ssf files are the `_a_` files, no generated answer contains the gold,
    the self-study caches are the absent ones, and N is n_tok(a) // 64 (1,636 and 1,740).
- **No flag on either memory.**
  - On 0862, 'L' is −9.47 under sst/mw and −9.41 under ssfd%30/mw. That is 0.17 and 0.11 nats below
    max(sst(a) −10.72, CK*(a) −9.30), and below −8.
  - On 6f9b, no gold token under either seed exceeds that envelope. The closest is 0.02 nats below.
  - Mass-keyed seeds sit nearer CK*(a) than k-means sst(a) does: 'L' at −9.4 against −10.7. That
    matches the better fit (below). They stay inside what the absent memory's own full KV gives.
- **Descriptive: the mass-key results repeat on the absent memories.**
  - CAPACITY(a), ssfd%30/mw − sst/mw: +0.330 [+0.281, +0.383] (0862) and +0.387 [+0.343, +0.430]
    (6f9b), against +0.338 and +0.387 on the present store.
  - KEYS on sst(a), sst/mw − k-means sst: +0.192 and +0.142, against +0.198 and +0.154.
  - Held error under sst/mw is 0.250 (p90 0.317) and 0.228 (0.341), against 0.357 and 0.334 for k-means
    sst(a). ssfd%30/mw is within G3 of it on both (−0.009 and +0.001; p90 +0.000 and +0.002).

  So the gain from mass keys is a property of the method on these memories, not of the gold sessions.
- Part 2 (seed size, present store) started at 07:55. I'll read G1′ on its loaded r = 64 seeds first.

**Dev run 8, part 2: 0862e8bf G1′** (read at 08:52; no calls yet).
- **G1′ passes.** The three loaded r = 64 mass-key seeds reproduce dev run 7's anchors and item
  scores exactly (max |diff| 0). The plumbing matches: the same sst cache, ssf file, half, rows,
  item log-probs, and weight sums.
- At r = 256 (N 409, 26.9 MB), each seed takes 601 s to fit, against 1,209 s at r = 64. The
  three seeds pass G3 against sst/mw at r = 256 (+0.007 and +0.009; p90 +0.008 and +0.010). The
  held error of sst/mw rises from 0.243 to 0.297 (p90 0.319 to 0.421). That is the size effect on
  the fit to the memory as a whole, and it is still below k-means at r = 64 (0.338).

**Dev run 8, part 2: 0862e8bf read** (11:40; 12,868 s for the question). There are no calls yet,
since a call needs both memories, but two outcomes are already fixed because they need both.
- **G3 passes at every r.** The largest excess is +0.016 (p90 +0.010), for ssfd%30/mw at r = 128.
- **Recovery rises steadily with seed size.** Seeds at r = 256, 128, 64, 32 (27, 54, 108, 216 MB):

  | seed | r = 256 | r = 128 | r = 64 | r = 32 |
  |---|---|---|---|---|
  | R_all, sst/mw | 0.10 | 0.24 | 0.44 | 0.55 |
  | R_all, ssfd%30/mw | 0.33 | 0.58 | 0.77 | 0.86 |
  | R_B, ssfAd%30/mw (facts the fit isn't aimed at) | 0.12 | 0.29 | 0.46 | 0.64 |
  | held error, sst/mw | 0.297 | 0.270 | 0.243 | 0.207 |
  | fit s per seed | 601 | 870 | 1,209 | 2,157 |

  - SIZE against r = 64: −0.44 and −0.20 at r = 256 and 128 (both "hurts"). At r = 32 it is +0.086
    [+0.064, +0.114], which is no difference under the +0.1 bar. SIZE sst (+0.114) and SIZE unseen
    (+0.172) help at r = 32.
  - The fact rows add a roughly constant amount: CAPACITY is +0.24, +0.33, +0.34, +0.31. A larger
    seed lifts what chunk self-study recovers by itself, and the fact rows ride on top.
  - TRANSFER is no difference at every r.
  - Drift stays "neither" at every r, and the drift penalty stays about 0.06–0.12.
  - The cat's name ('L') is lost at r = 256 (−8.5 to −9.4 against C −11.15) and recovered from
    r = 128 on (−0.01 under ssfd%30/mw).
- **Fixed already, whatever 6f9b354f shows.**
  - The compact default stays r = 64. At r = 128, 0862's R_all is 0.195 below r = 64, so it is
    outside the 0.05 margin.
  - The end-to-end gate isn't met at r = 256, 128, 64 or 32. 0862's CAPACITY is below 0.5 at each,
    and the gate needs both memories. What's left is r = 16 (part 4), the 60% share (part 3), and
    then the one generator-coverage run before the kill criterion.
- **What I take from it so far.** The increment from fact rows doesn't grow with N. So the 0.5 bar
  on the increment looks out of reach by size alone, while total recovery keeps rising (0.86 at
  216 MB, 3% of the full KV). I set the gate on the increment before the run, and I'm keeping it. I
  note the tension here, before reading 6f9b354f, so that whatever I do about it later is visibly
  after the fact.

**Dev run 8, part 2 result: seed size** (finished 15:28; 26,675 s of compute; read at 15:31).
- **G1′ passes on 6f9b354f too.** The loaded r = 64 seeds reproduce dev run 7 exactly, and the
  plumbing matches. G3 passes at every r on both memories. The largest excess is +0.016 (p90
  +0.010).
- **Recovery rises with seed size on both memories** (0862e8bf | 6f9b354f):

  | | r = 256 | r = 128 | r = 64 | r = 32 |
  |---|---|---|---|---|
  | MB | 27 \| 29 | 54 \| 57 | 108 \| 114 | 216 \| 229 |
  | R_all, sst/mw | 0.10 \| 0.02 | 0.24 \| 0.14 | 0.44 \| 0.28 | 0.55 \| 0.44 |
  | R_all, ssfd%30/mw | 0.33 \| 0.20 | 0.58 \| 0.40 | 0.77 \| 0.67 | 0.86 \| 0.81 |
  | R_B, ssfAd%30/mw | 0.12 \| 0.03 | 0.29 \| 0.14 | 0.46 \| 0.26 | 0.64 \| 0.40 |
  | held error, sst/mw | 0.297 \| 0.279 | 0.270 \| 0.253 | 0.243 \| 0.226 | 0.207 \| 0.200 |

- **Calls** (both memories must agree):

  | readout | r = 256 | r = 128 | r = 32 |
  |---|---|---|---|
  | SIZE | hurts (−0.44, −0.47) | hurts (−0.20, −0.27) | mixed (+0.086 no difference, +0.141 helps) |
  | SIZE sst | hurts | hurts | helps (+0.114, +0.154) |
  | SIZE unseen | hurts | hurts | helps (+0.172, +0.149) |
  | CAPACITY | mixed (+0.24 partial, +0.18 not) | partial (+0.33, +0.26) | partial (+0.31, +0.37) |
  | TRANSFER | no difference | no difference | no difference |

  - TRANSFER on 6f9b at r = 32 has a positive first-token DiD (+0.065 [+0.014, +0.133]). The
    main DiD is +0.016 [−0.023, +0.068], so the call stays "no difference".
  - Drift is "neither" everywhere, except 6f9b's ssfAd%30/mw at r = 256, which reads "budget"
    (gain ratio 0.83). At r = 256 the seed is short of entries, which fits.
- **The compact default stays r = 64.** At r = 128, R_all is 0.20 and 0.27 below r = 64, far
  outside the 0.05 margin.
- **The end-to-end gate isn't met at r = 256, 128, 64 or 32.** CAPACITY never reaches 0.5. r = 16
  (part 4) and the 60% share (part 3) remain.
- **The anchors follow R.**
  - 0862's 'L' is lost at r = 256 and recovered from r = 128 on.
  - On 6f9b, " lighter shade of gray" is fully recovered at r = 32 under both ssf seeds (−0.00
    for each token). 'a' is −1.48 under ssfAd%30/mw, against CK* −2.10.
  - Under sst/mw alone at r = 32, 'a' is −2.63 and the rest −0.03 to −0.36. That is 0.74–0.98 of
    the gap with no fact rows at all.
- **What this changes.**
  - The size curve is the main Stage 1 result so far. On both memories, total recovery rises
    smoothly with N. The facts the fit isn't aimed at rise fastest with N (R_B 0.03–0.12 at
    r = 256, 0.40–0.64 at r = 32).
  - The fact rows add a fixed +0.2 to +0.4 at every size.
  - So the fixed-size condition isn't met: at a fixed r the seed grows with the memory, and
    each halving of r cuts what is lost (1 − R_all of ssfd%30/mw) by a quarter to a half. At this memory length, 216–229 MB (3% of the 7.3 GB
    KV) gives R_all 0.81–0.86.
  - I'll read part 3 (dose), then part 4 (r = 16), and apply the gate and kill criterion as written.
- **Write-up.**
  - `analysis/lme_seed_size.py --json` writes R per memory, r and seed, with seed MB and sst/mw held
    error, to `results/phase2/lme_seed_size_summary.json`.
  - `analysis/figures.py` draws it as Figure 4 (`images/lme_seed_size.png`), and the three earlier
    figures regenerate byte-identical.
  - REPORT and README carry the size result. Parts 3 and 4 will be added when read.

**Dev run 8, part 3 result: dose under mass keys** (finished 16:56; 4,946 s of compute; read at
16:58).
- **G1′ passes.** The loaded sst/mw and ssfd%30/mw reproduce dev run 7 exactly on both memories, and
  the plumbing matches.
- **60% passes G3 under mass keys**, unlike under k-means: +0.015 (p90 +0.021) on 0862 and +0.011
  (+0.019) on 6f9b.
- **DOSE: no difference** (+0.048 [+0.032, +0.068] and +0.065 [+0.040, +0.088]). Under the
  rule (60% if it passes G3 and gains ≥ 0.05 on both memories), I take 30%. 0862 misses the 0.05 by
  0.002. CAPACITY at 60% is +0.385 and +0.452 ("partial" on both), so the 60% branch of the gate
  fails whatever the dose call.
- **Descriptive.**
  - R_all at 60% is 0.820 and 0.734 (30%: 0.772 and 0.670).
  - 'L' at 60% is −0.00. On 6f9b, " lighter shade of gray" is −0.00 to −0.19, and 'a' is −4.62
    (CK* −2.10).
  - Drift is "neither" on both.
- **Where the gate stands.** The only branch left is r = 16 (part 4, running). CAPACITY at r = 32
  was +0.31 and +0.37, and CAPACITY has been flat in r, so I expect r = 16 to miss too.
- **Plan after part 4, as the rule and amendment set it** (none of it ran; see 2026-10-13).
  1. Apply the `--fit_seed` patch, drafted while part 2 ran: the fit seed reaches the k-means start
     and the fit/val/held split, and saved seeds carry their fit seed and the unfitted seed's key
     placement, with asserts on load.
  2. Refit r = 64 (the compact default) with fit seed 43 on both memories.
  3. If the gate still isn't met, run the one generator-coverage run, designed and logged before it
     runs. Then the kill criterion applies.

## 2026-10-12 (evening) — Stage 1 dev runs 9 and 10: a second fit seed, and the one generator-coverage run (setup and reading rule; never run, dropped on 2026-10-13)

Part 4 of dev run 8 (r = 16) is running, and I haven't seen any of its output. This entry fixes
what runs next for each outcome.

**If r = 16 meets the gate** (CAPACITY ≥ 0.5 on both memories, with G3): dev run 9 refits r = 64
and r = 16 with fit seed 43. If the gate holds there too, I write the end-to-end plan. Dev run 10
doesn't run.

**If r = 16 doesn't meet the gate** (expected: CAPACITY was +0.31 and +0.37 at r = 32), dev runs 9
and 10 run in that order.

**Dev run 9: a second fit seed** (`lme_seed_dev9.jsonl`). The amendment of 06:07 asks for this
before I act on the gate or the compact default.
- `--fit_seed 43` (the patch drafted during part 2, applied after part 4 ends) changes the k-means
  start and the fit/val/held split. The self-study, the ssf halves and the probe tokens keep seed 42.
- r = 64 (the compact default) and r = 32 (the largest r that finished in part 2), present store,
  both memories. Seeds sst/mw, sst+ssfd%30/mw and sst+ssfAd%30/mw, all fitted, nothing loaded, no
  `--drift`.
- **Analysis.** `analysis/lme_seed_size.py --base lme_seed_dev9.jsonl --json`, with the same calls
  as dev run 8 under seed 43, and R per (memory, r, seed) beside seed 42's.
- **What counts.**
  - The compact default stands if R_all(ssfd%30/mw) at r = 64 moves by less than the 0.15 margin
    by which r = 128 failed.
  - The gate's failure stands if CAPACITY at r = 64 and r = 32 stays below 0.5 on at least one
    memory.
  - Each dev run 8 call at r = 64 and 32 (SIZE r32, SIZE sst r32, SIZE unseen r32, CAPACITY,
    TRANSFER) is reported as the same or different under seed 43. A call that differs is reported
    as fit-seed dependent, not as a result.
  - I'll report |ΔR| between fit seeds per (memory, r, seed) as the fit's own noise.

**Dev run 10: the one generator-coverage run** (`lme_seed_dev10.jsonl`). The kill criterion allows
one run that widens the generator's coverage before I stop. Generator entry 2026-10-11 noted that
questions about what the assistant said would need assistant-turn items. In the current files, 22
and 16 of the 50 sessions have no items at all.
- **The change.** `lme_ssf_gen.py --role assistant` writes items from the assistant's turns. Each
  assistant-turn piece (at most 400 tokens), after the last 160 tokens of the user turn before it,
  gets SS_FACT_A. That is SS_FACT reworded to "the specific details that the assistant gives in its
  last message that the user might want to recall later", with a user-voice question and a one- to
  five-word answer copied from the assistant's message. Otherwise it is the same: at most 4
  pairs, greedy, memory-free, the same filters, and never the session's question.
  - That is 357 and 365 prompts (present store). The user path is unchanged: a dry run rebuilds the
    stored prompts of all four user files hash for hash, under the same tag (23c0c6003537).
  - `experiments/lme_ssf_merge.py` joins the user and assistant files into one ssf file per
    memory (`results/phase2/ssf_cov/`). User items come first, in their original order, and every
    item carries its role.
  - Generation runs on the SGLang server after dev run 8 ends and before dev run 9, never beside an
    HF job. Both stores are generated, but only the present store runs here.
- **The fit.** r = 64, present store, fit seed 42.
  - sst/mw is loaded from dev run 7, through a copy of the seed file that holds only sst/mw. A plain
    `--load_seeds` from dev run 7's directory would also load its user-only ssf seeds under the
    same ref names.
  - sst+ssfd%30/mw and sst+ssfAd%30/mw are fitted on the merged items, at the same 30% share. So
    each item gets less weight than in dev run 7.
- **Gates.**
  - G1′: the loaded sst/mw reproduces dev run 7's anchors and its scores on the user items (max
    |diff| 0).
  - Plumbing: the merged file's user items equal dev run 7's items, in order. Their CK* and C
    log-probs match, and the half is the same.
  - G3 as in dev run 8, against sst/mw at r = 64.
- **Readouts** (memory items with Δ ≥ 1, the same bootstrap and thresholds as dev run 8):
  - **CAPACITY_cov (the gate readout):** R_all(ssfd%30/mw) − R_all(sst/mw) over all memory items,
    user and assistant.
  - **COST:** R over the user items under the merged fit minus R under dev run 7's ssfd%30/mw. That
    is what widening costs the facts dev run 7 already held.
  - **ASSISTANT** (descriptive): R over the assistant items under sst/mw and under the merged
    ssfd%30/mw, with their CAPACITY.
  - **TRANSFER_cov:** R_B(ssfAd%30/mw) − R_B(sst/mw) over all items.
- **Prediction, logged before any output.**
  - The gate fails. At 60% the fact rows added only +0.05, so the fit isn't short of weight, and
    more items at the same share should dilute it.
  - I expect COST to be negative, and the assistant items to recover less than the user items.
    Assistant answers are longer and lexically generic.
- **Then the kill criterion.** If CAPACITY_cov is below 0.5 on either memory, I stop developing on
  these two questions. I'll write Stage 1 up as it stands and put the case for and against an
  end-to-end comparison before deciding on it. If it passes on both with G3, dev run 9's check comes next
  for the merged fit, before any end-to-end plan.

**Amendment to the dev runs 9 and 10 rule** (17:30). Part 4's file has only the baseline blocks so
far, no seeded score, and none of dev runs 9 and 10 has run.
- **Review.** I ran a three-lens agent review (run path, analysis, rule) and an adversarial check of
  each finding against the code, data and these notes. Twelve findings held and seven were refuted.
  The seven covered cases that the scripts or logged rules already handle: an sst-set mismatch
  across files, concurrent GPU containers, the stop rule's void and leak cases, and the coverage
  gate's pooled item set, among others.
- **Corrections to earlier entries.** I'm noting these here, not rewriting the entries.
  - Part 2's "the facts the fit isn't aimed at rise fastest with N" is wrong. From r = 256 to 32,
    R_B(ssfAd%30/mw) gains +0.51 and +0.37. R_all(ssfd%30/mw) gains +0.53 and +0.61, and sst/mw
    gains +0.45 and +0.42. They rise with N about as much as the rest.
  - "A roughly constant amount" (part 2) and "CAPACITY has been flat in r" (part 3) overstate it.
    CAPACITY rose from r = 256 to r = 64 (6f9b +0.18 to +0.39, 0862 +0.24 to +0.34), then dipped
    slightly at r = 32. That dip, not flatness, is why I expect r = 16 to miss.
  - In the evening entry, r = 32 is the *smallest* r (the largest seed) that finished in part 2,
    not the largest.
- **Dev run 9: what happens when a seed-42 decision doesn't stand.**
  - `run_dev9.sh` now takes the second ratio as a required argument: 32 if part 4 doesn't meet the
    gate, 16 if it does.
  - **The gate.** Its failure stands if, at each refitted r, CAPACITY under seed 43 is below 0.5 or
    fails G3 on at least one memory. If seed 43 meets the gate at a refitted r, the gate is
    fit-seed dependent. That still counts as "not met", because a pass needs both fits. Dev run 10
    then runs, the kill criterion applies as written, I report the dependence, and I add no
    further fit seeds.
  - **The compact default.** If R_all(ssfd%30/mw) at r = 64 moves by 0.15 or more on either memory,
    the compact default is fit-seed dependent. I'd refit r = 128 with seed 43 before any end-to-end
    plan relied on a compact default. Dev run 10 stays at r = 64 either way.
  - **The command.** From `code/`: `uv run --with numpy python analysis/lme_seed_size.py --base
    ../results/phase2/lme_seed_dev9.jsonl --ref ../results/phase2/lme_seed_size_summary.json --json
    ../results/phase2/lme_seed_dev9_summary.json`.
    - The `--ref` file stays dev run 8's seed-42 summary, Figure 4's input. I'll regenerate it once
      part 4 is in, now with the calls.
    - Section 7 prints R under both seeds, the largest |ΔR| per memory and r, the compact-default
      and gate verdicts above, and each call as "same" or "different (fit-seed dependent)".
    - A missing `--base` now stops the script.
- **Dev run 10: the kill on one fit.** A pass needs two fits, so a miss needs a reason why one is
  enough.
  - The miss stands on one fit if CAPACITY_cov falls short of 0.5 on a memory by more than the sum
    of dev run 9's |ΔR_all| at r = 64 for sst/mw and ssfd%30/mw on that memory. That sum bounds the
    fit noise of the difference.
  - Otherwise I refit the merged seeds with fit seed 43 before applying the kill. sst/mw would come
    from a sst/mw-only copy of dev run 9's seed-43 r = 64 file.
- **The coverage analysis** (`analysis/lme_seed_cov.py`).
  - It prints the placebo beside TRANSFER_cov.
  - It downgrades only a main "helps" or "hurts" that the first-token DiD doesn't confirm, as dev
    run 8 does.
  - A file still being written reads "not scored yet", not "G1′ FAIL".
- **Write-up.**
  - REPORT now says dev run 8's rule was amended after launch. The two part 2 claims are corrected
    there and in README.
  - Figure 4 adds chunk self-study's R_B, the comparator TRANSFER uses, paired by colour with the
    half-A fit's R_B. Its caption gives the item counts and one fit per point.

## 2026-10-13 (late morning) — Stage 1 dev run 8, part 4 (r = 16): one memory finished, and the gate isn't met at r = 16

**What ran.** Part 4 fitted and scored 0862e8bf at r = 16: N = 6,552, 431 MB, 4,349 s of fit per
seed, 14,069 s in all. During that fit the CUDA allocator logged one failed allocation and retry
(2.4 GiB asked, 1.6 GiB free). On 6f9b354f the baseline blocks were written by 21:11. At 21:18 the
kernel's out-of-memory killer stopped the run's python during the r = 16 fit. That was a global
out-of-memory, not a container limit: dgx-spark's CPU and GPU share one memory, and this fit
reached its limit. `run_dev8.sh` stopped there, and "dev run 8 done" was never printed.

**0862e8bf at r = 16** (`results/phase2/lme_seed_dev8_analysis.txt`, which covers parts 1–4).
- G3 passes: the held error is 0.172 for all three seeds, p90 within +0.003 of sst/mw.
- R_all: sst/mw 0.69 (r = 32: 0.55), ssfd%30/mw 0.93 (0.86). R_B of ssfAd%30/mw is 0.74 (0.64).
- SIZE, r = 16 against r = 64, helps on all three: ssfd +0.16 [+0.13, +0.19], sst/mw +0.25
  [+0.21, +0.29], unseen +0.28 [+0.21, +0.32]. LOSO ranges are all above 0.
- CAPACITY is +0.24 [+0.17, +0.31], LOSO [+0.22, +0.27]: partial, and down from +0.31 at r = 32.
  TRANSFER is +0.019 [−0.018, +0.052]: no difference. Drift reads "neither" for both ssf seeds.

**The gate at r = 16 is not met.** It needs CAPACITY ≥ 0.5 on both memories, and 0862e8bf is at
+0.24, so 6f9b354f's r = 16 can't change the call.
- Why one memory is enough: the gate is a conjunction over the two memories.
- Why one fit is enough here: with ssfd%30/mw at 0.93 of the full KV's effect, CAPACITY ≥ 0.5 would
  need sst/mw at about 0.43 or below. It is 0.69, so the fit would have to move it by 0.26. For
  scale, doubling the seed from r = 32 to 16 moved it by 0.14. The planned second-seed refit (dev run 9, never run) covered only
  r = 64 and 32, not at 16.

**Something the gate's arithmetic hides.** CAPACITY is a difference, ssfd%30/mw minus sst/mw, and
the first can't usefully go above about 1. So the room for CAPACITY is about 1 − R_all(sst/mw).
That room was 0.56 and 0.72 at r = 64, 0.45 and 0.56 at r = 32, and is 0.31 on 0862e8bf at r = 16.
As the seed grows, chunk self-study alone recovers more of the effect, and "the fact rows add half
of it" stops being possible. At r = 16 the gate is out of reach by construction. At r ≥ 64 it was
in reach and missed (+0.34 and +0.39 at r = 64). I'm not changing the gate. I'll state this in the
write-up and in the case I set out for the end-to-end decision: the bar asks whether fact rows add
a lot, while at large seeds the plain seed is already most of the way there.

**Not rerunning 6f9b354f at r = 16 now.** It decides nothing: the r = 16 gate is settled by
0862e8bf. The r = 16 SIZE readouts are 0862e8bf only, and I'll report them that way. A rerun would
take about four hours and would need the fit's memory brought down first, since the same fit
stopped the machine. Figure 4's 6f9b354f panel ends at r = 32.

**Next, as logged on 2026-10-12 (evening).** r = 16 didn't meet the gate, so: the fit-seed patch
first, then assistant-item generation for dev run 10, then dev runs 9 and 10,
one at a time. Before generation I'll check free memory, and
I'll watch it during each run.

## 2026-10-13 (afternoon) — Stage 1 stopped; a review of every gate against the question

**I stopped Stage 1 here** (12:17) to review every gate against the question: the gate designs kept turning out badly.
Generation for dev run 10 (`run_gen10.sh`) was stopped partway. It
had written three of its four files to `results/phase2/ssf_asst`: assistant items for 0862e8bf on
both stores (1,014 and 1,018 items) and for 6f9b354f on the present store (1,101). The 6f9b354f
absent store stopped at 320 of 366 prompts, and the merge never ran. Dev runs 9 and 10 have not run.
The `--fit_seed` patch is in `code/experiments/lme_seed.py` and compiles, but has never run. Nothing
is on the GPU.

**The review.** I ran an agent review, orchestrated with Claude Code, of LABNOTES, the plans, the
report and the results. Four auditors split LABNOTES, one pinned down the question from the plans,
one took stock of the project and one read the literature review. A critic then read all of their
output. What follows is my reading
of it, checked against these notes.
- **Gates.** The auditors classed 93 rules. The validity checks did their job: the injection suite,
  the kvseed checks, the reproduction gates and the leak envelopes. Nearly every rule that decided
  what to do next had a flaw. The most common were a proxy measure standing in for the question
  (30) and a threshold with no derivation (27). Next came decisions resting on a two-memory
  conjunction (21) and bounds set by a ceiling or floor (16).
- **CAPACITY was the clearest case, and I'd seen it coming.** Its room is about 1 − R_all(sst/mw).
  A reviewer raised that before dev run 6's data (the 'Refuted' list in the dev run 6 amendment), and I overruled it on the wrong grounds.
  I named the mechanism again before dev run 8 (the caveat on the gate in the dev run 7 result) and kept the gate.
- **Correction to this morning's entry.** On 0862e8bf the bar was out of reach at r = 32 as well as
  at r = 16: a pass needed R_all of 1.05 and 1.19. At r = 64 it needed 0.93. Under mass keys, dev
  run 8 had almost no live path to a pass.
- **Correction to this morning's write-up.** README said the fact rows add "at most +0.39". At a 60%
  share they reached +0.452 on 6f9b354f. I've fixed that line and the REPORT's room sentence.
- **Stale claims, not yet fixed.**
  - README calls the seed "a fixed set of entries", but it has N = T/r.
  - README and REPORT say the seed carries an answer "only when fitted in sample on the test
    question's queries". Memory-only mass-key seeds reach 'L' −0.05 (dev run 7, 0862e8bf read).
  - REPORT's "~58%" for the model's own answer predates the sealed 2AFC of 0.54.
  - REPORT's decoder range of 0.95–1.00 has a minimum of 0.932 in the interference table (other, K = 32).
- **Against the question.** The end-to-end comparison was never designed. It appears only as a
  one-line promised next stage behind proxy gates (the 'What would change the plan' lists of dev runs 6, 7 and 8, and the dev runs 9-10 plan). Stage 1 measured
  fidelity to the raw KV (R against CK*) on two questions at one length, about 43 h of run time. Dev
  runs 9 and 10 can't bear on the question even if they pass.
  - On LongMemEval_S at ~108k tokens, the agent's first search already finds the evidence (hit@3
    0.94–1.00). So that setting leaves little room for any hint, text or seed.
  - The seed is fitted against the full KV, so it can't be built beyond the context window. That is
    where the thesis expects it to win.

I dropped dev runs 9 and 10: they can't bear on the question.

## 2026-10-13 (afternoon) — room check: infrastructure smoke (setup, before any GPU minute)

Before building any more seeds, I run a room check. It asks whether fresh LongMemEval
questions have a memory size and tool budget where a text-hint agent falls short of the evidence,
and where the whole memory in context (the most a perfect seed could give) beats that agent by a
detectable margin. The design is in review and gets its own entry before any data run.

This smoke touches no question. It asks whether SGLang on dgx-spark can serve Qwen3.8-27B with a
262,144-token context and prefix caching. The M arm needs both: at 200k tokens without caching,
every agent step would re-read the whole memory. The steps:
- Read the image's SGLang options.
- Start a long-context profile of `serve_sglang.sh`: radix cache on, context 262,144, a larger
  static memory fraction, fewer running requests.
- Time one prefill of about 200k tokens of filler text.
- Send the same prefix again with a different suffix, and check that it is a cache hit (time and
  `cached_tokens`) and that greedy output on a short prompt matches with the cache cold and warm.
- Watch free memory throughout. If MemAvailable drops below 8 GiB, I stop the server.

**Smoke result** (21:22–21:35). `serve_sglang.sh start-long` came up in about 7 minutes:
- **Pools.** The full-attention KV pool is 231,183 tokens (7.06 GB each for K and V). The GDN-state
  cache holds 86 states in 12.2 GB. After the pools, 30 GB of GPU memory and 26–28 GiB of host
  MemAvailable were left.
- **Prefill.** One 185,377-token prompt (two H1 histories) took 244 s, about 760 tokens/s.
- **Cache reuse.** The same prefix with a different question was served from the radix cache:
  185,344 tokens cached, 16.5 s in all, mostly decoding.
- **Determinism.** Greedy output on a short prompt was identical with the cache cold and warm.
- **Decode.** 4.5 tokens/s for a single stream.

What this means for the design:
- The M arm at 200k plus up to 8 file reads (up to ~48k tokens) doesn't fit a 231k pool. Either
  the GDN-state cache shrinks to make room, or the reads or the large size do.
- Prefill at ~760 tokens/s puts the M arm's memory prefills at roughly 10 h for 120 questions
  across the three sizes, before any agent step. The review is costing the alternatives.

## 2026-10-13 (night) — room check: design and decision rule (before any question is run)

**Review of the draft.** I ran a four-lens agent review (rule, validity, feasibility, skeptic),
orchestrated with Claude Code, and checked each finding myself. The draft had these problems:
- **Budget-capped agents.** It compared the whole memory in context (M) with agents capped at 0–2
  tool calls. Of the fresh pool, 70% of questions need two or more evidence sessions, so any gap at
  a small budget comes from the cap, not from hints falling short.
- **Wrong reference point.** It measured "a realistic seed gets half the gap" from the text-hint
  agent. A seed starts from the no-hint agent.
- **Impossible worked examples.** Some broke the bound that the discordance q is at least |G|.
- **A strawman hint.** The text hint was only 8 TF-IDF words per file.
- **Invisible filler.** The 200k filler was screened so hard that search could never stumble on
  it.
- **Oracle dates.** The oracle file's dates differ from S's.
- **Undeclared noise.** Serving noise wasn't measured.
- **Cache-breaking prompts.** Prompts were re-rendered as text each turn, which defeats the prefix
  cache.
All of these are fixed below.

**Data.** LongMemEval_M (cleaned, same revision as S) is now on disk. For the 120 calibration
questions, its evidence sessions and their text are the same as S's. But every date differs, and so
does the question date, and S's sessions aren't a subset of M's. So I build every store from M, with
M's dates and question date.

**Questions.** I drew 120 calibration questions with `random.Random(42)`, stratified by type:
11 SSU, 32 MS, 42 TR, 16 KU and 19 SSA. They come from the 362 fresh questions outside H1, smoke,
plumbing, dev16 and `lme.EXCLUDE`. The other 242 are held out, and the run script refuses them. No
calibration id has an answer-error issue on LongMemEval's tracker (checked 2026-10-13). Seven
held-out ids do (370a8ff4, gpt4_88806d6e, eac54add, 9aaed6a3, gpt4_5dcc0aab, gpt4_f420262d, and one
duplicate-filler report). Those are dropped from any later test, which leaves n_test = 235.

**Stores** (`statelm/room.py`, built on CPU before the server starts). For each question I take its
M haystack and fix one random order of its non-evidence sessions with
`random.Random("room-" + qid)`. Each size keeps every evidence session plus the first k sessions of
that order, sorted by M's dates.
- O: evidence only.
- small: about 25k tokens.
- mid: about 110k tokens, the size of S.
- large: about 200k tokens. That leaves room for 8 reads inside the 262k context.
The stores are nested, so distractor load is the only thing that changes between sizes. The
distractors are the authors' own, so I add no re-dating and no cross-user screening.

**Arms.** SGLang, greedy, thinking off. Agents stop on their own, with a cap of 8 tool calls.
- N: no memory, no tools.
- A: memory tools, no hint. This is the floor a seeded agent starts from.
- T: tools plus the keyword index (`lme.keyword_index`, k = 8).
- D: tools plus a one-line note per file. The same model writes the notes greedily and
  question-blind, at most 25 words each, cached by session. This is the MEMORY.md-style hint an
  ordinary agent would have.
- M: the store in context in H1's format (the `compile_text` prefix), plus tools. This is the
  ceiling for a perfect seed: the seed is fitted to reproduce that prefix.
- O: M on the evidence-only store.

One call is executed per assistant turn. After the 8th result, or before a context overflow, a
user turn asks for the answer. A turn cut off inside a call counts as no answer. Search returns the
top 10 lines by BM25, with digits kept, each line tagged with its role, and a 300-character window
around the best term. Reads are capped at 24k characters.

**Scoring.** The judge uses LongMemEval's own answer-check prompts (verbatim from
`evaluate_qa.py`) on Qwen3.8-27B, greedy. A second judge, Claude agents blind to arm with items
shuffled, labels every M-vs-best-baseline discordant pair at mid and large, plus a random 20% of
concordant pairs. G uses the second judge's labels where it has them, and I report agreement by
arm.

**Noise first.** I rerun T and M at small on 40 questions in a different order, and the label flip
rate is added to q.

**Readouts** (paired, bootstrap over questions, 2,000 resamples, seed 42). For each size: the
accuracy of every arm, calls and tokens read, the by-type breakdown, and the room for any hint
O − B*.
- B* is the stronger of T and D at that size, chosen on calibration. That choice favours the
  baseline, which is conservative for the thesis.
- G = acc_M − acc_B*.
- q = max(discordance of M vs B*, discordance of A vs B*) + flip rate.
- D_min = sqrt(7.84·q / (235 + 7.84)): the smallest difference a 235-question paired test detects
  with 80% power at two-sided 0.05. It is about 0.08 at q = 0.2 and 0.10 at q = 0.3.

**Decision**, at each thesis size s ∈ {mid, large}. Small is descriptive only.
- **NO-GO(s):** G's 95% upper bound < D_min. Even a perfect seed can't beat the best text-hint
  agent by a margin we could detect.
- **GO(s):** G's 95% lower bound > 0 and G ≥ D_min. I also report
  f* = (acc_B* − acc_A + D_min) / (acc_M − acc_A), the share of the memory's accuracy effect over
  the no-hint agent that a seed must deliver to win detectably.
- **INCONCLUSIVE(s):** anything else.
- **Overall:** NO-GO if both sizes are NO-GO, GO if either is GO, otherwise INCONCLUSIVE.
- **Actions.** NO-GO: stop seed work on in-window LongMemEval. GO: first measure a real seed's f on
  a small subsample. CK* plus tools against M on about 20 questions comes first, then the seed, and
  only then any full test, with the seed-build cost stated. INCONCLUSIVE: report it, and I choose
  the next step then.
- **What it licenses.** GO is necessary, not sufficient. The seed's ceiling is CK*, not M:
  positions are collapsed and the floor C sits below A. NO-GO is the strong outcome.

**Checks on the rule.**
1. **Thesis link.** NO-GO means no seed of this construction can win here. GO means a perfect seed
   could, and a real one needs f*.
2. **Room.** D_min is about 0.08–0.10. G is bounded by acc_M − acc_B*, so the bar is reachable
   unless B* is above about 0.9, and a cell that high genuinely has no room. D_min is derived from
   q, so q ≥ |G| holds.
3. **Branches.**
   - Search finds the evidence (as in H1): B*(mid) ≈ 0.65 and M(mid) ≈ 0.55, so G < 0. Large reads
     the same way, so NO-GO.
   - The thesis world: B*(large) 0.45 and M(large) 0.60, so G = 0.15 and GO.
   - G = 0.05 ± 0.08 at both sizes gives INCONCLUSIVE.
   - Every branch is reachable, and each leads to a different action.
4. **Noise.** Question sampling (the bootstrap), judge disagreement (the second judge) and serving
   flips (the rerun) are all measured, not assumed.
5. **Conjunction.** NO-GO needs both sizes, so the conjunction leans toward continuing. GO needs
   one of two sizes, and a GO is followed by an independent seed test on held-out questions.

**Staging, to save GPU.** Stage 1 runs digests, N, O, A, T and D at all sizes, M at small, and the
flip reruns. If, at size s, the upper bound of O − B* is below D_min and acc_M(small) ≤
acc_O + 0.05, M can't beat B* at s (since G ≤ O − B* when M ≤ O). Then s is NO-GO without running
M@s. Otherwise stage 2 runs M@s, one size at a time and one large question at a time.

**Cost estimate.** Stage 1 is about 10–12 h. Stage 2 is up to about 25 h: M's 110k and 200k
prefills run at about 760 tokens/s. Stop rule: after 10 questions of M@large, if the projected
total for M@large exceeds 24 h, I stop and re-plan.

**Safety.** Data are prepared on CPU before the server starts, since loading M alone takes 24 GB.
The client exits if MemAvailable drops below 8 GiB. No HF job runs while the server is up.

**Code review, and amendments before any question runs** (22:15). I ran a three-lens agent review
of the code (agent loop, analysis against this rule, data and leaks), orchestrated with Claude Code.
Each finding came with a CPU test, and I checked each one.
- **Fixed in the agent runner** (`room_agent.py`):
  - **Overflow never produced an answer.** The overflow check made the forced answer impossible
    to send, so every overflow scored wrong. Room for a forced answer is now reserved before every
    turn, and tool results are capped at 9,000 tokens.
  - **Context limit.** The limit is now read from the server. The pool allows about 231k tokens,
    not the 262k of the context window, and an over-length 400 from the server is handled as an
    overflow.
  - **Watchdog.** The memory watchdog didn't stop the run, and worker errors stayed silent. Now
    any worker error, or MemAvailable below 8 GiB, stops the run at once (it resumes later).
  - **M concurrency.** M at 16 workers would have evicted each question's cached prompt. M is now
    capped at 6 workers at small, 2 at mid and 1 at large.
  - **Smaller fixes.** Malformed calls are flagged. Arm D checks that every session it needs has a
    note.
- **Fixed in the tools and data** (`room.py`):
  - **Search crash.** Any query containing "user" or "assistant" crashed search. Terms now come
    from the text after the role tag, and the window lookup is guarded.
  - **Stemming.** Search now uses light stemming (`lme.stem`), so "recommended" finds
    "recommend". This makes the text baseline a little stronger, which is the conservative
    direction for the thesis.
  - **A literal control token.** One distractor session (`sharegpt_hKWIYBm_27`) contains the
    literal string `<|endoftext|>`. Special-token strings in memory text, in tool results and in
    notes are now defused, so they encode as text. Its digest prompt went out un-defused, which
    affects one note.
- **Fixed in the analysis** (`analysis/room_check.py`):
  - **Labels.** Second-judge labels must be JSON booleans and must be in the key.
  - **B\*.** It is chosen on Qwen's labels, recorded in the key, and checked again on reload, so a
    stricter second judge can't flip the comparison.
  - **Completeness.** A size is decided only when all 120 questions are in, the noise reruns exist
    and the second judge has covered both arms. Until then the verdict prints as provisional.
  - **Reporting.** Judge agreement is printed by arm. GO is returned as soon as either size is GO.
- **Amendments to the rule** (before any question runs):
  1. **Second judge.** It labels every answered response of the two arms that enter each decision
     (X and B\* at mid and large; about 480 responses), not discordant pairs plus 20% of the rest.
     That is a superset of the logged rule, and it removes the bias of an unweighted 20% sample.
  2. **Skip rule.** It becomes UB(O − B\*) + max(0, acc_M(small) − acc_O) < D_min. As logged, the
     +0.05 slack broke the rule's own logic: it allowed G to exceed O − B\*.
  3. **Test exclusions.** I listed the held-out exclusions wrongly. There are six ids with
     answer-error issues; the duplicate-filler issue is not an answer problem. The seventh
     exclusion is 6456829e: H1's smoke actually ran it on 2026-10-08, and `lme.sample`'s smoke draw
     differs from the ids that ran. n_test stays 235. 0977f2af, the other id H1's smoke ran, is in
     calibration, which is harmless.

## 2026-10-14 (afternoon) — room check: the file notes are written; the smoke starts nine hours late

- **Notes.** Arm D's notes are all written: 9,392 of 9,392, one per session in the calibration
  stores, in 19,728 s (5.5 h) on the generation profile, finishing at 03:20. None is empty and
  243 (2.6%) hit the 48-token cap and were cut to their first line. The median note is 17 words.
- **Lost time.** I chained the smoke to the notes with a waiter that polled
  `pgrep -f run_room_digest.sh`. That pattern also matched the waiter's own command line, so the
  waiter never saw the script exit, and the GPU sat idle from 03:20 until I checked at 12:39. The
  server had stopped cleanly. From now on I chain steps in one background command (`a; b`), or
  wait on a PID, never on a name match.
- **Smoke.** It started at 12:40 on the long-context profile, on the first 2 calibration
  questions: every arm at small, plus A, T and D at large, then the Qwen judge on those records.
  Nothing from it enters the decision. It checks plumbing only.
- **Smoke result (12:40–12:50).** It ran 18 records with no errors, and the server reported a max
  input of 237,168 tokens. Both questions are easy: every arm except N answered both correctly at
  small, and so did A, T and D at large. Qwen's labels agree with my reading of all 18. The
  plumbing checks out:
  - **Tool calls** parse and execute. The model often emits two searches in one turn; the second
    is dropped (as designed) and it reissues it on its next turn, so the call cap is what binds,
    not the turn count.
  - **Cache.** From the second turn on, each turn reuses almost the whole prefix (e.g. 24,064 of
    24,808 tokens for M at small).
  - **Index sizes.** The D index adds about 540 tokens at small and 4.6k at large (87–96 files).
    The T index adds 0.3k at small and 3.2–3.6k at large.
  - **Memory.** MemAvailable stayed at 25–26 GiB with the server up.
  - **Not exercised.** The smoke never hit the forced, overflow or malformed paths. M at mid and
    large is stage 2.
- **Stage 1** started at 12:51 (`run_room_stage1.sh full`): N, O; A, T, D at small, mid and large;
  M at small; then the flip reruns and the judge.

## 2026-10-14 (evening) — room check stage 1: agents with any text hint sit near 0.72 at every size; the evidence alone in context gets 0.93

- **Run.** 12:51–17:22 (4.5 h), all 1,520 records written, no worker error. The server reported a
  max input of 242,385 tokens this time; it varies with memory at start (231k, 237k, 242k so far).
  Then 1,274 distinct answers were judged by Qwen. These are Qwen's labels only; the verdict waits
  for M at mid and large and for the second judge.
- **Accuracy** (120 calibration questions; types SSU 11, SSA 19, MS 32, TR 42, KU 16):

  | arm | small (~25k) | mid (~110k) | large (~200k) |
  |---|---|---|---|
  | N (no memory, no tools) | 0.050 | | |
  | O (evidence only, in context, + tools) | 0.925 | | |
  | M (whole memory in context, + tools) | 0.908 | stage 2 | stage 2 |
  | A (tools, no hint) | 0.750 | 0.717 | 0.692 |
  | T (tools + keyword index) | 0.733 | 0.717 | 0.683 |
  | D (tools + one-line note per file) | 0.742 | 0.700 | 0.733 |

- **Noise.** The flip reruns gave 0 of 80 changed labels (T and M at small, reverse order), so
  greedy decoding here is deterministic under batching, and q is the discordance alone.
- **Text hints add nothing over the bare agent.** B\* is D at small, T at mid and D at large.
  B\* − A is −0.008, 0.000 and +0.042. This is the H1 result again on fresh questions and three
  store sizes.
- **Agent accuracy barely moves with store size.** A drops 6 points from 25k to 200k (8× the
  memory), and T and D drop 0–5. BM25 retrieval holds up on these stores (prep: an evidence file
  in the top 3 for 0.92–0.97 of questions), so the agent's limit is not store size in this range.
- **Room for any hint.** O − B\* is +0.183 [+0.108, +0.267] at small, +0.208 [+0.142, +0.283] at
  mid and +0.192 [+0.117, +0.275] at large. Nearly all of it is multi-session: agents score
  0.44–0.59 there, against 0.94 for O and 0.91 for M at small. At small, M − B\* is +0.167
  [+0.083, +0.250] (descriptive; small is not a thesis size).
- **Why the agents fail** (descriptive; I replayed every A/T/D call through the store; "saw" means
  an evidence file's name appeared in a search result or the file was read, which is generous):
  - About 82% of the runs that saw every evidence file were correct, and 11–30% of the rest.
  - Of the wrong answers, about half had seen every evidence file (54/93 at small, 50/104 at
    mid, 53/107 at large) and half had missed one or more. A quarter were forced at the 8-call
    cap.
  - On multi-session questions, runs that saw every file were right 44–52 times in 68–75. Runs
    that missed a file were right 1–2 times in 21–28.
  - So the gap is part finding (missing an evidence session) and part using: snippets of 300
    characters don't carry every needed fact, and agents stop reading early. Having the whole
    memory in context removes both.
- **Skip rule.** It did not fire at mid or large: UB(O − B\*) is +0.28, far above D_min (0.08 at
  mid, q 0.208; 0.09 at large, q 0.242). Stage 2 (M at mid with 2 workers, then M at large with 1)
  started at 17:25 (`results/audit/run_room_stage2.sh`). Every M prompt fits the server: the
  largest at large is 201,337 tokens, 211k with the answer and tool-result reserve. The stop rule
  is in the script: after the first 10 M@large questions it projects the stage-2 total and exits if
  that exceeds 24 h.
- **What decides it now.** The verdict hinges on whether whole-memory-in-context holds near its
  0.91 at 110k and 200k. If it does, G is near +0.18 at both sizes and the check is GO. If it
  falls to about the agents' 0.72, there is no room left for any seed to show.

## 2026-10-15 (morning) — room check stage 2 done; second judge setup (before any label)

- **Stage 2 run.** 17:25–08:09 (14.7 h), all 240 M records written, no error, server max input
  235,549. The stop rule didn't fire: after 10 M@large questions (mean 286 s each) the projected
  total was 14.3 h. Qwen then judged the 223 new answers.
- **Provisional readout** (Qwen labels; the verdict waits for the second judge):

  | | M | B\* | G = M − B\* [95%] | q | D_min | provisional |
  |---|---|---|---|---|---|---|
  | mid | 0.875 | T 0.717 | +0.158 [+0.083, +0.233] | 0.208 | 0.082 | GO, f\* 0.52 |
  | large | 0.800 | D 0.733 | +0.067 [−0.008, +0.150] | 0.242 | 0.088 | INCONCLUSIVE |

- **M degrades with length.** Whole memory in context goes 0.908 → 0.875 → 0.800 from 25k to
  200k. Multi-session goes 0.91 → 0.78 → 0.69, and KU goes 0.88 → 1.00 → 0.69 (n 16). The agents
  lose only 0–6 points over the same range.
- **Second judge, how the items reach it.** This is a change of method from the logged plan. I
  logged it as "items given inline". Passing 480 responses (300 KB) inline would mean re-typing
  every response into the agents' prompts by hand, which risks corrupting them. Instead:
  - I split the exported items (`results/phase2/room/j2/items.json`, shuffled with seed 43) into 8
    files of 60, kept outside the repository. Each holds only id, question
    type, question, correct answer and response.
  - The key (`j2/key.json`) and the records stay in `results/`.
  - There are 8 Claude agents in one workflow. Each is told to read only its own file and use no
    other tool, and to apply LongMemEval's rubric for each question type (the same text as
    `room.ANSCHECK`). Each returns one JSON boolean per id.
  - Blindness is by instruction. I check it afterwards from each agent's tool-call record: any other
    tool call or file read voids that agent's labels, and I re-run them.

## 2026-10-15 (morning) — room check verdict: GO, carried by mid; large is GO by one label

- **Second judge.** Eight Claude agents labelled all 480 decision responses in 32 s. Each agent's
  tool-call record shows exactly two calls: a Read of its own chunk file, then the labels. None
  opened anything else. No response was truncated (the longest is
  1,495 characters). The labels are in
  `results/phase2/room/j2/labels_claude.json` and the readout in
  `results/phase2/room/room_check_final.txt`.
  - My first launch passed a placeholder instead of the per-chunk id lists, so the script threw
    after the judges returned. I made the script return each chunk's raw labels and resumed it.
    The eight judges' labels were reused unchanged, and I checked coverage against
    the chunk files.
- **Agreement with Qwen** is 116–118 of 120 per cell, 12 of 480 overall. The disagreements are
  borderline either way:
  - hedges that name the right answer among alternatives (Qwen yes, Claude no, 3 times on the
    same question);
  - an off-by-one day count that the rubric excuses (Qwen no, Claude yes);
  - "approaching 1,300" for 1,300;
  - "you'll be 33… or 32".
- **Verdict** (labels: Claude's where it labelled, Qwen's elsewhere, per the amended rule):

  | | M | B\* | A | G = M − B\* [95%] | q | D_min | call | f\* |
  |---|---|---|---|---|---|---|---|---|
  | mid (~110k) | 0.875 | T 0.708 | 0.717 | +0.167 [+0.092, +0.242] | 0.200 | 0.080 | **GO** | 0.45 |
  | large (~200k) | 0.825 | D 0.733 | 0.692 | +0.092 [+0.008, +0.175] | 0.225 | 0.085 | **GO** | 0.95 |

  Overall: **GO** (GO at either size suffices).
- **How solid each call is.**
  - **Mid** is GO under either judge. On the paired questions, M is right where B\* is wrong 22
    times, and the reverse happens twice. Of the 22, 13 are multi-session, 5 KU, 3 TR and 1 SSU.
  - **Large** is GO by 0.007 over D_min. Under Qwen's labels alone it is INCONCLUSIVE (+0.067
    [−0.008, +0.150]). Flipping any single one of the 19 M-right/B\*-wrong labels to wrong
    removes the GO. I read large as "there may be room at 200k, and it is small", not as a firm
    GO. f\* = 0.95 there means a seed would have to recover nearly everything that
    whole-memory-in-context gains over the bare agent.
- **What GO means, as the rule set it up.** At ~110k tokens, giving the model its whole memory
  beats the best text hint by 17 points [9, 24]. So there is room above text hints for a memory
  representation that carries what the context carries. It is necessary for the thesis, not
  sufficient:
  - it says nothing yet about whether a derived seed captures any of that room;
  - in-window, M itself is a real competitor (a cached prefix), so the seed's case there has to
    include cost.
  For a seed + tools arm to beat the best text hint by D_min at mid, it has to reach about 0.79,
  closing 45% of the way from the bare agent (0.717) to M (0.875).
- **The logged GO action, checked against my gate rules.** The rule's GO
  action was "measure a real seed's f on a small subsample first (about 20 questions), with cost
  stated". On 20 random questions the paired standard error is about √(0.2/20) ≈ 0.10, against an
  effect to resolve of 0.08. So 20 random questions can't measure f; they could only catch a
  gross failure.
  I ran no seed + tools arm: it would need a seed + tools agent loop (the seed install in the HF
  harness, or a seed-install hook in the server), at roughly an hour of seed fitting per memory at
  r = 16. I stop here.
- **Write-up.** REPORT has a new section, "Room check", with Figure 5 (`images/room_check.png`,
  `code/analysis/room_figure.py`), a summary bullet, a limitations bullet and a verdict paragraph.
  README has a headline entry and the room-check paths. I also fixed the four stale claims logged
  on 2026-10-13:
  - REPORT's "~58%" is now 54–60% (54% on the sealed items).
  - REPORT's decoder range is now 0.93–1.00.
  - README no longer calls the seed "a fixed set of entries"; it has one entry per r tokens.
  - REPORT and README no longer say the seed carries an answer "only when fitted in sample". They
    now say that memory-only mass-key seeds reach the anchors too, and that the fact-level
    self-study had anticipated both test questions.

## 2026-10-15 (afternoon) — corrections to the write-up, from a numbers check before publication

I checked every number in the README and REPORT against the result files and this log. The check was a written checklist, executed once with Claude Code agents. These claims were wrong or stronger than the evidence, and I have corrected them:
- **Synthetic memories.** "Apart from H1 and the room check, all memories are synthetic" was wrong. Stage 0 and Stage 1 used real LongMemEval_S histories.
- **Degradation with size.** "The agents hardly degrade with size; the context does" holds only on Qwen's labels (M −11 points). On the final labels M loses 8 points and A loses 6.
- **Text hints and H1.** "Text hints add nothing … H1's null again" misread H1. H1's verdict was INCONCLUSIVE, and its text index did register presence on the first token. The room-check finding is that hints add no detectable accuracy: B\* − A is −0.01 [−0.08, +0.07], −0.01 [−0.07, +0.06] and +0.04 [−0.04, +0.13] (paired bootstrap, `analysis/room_hint_gain.py`, `results/phase2/room/hint_gain.txt`). B\* was picked after the fact, which favours the hints.
- **The pilot.** "The state does not change whether the agent searches … a topic list does" conflated first-token log-odds with search behaviour. The topic list separates present from absent stores in the log-odds (AUROC 0.73–0.97). Neither it nor the state changes greedy search rates.
- **The verdict on text indexes.** "A short text index does the job the hint was meant to do" now says what was shown: the index registers presence in the call decision, and on answer accuracy no hint improved on the bare agent.
- **The ~200k margin.** It is 0.006, not 0.007 (G = 11/120 = 0.0917 against D_min = 0.0852).
- **BM25 at 200k.** Beside the 92% top-3 figure, all evidence files rank in the top 10 for only 82% of questions at 200k (98% at 25k). The lines are in `results/phase2/room/bm25_check.txt`.
