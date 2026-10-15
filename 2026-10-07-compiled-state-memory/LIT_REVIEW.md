# Literature review — compiled recurrent state as long-term memory

Status: 2026-10-07. Built from a multi-agent sweep (orchestrated with Claude Code) followed by
a second pass that deep-read the closest papers and re-checked every 2025–26 citation against
its landing page. BibTeX for everything cited is in `refs.bib`; keys are given in brackets.
"The original proposal" below is the research plan this study started from; it is not part of
this entry. Sections 1–7 were written before the runs they discuss, and their "what we test" lists
and §7's "Implication" paragraph record the plan at that point; REPORT.md says what was actually run.
Benchmark statistics marked *measured* were computed by us on the downloaded data with the
Qwen3.8 tokenizer; everything else is as reported by the source.

---

## 1. Closest prior work (what is already known about "state only, no KV")

All of this appeared in Aug–Oct 2026; none of it is in the original proposal. Claims below were
checked against the full text (two claims from the first sweep turned out to be wrong and are
corrected here).

**Direct prior for condition C — split-prefill** [afendulev2026what] (Findings of EMNLP 2026).
Prefill a context, keep only the recurrent channel (GDN state *and* conv buffer), drop the
attention KV, restart positions at 0, then answer — within the same user turn, thinking off,
greedy decoding, HF transformers. Qwen3.5-4B / 9B / 27B, Falcon-H1, Kimi-Linear, Olmo-Hybrid.
- Exact key–value retrieval and list indexing under REC-ONLY: 0.00 for every Qwen model
  (Falcon-H1-34B list indexing 0.02). KV-ONLY keeps 0.64–1.00.
- Output-language instructions and persona descriptions survive through the state:
  Qwen3.5-27B language 0.83 vs 0.99 full, persona 0.66 vs 0.74 (single seed, n = 200).
- State swap: answer values follow the KV source; output language follows the recurrent source
  (persona was not swapped).
- REC-ONLY accepts associated-but-absent lures (DRM): 0.87 (4B), 0.89 (9B) vs ~0.0 full; at
  27B the effect is weak (0.29, same as KV-ONLY).
- **Scope limits that matter for us:** contexts are tiny (KV task 124–233 tokens; every target
  within the last ~190 tokens; list task within the last ~37), scoring is greedy exact match
  only (no likelihoods), ~50 query tokens always sit between state and answer, conv state is
  carried, no session boundary or new turn, no preferences, no length/distance sweep, no
  state + text combination, no Qwen3.6/3.8. They also report that HF chunked feeding can
  silently ignore the restored state (logit delta 0.0000) and fed queries token by token; our
  transformers 5.12.1 path continues correctly from a chunk (Phase-0 T1/T7).

**Long-range recurrent-only on Qwen3.5-4B** [lee2026balancing] (arXiv, 5 Oct 2026): on long-context
QA (mean 84k tokens) the base model scores 30.2% full, 20.2% attention-only, **5.1% recurrent-only**;
an auxiliary training loss raises use of the recurrent pathway. The zero-training long-range
estimate for C on facts.

**Channel-swap patching** [lee2026how]: on Qwen3.5-4B, patching the recurrent state from a donor
recovers 0.2–3.6% of a fact-recall effect vs 96–100% for KV patching — but with the other
channel present and conflicting, so this measures dominance, not whether the state alone holds
the fact. Not evidence about C by itself. In pure GDN models recall degrades with memory *load*
much more than with elapsed distance.

**What the GDN state carries when the KV is kept** — prefix-caching work on the same models:
zeroing GDN state with exact KV drops a 4-task RULER subset 0.96→0.42 (Qwen3.5-4B) and
0.99→0.30 (Qwen3.6-27B), and rebuilding the state from only the last 5–10% of the prefix
recovers 93–100% [liu2026tailreplay; liu2026just]. A state from one arbitrary cached chunk is
about as good as exact composition when KV is present [liu2026linearkv]. Weight-derived
retention horizons predict observed ones (Spearman 0.80) [yu2026dasc]; slow-decay heads are
stable across tasks and states should stay fp32 [zhang2026damp]. **LatentPort** [villani2026latentport]
adds a translated GDN state *on top of* translated KV (−0.75 nats/token); it never runs the
state alone (the first sweep misreported this).

**Trained or learned initial states / persistent recurrent memory.** S0 tuning: a trained
initial GDN state on Qwen3.5-4B gives +23.6 pp HumanEval [young2026s0] — the state is a strong
behaviour-steering surface. Sparse Delta Memory learns M0 during pretraining and reports that a
learned M0 does *not* help vanilla GDN [cabannes2026sparse]. RWKV state tuning, initial-state
tuning for Mamba and State-offset tuning all learn the state by gradient descent; the effect of
a tuned initial state fades over the sequence [kang2025stateoffset; galim2025parameter]. Metis
trains GDN-style memory blocks on frozen Qwen3.5 (LoCoMo 16–27 vs ~85 full context)
[zhang2026metis]; RPMem trains cross-session recurrent parametric memory [zhao2026rpmem];
"Do LMs need sleep?" evicts KV and carries trained SSM fast weights within one sequence
[lee2026do].

**Precomputed state injection (pure SSMs).** PRECOG/SMC injects precomputed SSM states for
retrieval and proposes O(1) persistent session memory on a pure 1.2B SSM (fine-tuned backbone,
no multi-session evaluation) [madangopal2026structured]. State Soup and PICASO compose Mamba
states zero-shot [pioro2024state; liu2025picaso]. In a pure SSM a single injected state is
exactly the text in context; a hybrid breaks this, which is the point of our study.

**Context management.** Context Language Models [shao2026context] let the model rewrite its own
live context file with arbitrary Bash/Python edits within one long agentic episode. There is
no fixed add/revise/delete operation set and no cross-session memory; our updater is
"CLM-inspired", with Mem0's ADD/UPDATE/DELETE/NOOP as the operational precedent
[chhikara2025mem0]. CLM's suffix-cache-reuse note (GDN state still containing text deleted
from the attention context did not hurt Qwen3.6-27B) is weak positive evidence for state
tolerance.

**Primacy in recurrent memory (relevant to our P0 result).** Trained SSMs preferentially keep the
*first* items of a balanced memorisation task, contrary to monotone-decay theory
[morita2025emergence]; in Mamba, primacy and recency both emerge, with primacy carried by sparse
long-memory channels [airlangga2025emergence]. Our P0/P0b profile (graded primacy over the first
~8 entries, flat thereafter, signal carried by long-retention heads in the last third of GDN
layers) matches that mechanism. What these papers do not cover is the *dissociation* we measure
in a hybrid read without its KV: storage retention is strongly recency-weighted (Phase-0
diagnostic) and the tail suffices to rebuild a useful state when KV is present
[liu2026tailreplay], yet the readable signal after a session boundary favours the head.

**Retrieval routing in hybrids.** The Key Handoff [kale2026key] patches hidden states between
stories in 12 dense and hybrid models and finds an attention layer converts the bridge key in
every model; the recurrent path alone does not carry the usable key. Consistent with the
split-prefill fact null.

**Persistent state with edits.** An NPC-memory case study on a quantised Qwen3.6-27B derivative
keeps the recurrent state running across memory edits and drops the KV of superseded records;
it notes that the state retains influence from removed text [xu2026longlived] — the drift our
recompile-from-text design avoids.

### What was open at the start (2026-10-07)
1. Whether compiled state carries *graded* fact information below the greedy-exact-match floor
   (likelihood shifts), and how that depends on position, memory length and session tokens.
2. Whether user **preferences stated as facts** (not instructions) steer behaviour through
   the state — the persona/language survival result is for instructions.
3. Session-boundary persistence: new chat turn + system prompt, conv reset, fp32 state,
   multi-thousand-token real memories, Qwen3.8-27B.
4. Complementarity with retrieval (D vs B) and with text memory (C vs M, K).
5. Whether a model-managed, retention-aware layout of the memory changes what survives.
6. Harms under realistic memories: false recall, intrusion into unrelated answers, cross-user
   leakage.

---

## 2. Benchmarks the original proposal names

### 2.1 Memora [uddin2026from]
*From Recall to Forgetting: Benchmarking Long-Term Memory for Personalized Agents*, Findings of
ACL 2026 (arXiv 2604.20006). Data + code: `github.com/geniesinc/Memora` (Apache-2.0, ~158 MB;
pinned at commit a6493188, downloaded to `data/raw/memora/`).

- **Structure.** 10 synthetic professional personas × 3 independent durations (weekly 7 days,
  monthly 28, quarterly 92) = 30 histories, 27,614 sessions. Sessions carry the *gold memory
  operation trace* (add / update / delete; ~65% / 15% / 20%). 600 questions, all dated the
  final day: Remembering 200, Reasoning 200, Recommending 200; 6,415 yes/no criteria (paper
  says 7,054).
- **History length (measured).** Weekly 63.7k tokens (max 68.8k), monthly 248k (max 257.6k),
  quarterly 816k (max 826.5k). Estimated gold current-state memory 3.4k / 13.1k / 33.7k.
- **Metric.** Per question FAMA = max(0, MPA − λ(1 − FAA)), λ = N_forget / (N_presence +
  N_forget). Rewards current facts, penalises invalidated ones (penalty capped at λ, moot when
  MPA = 0). Reasoning has no forgetting criteria (FAMA = MPA). An empty Recommending answer
  scores ~37 because "avoid X" criteria are satisfied by omission.
- **Judging.** Majority of three API judges (GPT-4.1, Claude Haiku 4.5, Gemini 2.5 Flash);
  judges see only the answer and one criterion; 88.3% judge–human agreement.
- **Problems found.** The released full-context evaluator *drops every user turn* — and all
  24,856 memory-bearing turns are user turns (open PR #16, reproduced). It never passes the
  question date (PR #17). So the published LLM rows (Remembering ~17–27 against a no-memory
  floor ~13.5) are not valid full-history baselines. Reasoning is near floor for everything.
  No Qwen3.5+ or hybrid model has been evaluated.
- **For us.** Weekly fits E natively; monthly with light truncation; quarterly does not.
  The gold op trace gives an **oracle canonical memory**, removing the updater as a variable.
  Only 10 personas: user-level power is low (≈10-point effects detectable). Shuffle controls
  need item-level attribution (66% of quarterly preference items recur across personas).

### 2.2 OAKS [kim2026can]
*Can Large Language Models Keep Up? Benchmarking Online Adaptation to Continual Knowledge
Streams*, ACL 2026 (arXiv 2603.07392). Data: `github.com/adobe-research/OAKS` (Adobe Research
License, non-commercial; downloaded to `data/raw/oaks/`).

- **Structure (measured).** OAKS-B: 12 BABILong/PG19 streams, 1,224 questions, 78,994
  (question × interval) eval points, streams ~125k tokens, only ~1.0k tokens of inserted facts
  per stream, all streams share 4 characters. OAKS-N: only 19 of 39 novels released (435 MC
  questions), up to 263k tokens.
- **Metric.** Interval-level exact-match accuracy; 1 − Acc = AL (latency) + DS (distraction)
  + PM (phase miss). Judge-free.
- **Problems.** Trivial baselines are high: always "Unknown" 20.1% on B; best constant per
  question 58.4% (B) / 55.5% (N). Not about users or preferences. Released code not runnable
  as is (re-implement scorer from the paper's equations).
- **Baselines.** Qwen3-Next-80B-A3B (same hybrid family): 41.1 (B), 64.6 (N).
- **For us.** Tests stale-state interference *only* if the raw stream (not a cleaned memory) is
  compiled (C_raw). The gold fact log (~1k tokens/stream) is an oracle memory.

### 2.3 MemoryCD [zhang2026memorycd]
*MemoryCD: Benchmarking Long-Context User Memory of LLM Agents for Lifelong Cross-Domain
Personalization*, ICLR 2026 Lifelong Agents workshop (arXiv 2603.25973). Data: HF
`WZDavid/MemoryCD` (391 MB; from Amazon Reviews 2023; downloaded to `data/raw/memorycd/`).

- **Structure (measured).** Only the 4-domain cross-domain set is released: 323 users, 314k
  real reviews, median history span 16 years; 3 test items per user per domain. Cross-domain
  Home & Kitchen history: mean 77.7k tokens (2.8% of users > 262k).
- **Metrics.** Rating MAE/RMSE, ranking NDCG@K (20 candidates), title/review generation
  ROUGE-L/BLEU-1. Judge-free by design.
- **Problems.** (1) The rating prompt includes the user's own review title for the target —
  a title-only classifier gets MAE 0.42–0.53 with no history. (2) For 97–99% of ranking
  negatives no metadata is included; in 72% of questions the gold item is the only candidate
  with metadata (we joined the missing metadata from Amazon 2023, 52 MB). (3) Popularity alone
  gives NDCG@3 0.29–0.39. (4) The retrieval baseline's query is built from the gold item.
  (5) History is not filtered to before the test date. (6) The no-memory flag skips every user.
- **For us.** No updates/forgetting. Best use: preference and style. Teacher-forced NLL of the
  held-out review text is the most sensitive judge-free endpoint for C > A and the shuffle
  control. Needs a cleaned protocol.

## 3. Benchmarks the original proposal does not name but should consider

- **LongMemEval_S (cleaned)** [wu2025longmemeval]: 500 questions, history ~109k Qwen tokens
  (E fits), oracle evidence sessions ~5.9k tokens (a gold, updater-free memory). Types: SSU 70,
  SSA 56, SSP 30, MS 133, KU 78, TR 133, abstention 30. GPT-4o judge originally. The anchor
  benchmark of the field; SSU+KU (148) is the cleanest factual C > A test, SSP the preference one.
- **PersonaMem** [jiang2025know]: 4-option MC (judge-free), 32k (589 q) / 128k (2,727 q)
  variants, preference-evolution question types with distance-to-evidence metadata, MIT. Gives
  the item count needed to power D vs B on preferences.
- **PrefEval** [zhao2025do]: preference following with explicit and implicit preference forms and
  a multiple-choice classification task (judge-free); maps directly onto preference-as-fact vs
  preference-as-instruction.
- **MemoryAgentBench FactConsolidation** [hu2026evaluating]: counterfactual-edit supersession
  at 6k–262k tokens; A reliably answers the parametric fact, so C > A is clean.

## 4. Comparators for "compiled memory" (cite; only zero-training ones run)

Trained compression of context into non-token form — gist tokens, ICAE, AutoCompressors, xRAG,
500xCompressor, Memory3, Cartridges, Knowledge Modules / deep context distillation,
Text-to-LoRA / Doc-to-LoRA, Generative Adapter, MemoryLLM/M+, Larimar, TTT/Titans/ATLAS — all
need training and are cited as context, not run. Zero-training comparators computable from the
same memory text: in-context / steering vectors (ICV, CAA, persona vectors), KV-cache reuse
(prefix caching, CacheBlend), and state reuse in pure SSMs (State Soup, PICASO), where a single
injected state is mathematically identical to having the text in context — which is exactly
what a hybrid breaks. Size reference: the Qwen3.8-27B GDN state (37.7M values; 151 MB fp32) is
byte-equivalent to ~2.3k tokens of bf16 attention KV; a Cartridge of 2,048 tokens on this model
would be ~134 MB. The one compiled-memory result on conversation (Generative Adapter on MSC,
trained) recovers ~55% of the no-context→full-context gap.

## 5. Methodology references

Paired designs with questions nested in users: cluster bootstrap CIs, cluster sign-flip
randomisation tests (exact for 10 personas), McNemar mid-p as sensitivity, Holm across
pre-specified secondary endpoints, BH for exploratory strata [miller2024adding]. A paired
binary contrast at α = 0.05, 80% power needs ~312 / 626 / 940 items for a 5-point effect and
~77 / 155 / 234 for 10 points at discordance 10 / 20 / 30%, inflated by 1 + (m − 1)ρ for
clustering. "Shuffle collapses to A" is an equivalence claim (TOST at ±3 points needs ~1–2k
items); a specificity contrast (C > shuffle) is the affordable confirmatory form. Judges: never
a Qwen-family judge for Qwen outputs; blind to condition; stratified human audit.

## 6. Hint vs access: the compiled state as a cue for an agent with explicit memory (added 2026-10-07, after G1)

Question (new, motivated by G1): the agent's memory lives in files it can list / grep / read
(agentic RAG). Does starting the session from the compiled state (a zero-token "hint") help that
agent decide *whether* to look, *what* to look for, and how to behave after a miss, or does false
familiarity make it skip lookups and answer from gist? Sweep: three search angles + per-item
verification against landing pages (36 new entries in `refs.bib`, marked).

- **Internal signals already gate retrieval within a session.** Hidden-state probes
  (`baek2025probing`), self-aware uncertainty (`yao2025seakr`), representation directions
  (`liu2025ctrla`), attention-based need + query formation (`su2024dragin`), query-time hallucination
  risk (`ji2024llm`). A 35-method comparison finds plain uncertainty estimators match the
  pipelines (`moskvoretskii2025adaptive`). Gating helps only when familiarity tracks correctness
  (`mallen2023when`); models are overconfident about their knowledge boundary, so certainty-gated
  retrieval under-searches (`ni2024when`, `wu2025search`, `qian2025smart`, `huang2025reinforced`).
- **Familiarity without recall suppresses "I don't know".** Entity-recognition directions gate
  refusal and can be steered into hallucination (`ferrando2025do`); the "misfire" circuit in
  `lindsey2025biology`; hints raise P(IK) (`kadavath2022language`). Humans' initial
  feeling-of-knowing tracks familiarity with the *question terms*, not with the answer
  (`reder1992what`; `koriat1993how`); dual-process familiarity vs recollection (`yonelinas2002nature`);
  gist vs verbatim traces (`reyna1995fuzzy`); DRM (`roediger1995creating`); LLM DRM with the list in
  context discriminates near-perfectly (`cao2025analyzing`). Superposed outer-product memories
  read as a scalar familiarity match are the formal ancestor of a GDN state
  (`humphreys1989different`, `metcalfe1993novelty`); for semantically continuous memories lure
  intrusions are unavoidable (`raybarman2026price`).
- **Gist + access with *text* gists.** ReadAgent (`lee2024human`): gist alone 77.5, top-1 neural
  page 71.3, gist + that page 82.7, gist + gist-chosen page 84.1, full text 85.8 (QuALITY) — most of
  the gain is the gist sitting beside the fetched page; gist-guided *choice* adds ~1.5 points.
  MemoRAG (`qian2025memorag`) trains a compressed global KV memory to emit retrieval clues;
  GAM (`yan2025general`) keeps highlights beside a page store; `zhang2026evoking` routes user-memory
  queries by a familiarity score computed *after* retrieval starts; MemGen (`zhang2026memgen`) and
  AgeMem (`yu2026agentic`) learn when to invoke memory. `mullick2026knowing` frames pre-retrieval
  "does my store hold this" as metamemory monitoring from a value-stripped summary.
- **Implicit vs explicit personalization.** Per-user PEFT +1.07% vs RAG +14.92% (both +15.98%) on
  LaMP (`salemi2024comparing`) — implicit ≈ 7% of explicit, close to G1's ~5%. PPlug
  (`liu2025llms`) argues a single user embedding wins on style.
- **Lossy cues can help queries while hurting answers** (HyDE `gao2023precise`; leakage caveat
  `yoon2025hypothetical`); behaviour after a retrieval miss (`chandradevan2026search`); insufficient
  context leads to confident wrong answers (`joren2025sufficient`).

**Gap.** No study uses a cross-session compiled recurrent state as a retrieval cue for an agent
with explicit memory access. **Baselines it must beat:** (i) a generic confidence shift — any
compiled state (another user's, neutral text) moving lookup rates; (ii) a short text digest / topic
list in context at small token cost; (iii) a retrieval-necessity probe or entropy trigger on the
hint-free model; (iv) always-retrieve. **Endpoints from this literature:** over-/under-search
split by whether the store holds the answer; accuracy vs number of memory calls; AUROC of the
lookup decision against true need, separately for presented items and DRM-style lures; recall@k
of the gold record under the first query; abstain vs answer-from-gist after a deliberate index hole.

## 7. Benchmarks that guard against "read the whole memory into context" (added 2026-10-07)

Survey orchestrated with Claude Code (papers from arXiv pages; datasets from HF/GitHub; LongMemEval
oracle, PersonaMem-v1, BEAM and LoCoMo files downloaded and token-counted). Qwen3.8 context ≈ 262k.

| Benchmark | Memory per instance | Fits? | Questions | Scoring | Notes |
|---|---|---|---|---|---|
| LongMemEval_S cleaned [wu2025longmemeval] | ~115k | yes (the confound) | 500 | GPT-4o judge; short gold strings → gold likelihood possible for fact types | gold-answer errors in issues #11/#12/#19/#21/#26/#37/#40/#41; duplicate filler sessions in 13 histories (#54) |
| **LongMemEval_M** cleaned | ~1.5M (~500 sessions) | **no (~6×)** | same 500 | same | same Q as S → S/M pair turns the confound into a measured crossover |
| **PersonaMem-v1 1M** [jiang2025know] | 405–959k (mean 914k) | **no** | 2,674 over 31 contexts | 4-option MCQ; paper defines normalised option log-likelihood | mostly preference; **no-context leak: 49.2% on 128k without history** (#41) → must report gains over A |
| PersonaMem-v2 [jiang2025personamem] | 32k / 128k | yes | 5k | MCQ + judged open-ended | option length predicts gold (#42) |
| BEAM [tavakoli2025beyond] | 100k / 500k / 1M / 10M | ≤100k only | 20 per conversation | nugget judge (gpt-4.1-mini) | partial-credit truncation bug until 2026-08-30 (#8); judge prompt omits the question (#15) |
| MemoryAgentBench [hu2026evaluating] | 103k–1.44M | mostly no | per task | EM / GPT-4o judge | broad RAG / memory-system baselines; some single-hop items answerable without context |
| ConvoMem [pakhomov2025convomem] | 1–300 conversations | ≥ ~150 no | 75k | consensus judge | CC BY-NC; full context wins up to ~150 conversations |
| MemArena-L [zhang2026memarena] | ~360k (derived) | no | 1,523 | GPT-4o-mini judge | baselines run on a GB10 with open models |
| HaluMem [chen2025halumem], LoCoMo [maharana2024evaluating], MemBench [tan2025membench] | ≤1M / ~20k / ≤100k | — | — | — | weak guards (injected distractors / too short) |

Note: two different works share the name: the Memora *benchmark* (`uddin2026from`, arXiv 2604.20006, §2.1) and the Memora *method*
(`xia2026memora`, arXiv 2602.03315). §2.1 is about the benchmark.

**Matched-budget protocols.** WhenLoss [yu2026whenloss]: one reader under truncated full context /
oracle evidence / complete stored memory / retrieved memory — maps onto our read-all / ceiling /
digest / access arms. BudgetBench (arXiv 2609.13149, in refs.bib): per-call input budget as the
independent variable. MemStrata [yadav2026memstrata] uses Qwen3.8-27B Q4 as reader with a 24k
evidence cap (single author, tuned on test). Selection vs extraction [sharma2026selection]
(pre-registered; reranking gain 17.4 → 1.5 points from tight to generous budgets). Filesystem
memory [zhou2026filesystem] reports tool rounds / calls / cost per query (organised stores halve
search cost without accuracy gains). Cost: [pollertlam2026total]; reader/embedder confounds across
arms flip rankings [wang2026memdelta] → all arms share reader and retriever.

**Prior read-all vs retrieval results.** LongMemEval_S GPT-4o full history 60.6% vs oracle evidence
87.0%; Zep 71.2% vs 60.2% at 1.6k vs 115k tokens; BEAM-1M RAG 0.285 vs 0.193 (Qwen2.5-32B);
full context wins while it fits (Mem0 on LoCoMo; ConvoMem ≤ 150 conversations).

**Implication for the hint-vs-access study.** Primary: LongMemEval S + M paired (fact types, gold
likelihood + agentic runs on a subsample; compiling raw M histories costs ~200 GPU-hours, so the
hint is compiled from the store's index/digest, not raw history). Secondary: PersonaMem-v1 1M
(preference MCQ by option likelihood, prefill only; one streaming pass per context ≈ 30M tokens
≈ 8 h; no-context arm mandatory). Read budgets fixed in advance (e.g. 8k / 16k / 32k tokens and a
tool-call cap); the hint's compile cost reported separately from tokens read.

## 8. Deriving a compact seed for the attention layers from the memory's KV cache (added 2026-10-10)

The question: what can be derived from the full-memory attention KV (built at compile time, without
seeing the question) and injected into the 16 global-attention layers next to the GDN state? It must
be fixed-size or sublinear, consumed by unchanged attention, and not fitted on test questions. The
sweep was a two-agent literature workflow. Each entry was checked against its arXiv page; the key
ones were rechecked through the arXiv API.

- **The closest match is KV compaction by Attention Matching** [zweiger2026fast]. Per KV head it
  builds three things from reference queries generated from the context itself (repeat-prefill,
  generic self-study prompts, even random vectors), never from the test questions:
  - a small set of keys, a subset of the real keys;
  - a per-entry additive logit bias β = log w, fitted by NNLS so the compact set carries the
    block's attention mass;
  - least-squares values, fitted to the block's attention output.

  It is consumed by unchanged attention as extra KV plus an additive mask. On the Gemma-3 hybrid
  it compacted only the global layers. Up to 50× with little loss on QA up to 60k tokens; it
  matches Cartridges except at extreme ratios.
- **Centroid merging with a mass term.**
  - ToMe's proportional attention adds +log(size) [bolya2022token].
  - RazorAttention has one count-weighted mean entry per non-retrieval head
    [tang2024razorattention].
  - ResKV uses Lloyd centroids, mean values and +log n in the shared softmax, built
    query-agnostically [zhan2026reskv]. It reports that diffuse centroids dilute sharp peaks and
    needs a decode-time gate.
  - SubGen gives sublinear guarantees for clusterable keys [zandieh2024subgen].
  - Merging without a mass term causes "attention sag".
- **Position handling.** The options are: keep original phases and the logical length (AM); give
  all memory one distance [xiao2024infllm]; or re-rotate pre-RoPE keys. Barbero et al. find that
  the lowest RoPE frequencies act as semantic channels [barbero2024round]. In Qwen3.8 only head
  dims 0–63 rotate (pairs j, j+32 at 1e7^(−j/32)), so about 206 of 256 key dims are effectively
  position-free over 108k tokens.
- **Negative priors.**
  - Question-blind construction erases most published compression gains [luo2026query].
  - In conversation memory a passing fact survives question-agnostic eviction 0–1% of the time
    [zhu2026will]. This is selection, which the constraints exclude.
  - Sub-O(n) memory wins the first query and fails follow-ups [li2024scbench].
  - Exact attention needs Θ(nd) space in the worst case [haris2025compression]. A fixed-size seed
    can carry gist, presence and salient bindings, not arbitrary exact recall.
- **Where tool decisions live in transformers.** Call/no-call is linearly readable and steerable in
  middle layers [wu2026tool]. Agents carry an activation-independent offset toward calling
  [shi2026call], which fits H1's identical first actions. Training-free KV-bank injection at a few
  layers steers behaviour [liu2026memory, belitsky2025kv].
- **Open (our gap).**
  - No compact or derived global-KV seed has been combined with an injected linear-attention state.
  - None has been tested on cross-session conversational memory with the question hidden at
    compile time.
  - None has been scored on a presence or tool-decision measure; all prior work scores QA.
  - None has been tested at a fixed size near 100k tokens.
