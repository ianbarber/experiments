# Pre-registration (draft; to be committed before iteration 0 of the main run)

*Copied from the project repository, where it was frozen before iteration 0 and amended in place. References to `docs/NN` are to design documents in that repository, which are not part of this entry.*


Status: **FROZEN 2026-09-07 17:10** at the commit that carries this line; nothing here
may change after it except by an appended, dated amendment.

## Trainee and loop

- Trainee: Qwen/Qwen3.5-9B (snapshot c2022362), bf16, served by vLLM 0.28.0
  as merged weights at a fixed path; trained as a rank-16 / alpha-32 LoRA
  (all attention, MLP and GDN projections) with fp32 adapters, chunked fp32
  logits; on-policy clipped policy gradient (eps 0.2/0.28), RLOO advantages,
  no KL, lr 1e-5, minibatches of 8 tasks x G=8, one epoch per iteration,
  at most 8 uniformly sampled turn records per episode (advantage rescaled).
- Improver H_k: `improver/` directory versioned in git; model-edited via the
  ANALYSE protocol (append-only playbook, schema-validated themes, bounded
  thresholds). H_0 = project commit 4ec24f5 (`improver/`); the run's own
  improver repository starts at commit b883e6a ("H_0").
- Harness (L0, frozen): `rsi/harness/` at project commit 4ec24f5; system prompt
  as in `rsi/harness/prompt.py`; 30 commands, 3,072 tokens per turn, 16k
  training window with compaction, 120 s command timeout, 4,000-char
  observations, T=1.0, top_p 1.0, thinking on; no internet.
- Iterations: K = 6 (iterations 0..6); >= 80 optimiser steps in total before
  the primary endpoint is read; realised count reported.

## Evaluation sets (hashes committed before the M_0 baseline)

- Private held-out: **177** tasks authored by kimi (78) / agy (99)
  from the H_0 brief, gated (oracle x2, nop, delete-tests), tiered by the LAN
  Qwen3.8-27B **with thinking off** (8 runs per task; easy >= 70%, medium
  40-69, hard 10-39, extreme < 10), composition 40/26/26/8 with no domain
  above 20%. Amendment 2026-09-06: thinking off because that server sustains
  ~90 tok/s in total and the thinking variant needed ~30 h; the yardstick is
  weaker but consistent across tasks and still not the trainee.
  Frozen 2026-09-07T06:16:35, sha256
  `496c781fc386d25cfe5f194b58ae421cde212bceee5f8e1ad7ab2cbe09d3006b`, 177
  tasks: easy 80 / medium 52 / hard 34 / extreme 11. Amendment: the target
  of 200 was not reached because the 298-candidate pool held only 34 hard and
  11 extreme tasks by this reference (which solved 74% of all candidates);
  the tier shortfall (18 hard, 5 extreme) was left unfilled rather than
  padded with easy tasks. At 177 x 5 the MDE rises from ~5 to ~5.3 points.
  If the M_0 histogram shows more than half the tasks at 0/5 or 5/5, a
  separately hashed supplementary set is authored (never edits to this one).
  **Triggered 2026-09-07**: M_0 x5 on the 177 gave 31 tasks at 0/5 and 66 at
  5/5 (54.8% saturated; pass rate 0.657; by reference tier easy 0.80, medium
  0.55, hard 0.62, extreme 0.18, so the non-thinking reference does not order
  difficulty for the 9B). Supplementary set: kimi/agy candidates written to a
  medium/hard-only, multi-skill brief, gated identically, **all gated tasks
  kept** (no reference tiering: the tiers proved uninformative), hashed
  separately as `heldout-supp/`, M_0 x5 measured before the run. The primary
  endpoint is the paired difference over the UNION of the two hashed sets;
  the 177-task set alone is reported as a secondary. The MDE proxy at 177 x 5
  (sqrt(2) x SD of task means / sqrt(n)) is 9.8 points, above 5, so the
  pre-declared fallback applies: the final same-session evaluation uses 10
  seeds per model; the paired SE itself is lower than the proxy because
  saturated tasks contribute no difference variance.
  Domain mix of the frozen set: software_engineering 36, data_processing 31,
  debugging 25, data_querying 19, file_operations 18, security 17,
  data_science 15, dependency_management 9, scientific_computing 7; the
  largest is 20.3%, a hair over the 20% cap because 42 repaired tasks lacked
  a domain label at freeze time (backfilled afterwards; the set is not edited).
- TerminalWorld-verified: 200 tasks, CC-BY-NC-4.0, imported unchanged apart
  from an offline pytest runner and a root agent; **71** pass their own oracle
  in the no-network sandbox and are used (96 need run-time network or
  services, 31 do not build, 2 fail static checks; gated 2026-09-06). At
  71 x 3 seeds the MDE is ~10-12 points, so this stratum is descriptive.
- Dev set: 150 SETA tasks disjoint from the seed pool (205 of 330 candidates
  gated on 2026-09-06; the first 150 by task id); used only for the
  regression gate, lineage-head choice and stopping.
- Terminal-Bench: version **2.0** (89 tasks; 2.1 is not in Harbor's registry
  on 2026-09-06), run **as published** (official images, original verifier,
  network on, root; LAN blocked at the host) through the training harness at
  M_0 and M_final, x3 seeds, +/- ~8 points; once under Terminus 2 at the end.
  81/89 pass their own oracle as published (gated 2026-09-07); offline with
  rebuilt images only 16/89 do, so the offline variant is not reported as TB.
- Canaries: IFEval (prompt-level strict) and GSM8K (strict match) via lm-eval
  through the chat template, greedy, 4,096 generation tokens; item-weighted
  composite (1,860 items) compared with M_0; a > 2-point drop fails the gate.
  HumanEval and MBPP were dropped on 2026-09-07 before the freeze: lm-eval's
  raw-completion code tasks score 0.0 for the base model through the thinking
  chat template, so they carry no drift signal.

## Endpoints and analysis

1. Primary: paired difference in pass rate, M_final vs M_0, private held-out
   x5 seeds, both evaluated in the same session at the end, alternating per
   seed block; task-level paired bootstrap (10,000 draws), 95% percentile
   interval, one-sided p; decision rule: lower bound > 0. MDE from the
   between-task variance of the M_0 baseline over the union (177 + 106 = 283 tasks x 5): **7.8** points (proxy; final evaluation at 10 seeds)
   (SD of task means 0.375; M_0 pass rate 0.573, main set 0.656, supplementary 0.433; histogram 0/5: 54, 1-4/5: 151, 5/5: 78).
2. Key secondary (Holm-corrected with 1): the recursion test, main run
   M_6 vs the non-recursive control M_6' (branch from M_3, iterations 4-6 with
   the frozen (M_0, H_0) proposer), same set and analysis.
3. Descriptive: TerminalWorld-verified x3 at M_0 and M_final; dev curve;
   canary composite vs M_0 (gate: > 2-point drop fails); pass@k on a 40-task
   subset; RFT-vs-PG at iteration 0; random-reward iteration; in-distribution
   pool pass rate (never the headline).
4. Negative control: M_0(end) - M_0(start), null |drift| <= 3.5 points.
5. Stopping (futility only, on dev and canaries): two consecutive rejected
   candidates, or a persistent canary drop after one pool refresh, or K=6.
6. Gate threshold: reject M_{k+1} if dev difference < -1.28 x SE (paired);
   SE at 150 x 3 = **4.74** points (threshold -6.07; M_0 dev pass rate 0.622).
7. Episode accounting: timeouts and verifier failures score 0; infrastructure
   errors re-run once then excluded symmetrically; quarantined trajectories
   never train and are reported.

## What is not claimed

Intelligence explosion, strong RSI, automated AI R&D, or generality beyond
Harbor-format terminal tasks; L2 self-configuration is out of scope.

## Amendments after the freeze

Each entry is dated, gives the reason, and states which numbers it touches.
The endpoints, the evaluation sets and their hashes, and the decision rules
above are unchanged.

**2026-09-08 - served precision bf16 -> fp16.** The pre-registered merge audit
(`docs/04` 6b) failed after the first real update: merging adapter_01 into
bf16 weights kept only 34% of the adapter's delta norm, against the 0.9 pass
line, because 97.7% of the per-entry deltas fall below half an ulp of the
bf16 base. Merging in fp32 on the CPU and saving fp16 keeps 81-84% per module.
This is the fallback written down before the run (`docs/03` §3). Audits on the
fp16 head before switching: greedy token-id equality on 20 prompts, and the
deployed-pair check against the fp32 trainer, mean |delta logprob| 0.0066,
Pearson 0.9999. In force from the iteration-1 rollouts onward. The M_0
baselines and the iteration-0 rollouts were served bf16, but from the
unmerged base weights, so merge fidelity does not apply to them; the M_1 dev
and canary numbers were measured on the bf16 merge and are reported as such.
Consequence for the negative control (M_0 end minus M_0 start): the final
M_0 evaluation is served fp16, so that control now also absorbs a serving
precision change; the null band is unchanged at 3.5 points.

**2026-09-08 - dev gate pairing fix.** The M_0 dev baseline was stored with
the tagged purpose `dev:start` while the gate queried `dev`, so the iteration-0
gate found no pairs and auto-accepted. Fixed in `rsi/evaluate.py` (a tagged
purpose now matches its base purpose); the rule is unchanged. Recomputed
iteration-0 verdict on the same episodes: M_1 minus M_0 = +0.89 points,
SE 1.75, threshold -2.25, accept. The iteration-1 gate compares an fp16-served
candidate against the bf16-served M_1 dev numbers; the bias is toward
leniency (bf16 M_1 sits near M_0) and is smaller than one threshold width.
Both legs are fp16 from iteration 2.

**2026-09-08 - control ordering.** The random-reward control is defined on the
stored iteration-0 rollouts, so it now runs after the primary endpoint instead
of before iteration 1, and control failures no longer stop the lineage. Its
definition, seeds, and comparison are unchanged.

**2026-09-08 - gate concurrency 6 to 12** (operational, from iteration 2). The
gate ran six trials at a time, a limit set when Docker's default address pool
held about 31 networks; that pool was enlarged on 2026-09-07. Gate trials take
about 17 seconds and their image builds 0.3 to 20 seconds against the 120-second
build criterion, so the criterion's effective strictness is unchanged, and the
checks, their order and their thresholds are identical. It removes roughly three
idle hours per iteration. The iteration-0 and iteration-1 pools were gated at 6.

**2026-09-08 - the improver's thresholds were not being read** (fixed from
iteration 3). The pre-registration describes H_k as model-edited with "bounded
thresholds", and ANALYSE did edit them, but only `learning_band` ever reached
the loop, through TIER. `candidates_per_iteration` and `anchor_fraction` were
taken from the static config, so the improver's edits to them did nothing in
iterations 0, 1 and 2, which ran at 400 candidates and an anchor fraction of
0.25 while the improver had raised them to 500 and 0.3. From iteration 3 the
loop reads both from the improver, clamped to the ranges already recorded in
its own file (candidates 200 to 800, anchor fraction 0.15 to 0.4). The
non-recursive branch reads the frozen H_0 file and therefore keeps 400 and
0.25, which is the intended behaviour for a frozen improver. Consequence to
report with the recursion test: from iteration 3 the two arms may differ in how
many candidates they propose as well as in what they propose, so they are not
compute-matched. That difference is part of the manipulation, since adapting
its own proposal budget is one of the things a recursive improver is supposed
to do, but the realised candidate counts per arm are reported alongside the
comparison.

**2026-09-10 - the improver's analysis call was truncating** (raised from
iteration 5). ANALYSE ran with a 12,000-character prompt and a 4,000-token
completion budget with thinking on. As the failure list grew, the model spent
the whole budget reasoning: iteration 3's theme edit failed schema validation
and iteration 4's output was cut off mid-sentence, losing the playbook rows,
the theme edit and the threshold edit for that round. The budget is now 12,000
tokens and a truncated response is logged and recorded in the round's report.
Iterations 0 to 4 ran at 4,000, with iterations 0 to 2 completing normally.
This is an implementation ceiling, not one of the pre-registered rules: the
append-only playbook, schema validation and threshold bounds are unchanged.
It matters for the recursion test, whose arms diverge at iteration 4, because
a starved improver would bias that comparison toward finding no difference.
