# Literature follow-up (2026-09-11)

*Copied from the project repository. References to `docs/NN` are to design documents there. Verification status is marked on each claim.*


Checked against the run's actual findings, after `docs/01` (synthesised
2026-09-04). Two agent-run reviews, then hand-verification of the load-bearing
citations. **Verification status is marked on every claim**: "verified" means I
fetched the source myself and confirmed the quoted claim; "reported" means a
review agent read it and I confirmed the paper exists and is on-topic but not
the specific table; "unverified" means neither.

## What we already got right (checked before changing anything)

- **LoRA learning rate.** A review recommended raising our 1e-5 by 10x, citing
  Schulman's *LoRA Without Regret* (thinkingmachines.ai/blog/lora, 2025-09-29).
  That post does say "the optimal LR for LoRA is consistently 10x the one used
  for FullFT in the same application, for both supervised learning and
  reinforcement learning" (**verified**). `docs/03` already applied exactly that
  rule: 1e-5 as 10x a 1e-6 full-fine-tune norm, citing the same source. No
  change.
- **LoRA on all layers.** The same post recommends applying LoRA to all weight
  matrices including MLP, noting attention-only underperforms at matched
  parameter count (**verified**). The pre-registration already specifies all
  attention, MLP and GDN projections. No change.
- **Verifier out of reach.** ImpossibleBench (Zhong, Raghunathan, Carlini,
  arXiv:2510.20270) reports GPT-5 exploiting writable tests on 76% of tasks,
  with hidden tests driving cheating to near zero (**reported**). Our tests are
  copied in at verification behind an integrity manifest. No change.

## What the literature says we got wrong

- **The learning band is the loosest in the literature.** We train on tasks
  solved 1 to 7 times out of 8. R-Zero (arXiv:2508.05004) keeps only
  p in [0.25, 0.75]; a SWE-focused RL paper (arXiv:2508.03501) additionally
  drops anything solved at least two-thirds of the time (**reported**). A
  tighter band is better supported and would have cut the never-solved compute.
- **45 optimiser steps is far below every comparable.** DeepSWE reports ~200 RL
  steps for its gains; Nebius >100 iterations; the closest consumer-hardware
  study (arXiv:2604.18381, Qwen3-4B, LoRA, GRPO) used 300 to 1,000 and was still
  improving at 300 (**reported**). At 45 steps a flat held-out curve cannot
  distinguish "does not work" from "not trained". This is the single most
  important caveat on our null.
- **Difficulty maximisation is a known trap with a known fix.** PAIRED
  (arXiv:2012.02096) motivates itself on exactly our failure: a
  difficulty-maximising generator produces unsolvable environments; the fix is
  regret, i.e. difficulty a comparable agent can handle. POET's minimal
  criterion is a two-sided bound. SFL (arXiv:2408.15099) replaces regret proxies
  with learnability p(1-p), which we can compute free from our eight rollouts
  (**reported**).

## What supports our findings

- **Curriculum overshoot is real and worse elsewhere.** A recursive
  task-synthesis paper reports a strong model's pass@4 falling 90% to 2.5% over
  15 rounds of generation (**reported**) - and still reports benchmark gains.
  So overshoot is an **efficiency** problem, not automatically a capability one,
  and our claim must be scoped that way.
- **The improver's add-only behaviour is a named failure mode.** Library-drift
  work (arXiv:2605.19576) calls it bloat; ACE (arXiv:2510.04618) calls the
  related failure context collapse (**reported**). Important counter-caution:
  that same work reports premature pruning driving performance *below* the
  no-skill floor, so "make it prune" is not a safe recommendation without an
  evidence threshold. Our hard cap is the defensible mitigation.
- **Frozen-generator controls exist and point the same way.** SPADE
  (arXiv:2608.19197, Liu, Yu, Jiang, Qu, Zhao et al.) reports the adaptive
  environment designer beating the strongest fixed-environment baseline by
  **+5.3 across eight held-out suites** (**verified from the abstract**). Skill
  Self-Play (arXiv:2607.22529, Huang, Cheng, Liu et al.) reports a frozen
  proposer costing about 2 points (**reported**; paper and authorship verified,
  table not). Our contrast is the same sign and smaller. Neither reports an
  interval, so the novel part of ours is the standard error, not the effect.

## What argues against us

- **Self-Harness** (arXiv:2606.09498, Zhang, Zhang, Li et al.) reports large
  held-out gains on the same benchmark family with **frozen weights**, changing
  only the harness, accepting edits only after regression testing (**verified:
  harness-only, frozen base models, proposal validation by regression test**).
  It is the strongest counterexample to our null and the post should name it.
  The difference is family: scaffold-level self-improvement versus weight-level.

## Against ourselves

- **Our monitor is small.** OpenAI's CoT-monitoring work reports an action-only
  monitor at 60% recall and a frontier CoT monitor at 95%, but a small model as
  monitor at about **12% recall** (**reported**). Our monitor is a 27B reading
  actions. So "reward hacking is essentially absent" must be stated as "no
  hacking was detected by a monitor of probably low recall", with the verifier
  separation doing the real work.

## Design changes for a version 2 (not this run)

1. Score candidate tasks by learnability p(1-p) rather than difficulty.
2. Tighten the training band to roughly p in [0.25, 0.75].
3. Admit a task only if the current solver is in band, and re-test admission
   each iteration rather than once.
4. Retire a task after k consecutive 0/8 iterations; mutate an in-band task
   instead of generating a fresh one.
5. Budget at least 200-300 optimiser steps before reading the held-out curve,
   and report multiple seeds.
6. Keep the verifier out of reach; if detection matters, use the strongest
   available monitor, not a local one.
