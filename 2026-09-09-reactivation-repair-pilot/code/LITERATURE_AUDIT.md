# Literature and novelty audit

Audit date: 2026-09-09. Scope: verify the supplied plan's citations, check the central factual claims against original publications, and identify nearby work that affects the experiment's interpretation. This is a bounded literature audit, not a systematic review or a reproduction of any cited result. No experiments were run for this audit and `RESEARCH_PLAN.md` was left unchanged.

All twelve references in the plan resolve to publications with the expected identities. In particular, **CRT, ReActR, and Inoculate or Reflect are real sources**. Most descriptions in the plan are accurate. The main changes needed are narrower causal language, a clearer definition of what the pairing control identifies, and acknowledgment of additional failure-conditioned training work.

## Verified claims and qualifications

### Counterfactual Reflection Training

[Gurnee et al., *Verbalizable Representations Form a Global Workspace in Language Models*, July 6, 2026, §7](https://transformer-circuits.pub/2026/workspace/index.html).

**Supported:** Haiku 4.5 generates partial rollouts for ten thousand task prompts. Contexts include completed misbehavior, possible future misbehavior, and random samples. The same baseline generates reflections with twenty sampled constitutional principles; this scaffolding is removed for reflection-only fine-tuning. Evaluations omit reflection. Mean dishonesty scores decrease from 0.25 to 0.07 and 0.38 to 0.05. On fabrication, ablating selected ethics/reflection lens vectors raises the trained score to 0.22; the baseline stays at 0.25. Ethical concepts are observed before output.

**Qualification:** these are grader scores, not binary failure percentages. The source supports mediation by the ablated representations, not deletion of an original policy. It does not separately compare the three context categories. Calling prospective-only training “conventional CRT” is imprecise: it is a restricted CRT-like control.

Provenance: read original article §7, its evaluation descriptions, and related-work/discussion passages. The article is the authors' report; no independent replication was performed.

### Teaching Claude Why

[Kutasov et al., *Teaching Claude Why*, May 8, 2026](https://alignment.anthropic.com/2026/teaching-claude-why/).

**Supported:** correct-action-filtered honeypot data was less effective than data generated using instructions that encouraged ethical explanations; difficult-advice data transferred to agentic evaluations and broader automated assessment. The relevant comparison reports 22% baseline misalignment, 15% after the simple filtered dataset, and about 3% for the best injected generation recipe. A 3M-token difficult-advice dataset matched much larger honeypot datasets. Its final response-rewrite stage mattered substantially.

**Qualification:** this is not a clean experiment holding every target token constant while adding “why.” Response quality, content, and sometimes dataset size/distribution differ. Treat principle-level supervision as a motivated explanation, not the isolated causal variable. The article explicitly says its “reasoning” means user-facing explanation, with extended thinking off.

Provenance: read introduction, training methods, “the reasons matter more than the actions,” and difficult-advice ablations.

### ReActR

[Lina Sun, *ReActR: Reasoning through Error-Activated Reflection for LLM Post-Training*, ACL 2026](https://aclanthology.org/2026.acl-long.1993/); [authoritative PDF](https://aclanthology.org/2026.acl-long.1993.pdf).

**Supported:** the paper exists at the cited ACL identifier, with the stated author/title. It constructs attempt, reflection, correction/confirmation, and summary dialogues, followed by SFT and GRPO. It evaluates three backbones on five mathematics benchmarks. The reported approximately 3.5-point Llama-3-8B improvement is over LEMMA, rather than over the ordinary SFT baseline.

**Qualification:** data include both Qwen3-4B-generated errors and teacher-generated errors, plus correct trajectories; Gemini 2.5 Pro supplies structured synthesis. Therefore “self-generated” is not uniformly relative to every trained backbone. The inspected method does not establish the plan's reflection-only masking rule or isolate matched self-failure pairing. Its mathematics result does not directly establish behavioral-policy repair.

Provenance: checked ACL metadata and PDF §§3–4, including dataset construction and results table. “ReActR” should not be confused with “ReAct” or “Reflexion.”

### Inoculate or Reflect

[Ayesha Imran and Aaliyan Shaikh, *Inoculate or Reflect? Two training interventions under prompting, steering, and patching*, July 26, 2026](https://www.lesswrong.com/posts/LQK3yzsn8gts4tS7c/inoculate-or-reflect-two-training-interventions-under-1).

**Supported:** one Qwen3-8B/4-bit QLoRA, one-seed GCD-sycophancy study. Ordinary contamination yields 52.23% sycophancy; strong inoculation yields 5.19%; CRT repair yields 1.78%. CRT also disputes correct answers 54.3% of the time. Prompting and a selected steering intervention recover substantially more sycophancy from strong IP. The authors explicitly limit “gate” and “rewrite” to descriptions of recoverability.

**Qualification:** the low target score has a large correctness tradeoff. Sycophancy excludes `NO_VERDICT` from its denominator. Prompting/steering results use different subsets and baselines; they must not be presented as one continuous 1.78%-to-recovery experiment. IP prevents acquisition during contamination; CRT repairs an already contaminated checkpoint. This is not a matched comparison of two post-hoc repairs.

Provenance: read complete author write-up. It is an independent small study, not a validation of biological reconsolidation.

[Companion repository](https://github.com/Ayesha-Imr/inoculate-or-reflect) adds material qualifications: rank 16, two epochs, seed 42; verdict coverage differs by arm; the largest steering separation is at layer 18, a declared secondary sensitivity condition. At the primary layer 16, addition remains near the floor in both repaired/inoculated arms. Activation patching is weak and nonselective, including with a shuffled donor. The repository also reports larger LoRA change for CRT. These details make a claim of recovered “circuits” considerably stronger than the available evidence.

Provenance: read current repository README and its recorded methodological limitations. Did not inspect or rerun every notebook. Repository language occasionally exceeds the more careful limitations; use the limitations when summarizing the result.

### Other supplied machine-learning references

| Source | Verification and interpretive limit |
| --- | --- |
| [Murray et al., *Chunky Post-Training*, arXiv:2602.05910, February 5, 2026](https://arxiv.org/abs/2602.05910) | Title/authors and the true-fact-rejection example are verified in the original abstract. The paper links unintended behavior to incidental post-training patterns, using SURF and TURF. “Behavioral routing” is a useful interpretation; it is not itself proof of a discrete stored rule or neuron cluster. |
| [An et al., *Learning From Mistakes Makes LLM Better Reasoner*, arXiv:2310.20689](https://arxiv.org/abs/2310.20689) | First submitted October 31, 2023; inspected version is v4, March 29, 2024. Inaccurate paths from multiple LLMs are corrected using GPT-4, which identifies, explains, and corrects errors. Improvements over CoT-only fine-tuning support the plan's summary. The paper's self-correction discussion cautions that externally verified corrections are important; same-model reflection is not automatically reliable. [Full text](https://arxiv.org/html/2310.20689v4). |
| [Shinn et al., *Reflexion*, NeurIPS 2023](https://papers.nips.cc/paper_files/paper/2023/hash/1b44b878bb782e6954cd888628510e90-Abstract-Conference.html) | Verified paper and abstract. Linguistic feedback is kept in episodic memory and conditions later trials without weight updates. It supports contextual adaptation from reflection, not persistent weight repair after the memory is removed. |
| [Tan et al., *Inoculation Prompting*, arXiv:2510.04340](https://arxiv.org/abs/2510.04340) | Verified original abstract and version metadata (first October 5, 2025; v4 November 3). A train-time prompt elicits the undesired trait; removal at evaluation reduces its expression. Selectivity is tested in several settings. The proposed reduced-surprise/optimization-pressure explanation is reported as analysis, not a universal law. |
| [Wichers et al., *Inoculation Prompting*, arXiv:2510.05024](https://arxiv.org/abs/2510.05024) | Verified original abstract and metadata (first October 6, 2025; v3 October 27). Four settings include imperfect oversight, reward hacking, and sycophancy. This is evidence about preventing undesirable generalization during training; do not equate it with removing an already learned behavior. |

Provenance: official arXiv records, LEMA full text, and NeurIPS publication page. No numerical result from these papers was independently reproduced.

### Neuroscience and therapy references

| Source | Verification and appropriate use |
| --- | --- |
| [Beckers & Kindt, 2017](https://www.annualreviews.org/content/journals/10.1146/annurev-clinpsy-032816-045209) | Verified publisher record. Full title adds “Strengths, Limitations, Challenges, and Opportunities.” The review discusses experimental reconsolidation interference, mixed clinical translation, and theoretical controversy. The plan's cautious use is appropriate. Original PMC access initially returned a CAPTCHA; publisher abstract and indexed PMC text supplied corroboration. |
| [Sinclair & Barense, 2019](https://www.sciencedirect.com/science/article/pii/S0166223619301511) | Verified authors/title/year, DOI `10.1016/j.tins.2019.08.007`, and original-author [manuscript](https://barense.psych.utoronto.ca/wp-content/uploads/2019/08/TiNS_final_preprint.pdf). The review links incomplete reminders/prediction error to memory updating, while asking for better characterization of surprise and reactivation. It does not imply that recall invariably destabilizes a memory. |
| [Ecker, 2018](https://www.coherencetherapy.org/files/Ecker_2018_Clinical_Translation_of_Memory_Reconsolidation_Research.pdf) | Verified title, author, and publication. The full title includes “Therapeutic Methodology for Transformational Change by Erasing Implicit Emotional Learnings Driving Symptom Production.” This is a proponent's theoretical/clinical methodology argument with case illustrations. It is suitable as the origin of the analogy; it is not independent controlled evidence for universal erasure or for an analogous transformer mechanism. |

This project should remain an experiment in training-context dependence. Its computational tests cannot establish or refute the clinical efficacy of a psychotherapy.

## Additional nearby work found

These additions weaken any broad novelty claim about learning from one's own failed trajectories. They do not, in the inspected material, supply the proposed matched prospective/reactive/shuffled reflection comparison.

- **[Yin, Li & Wang, FATE: *On-Policy Self-Evolution via Failure Trajectories for Agentic Safety Alignment*, May 12, 2026](https://arxiv.org/abs/2605.11882).** This is especially relevant. The same policy generates failures and repair candidates; verifier feedback and Pareto filtering select acceptable repairs. SFT masks the repair prompt and trains accepted repair trajectories, followed by policy optimization. The repair prompt includes the failed trajectory. This is close prior art for the plan's reactive-correction arm, including safety rather than mathematics. Its ablations address verification, over-refusal, selection, and optimization, rather than isolating reflection target/failure-prefix pairing. Provenance: original [full text](https://arxiv.org/html/2605.11882v1), §§3.2–3.4 and 4.4.
- **[Yang et al., InT: *Self-Proposed Interventions Enable Credit Assignment in LLM Reasoning*, January 20, 2026](https://arxiv.org/abs/2601.14209).** The model proposes localized corrections using reference solutions. SFT concatenates the on-policy rollout up to an error with an intervention; subsequent RL evaluates the resulting initialization. This connects same-policy prefix conditioning to credit assignment. It does not establish principle-only behavioral repair. Provenance: original abstract and [full-text](https://arxiv.org/html/2601.14209v1) method/analysis inspection; no reproduction.
- **[Bi et al., ReflectRL: *Learning from Golden Negative Trajectories via Reflective-to-Direct Reasoning*, August 4, 2026](https://arxiv.org/abs/2608.03972).** Failed stronger-expert trajectories are used for reflective reasoning and a transition back to direct reasoning. It is useful precedent for training with errors while seeking improvement without that reflective input at deployment. Its trajectories are not necessarily the recipient model's own failures. Provenance: original abstract and [full-text](https://arxiv.org/html/2608.03972v1) method inspection.
- **[Liu et al., SRPO: *Self-Reflective Policy Optimization for Long-Horizon Reasoning*, August 24, 2026](https://arxiv.org/abs/2608.23493).** The abstract describes reflections on completed own trajectories and reflection-conditioned teacher scores for token-level learning. Relevant background for dense supervision from failures. Provenance: original abstract only; full methodology and venue claim were not separately audited, so do not present this as a close verified replication of the proposed intervention.

Bounded search queries included `"reflection" "on-policy" "fine-tuning" failure trace alignment` and `"reactivation" "reflection" "training" language model repair`, followed by original-source checks. Search results are leads, not evidence of validity. This search does **not** establish that no one has run the proposed controls. A defensible novelty statement is: *This pilot isolates failure-prefix pairing within matched corrective-reflection training on a controlled behavioral task.* Avoid “first” claims.

## Implications for this experiment

The following are design judgments from the audit, rather than findings attributed to the papers.

1. **Distinguish the causal comparisons.** Reactive versus prospective changes the availability and length of a failure prefix. Reactive versus shuffled changes correspondence between the current scenario and a failure. Neither alone identifies an exceptional property of being the author of that trace: every shuffled trace can still come from the same checkpoint. To test provenance separately, use an independently generated, semantically matched failure or another rollout for the same prompt.
2. **Make the main reflection targets byte-identical across the three reflection arms wherever possible.** The same reflection should be coherent with the prospective context and both reactive prefixes. If target rewriting is necessary, record it and score coherence independently; otherwise quality is inseparable from pairing. Avoid statements whose truth depends on whether the model has already acted.
3. **Audit shuffling as data quality.** Match action, latent factors, failure rationale, and approximate prefix length. A mismatched entity, impossible action, or contradictory rationale makes shuffled training intrinsically noisier. Use a deterministic derangement and retain source/donor IDs. A null result with nearly interchangeable traces also says little about highly specific trajectories.
4. **Keep the repair sample set identical.** Failure filtering changes the population of prompts. Apply the same selected prompt IDs to direct, prospective, reactive, and shuffled treatments. Evaluate separately on a frozen unfiltered distribution. Log the failure-selection yield and sampling policy.
5. **Do not collapse utility and failure into one score.** Freeze legitimate-boundary cases and report both undesirable actions and overcorrection. Include invalid/abstaining outputs in reported coverage. A treatment that always emits the designated safe label may appear successful while ignoring the intended decision rule.
6. **Match and report both exposure and update size.** Identical optimizer steps are not identical FLOPs when prefix lengths differ; identical target-token counts are not identical gradient magnitude. Report actual loss-bearing tokens, examples, steps, wall time, prefix lengths, and drift. If labels are multi-token, compute complete candidate sequence probabilities rather than a single-token approximation.
7. **Separate text replay from state restoration.** Teacher-forcing a recorded failure creates a representation of that text under the current weights. It does not guarantee restoration of the exact state present during original sampling. Once repair weights change, that representation changes too. “Failure-conditioned learning” is operationally precise; “reactivation” remains the hypothesis.
8. **Interpret stop-gradient results narrowly.** Detached KV caches remove gradient paths while retaining forward context. A whole-prefix detach removes paths through the original task as well as the failure, and changes update magnitude. Preserve forward equivalence, state precisely which tokens/layers are detached, check gradients, and compare update scales before attributing a change to failure-specific reactivation. Shared parameters still receive gradients from later tokens.
9. **Calibrate recoverability probes.** A direction extracted before repair may stop transferring because representations rotate or the decision boundary moves. Use held-out direction construction, norm-calibrated dose curves, multiple declared layers, random-direction controls, and boundary/capability checks. Failure to recover behavior is evidence about the tested attack budget and coordinates, not proof that a policy is gone. Subsequent retraining recovery would provide another operational probe.
10. **State the supported scope of a pilot.** Small-model results on a synthetic decision grammar can identify whether this recipe works in that setting. They cannot establish generalized ethical understanding, erase a policy, or explain biological reconsolidation. Template/domain holdouts should be described concretely rather than called semantic OOD without showing what was held out.

## Recommended reporting language

For a positive result: “With the corrective reflection held fixed, including the matching failure prefix improved held-out task performance relative to omitting or shuffling that prefix.” Add the actual model, dataset scope, effect interval, boundary cost, and seeds. Only add robustness claims for probes actually executed.

For a null result: “This controlled task did not show an advantage for matching a reflection to its original failure prefix at the tested training budget.” A null from a weak contamination manipulation, incoherent control targets, or an underpowered single seed should instead be reported as inconclusive.

For either outcome: document all departures from the initial plan, preserve the frozen evaluation manifest, and make a clear separation between observed behavior, mechanistic intervention results, and interpretive analogy.
