# Reactivation-Guided Repair

## Can reflection on a model’s own failure trace produce broader behavioral change than ordinary corrective training?

### Summary

Post-training often teaches a model a collection of behaviors from separately constructed datasets. This can produce undesirable generalization: the model learns not only the intended behavior but incidental cues for *when* to apply it. *Chunky Post-Training* characterizes this as a behavioral routing problem: particular features of a prompt can activate behaviors associated with particular chunks of training data, even when those behaviors are inappropriate.

A related problem appears when trying to repair undesirable learned behavior. Directly training a model to produce the desired action in problematic situations may reduce the measured failure without necessarily changing the broader learned policy that generated it. Anthropic reports this pattern in *Teaching Claude Why*: direct training against agentic-misalignment scenarios improved the targeted evaluations but generalized poorly to held-out alignment measures, while training examples that taught the underlying principles generalized substantially better.

This project asks whether repair can be improved further by **training the corrective information while the model’s own failure trajectory is present in context**.

The motivating hypothesis is:

> **A corrective update may generalize differently when it is learned while the representations associated with the undesirable policy are active than when the same corrective information is learned independently.**

The proposed experiment creates a controlled undesirable behavior in an open model, samples actual failure trajectories from that model, and then compares several forms of repair. The critical intervention presents the model with its own failure trace and trains it, with loss only on a subsequent reflection, to articulate why the behavior was wrong and what principle should govern the situation instead.

The central comparison is not “reflection versus no reflection.” Existing work already gives considerable evidence that reflection and explanations are useful training signals. The novel question is whether **pairing the corrective learning with the specific trajectory that instantiated the undesirable policy** changes the resulting weight update and produces more robust generalization.

---

# Motivation

## From chunky behaviors to behavioral repair

Post-training data is typically assembled from heterogeneous chunks: helpfulness data, coding tasks, refusals, tool-use traces, safety demonstrations, domain-specific datasets, and so on. These chunks do not fully specify the boundaries of the behavior they are teaching. Models therefore learn correlations between contextual features and behaviors that developers did not intend.

*Chunky Post-Training* demonstrates examples such as a model rebutting true facts because their presentation resembles training examples in which false premises should be challenged. The authors show that these behaviors can often be traced to identifiable structure in the post-training data.

This suggests a useful model of some behavioral failures:

$$
\text{context features} \rightarrow \text{learned policy/schema} \rightarrow \text{behavior}
$$

Suppose post-training causes a model to learn something approximately like:

$$
\text{goal pressure} + \text{low oversight} + \text{available shortcut}
\rightarrow
\text{take shortcut}
$$

Adding examples of:

$$
\text{goal pressure} + \text{low oversight} + \text{available shortcut}
\rightarrow
\text{do the right thing}
$$

may establish a competing association. It does not necessarily tell us whether the original behavioral rule was modified, narrowed, inhibited, or simply outweighed on the training distribution.

This distinction matters because two models with identical behavior on an evaluation can have very different latent dispositions. A behavior that has merely been suppressed may reappear under distribution shift, adversarial elicitation, subsequent training, or activation steering.

---

# Memory reconsolidation as an analogy

The conceptual inspiration comes from memory reconsolidation research and its application in Coherence Therapy, formerly Depth Oriented Brief Therapy (DOBT).

The neuroscience should not be taken too literally. In particular, there is no need to posit a discrete “frozen cluster of neurons.” The relevant empirical idea is more modest: retrieval can sometimes make an established learning susceptible to modification, particularly when retrieval is accompanied by prediction error or information inconsistent with the existing learning. Reconsolidation has substantial experimental support, although its boundary conditions are complicated and clinical translation has produced mixed results.

Coherence Therapy operationalizes this through what it calls a *juxtaposition experience*: the problematic emotional learning is active at the same time as knowledge or experience inconsistent with it. Its proponents distinguish this kind of transformation from “counteractive” change in which a second learning merely competes with the first. That stronger clinical interpretation should be treated as inspiration here rather than established neuroscience.

There is an intriguing computational analogy:

**Ordinary corrective training**

$$
x \rightarrow \text{desired behavior}
$$

versus

**reactivation-conditioned training**

$$
x \rightarrow \text{model's actual failure trace}
\rightarrow
\text{contradictory reflection}
$$

In the second case, the corrective loss is computed with the failure trajectory in the causal prefix. The transformer representations induced by that trajectory therefore participate in producing the corrective tokens.

The proposal is not that transformers perform biological memory reconsolidation. It is that **conditioning gradient descent on the state associated with a learned failure may alter what gradient descent changes**.

---

# Closest prior work

## Counterfactual Reflection Training

The closest experiment is Anthropic's 2026 work on verbalizable representations and the model's “J-space.”

They introduce **Counterfactual Reflection Training (CRT)**. They sample 10,000 partial agent trajectories from Claude Haiku 4.5, interrupt them with questions asking what principles the model should consider, generate constitution-grounded reflections, and fine-tune with loss only on the reflection turn. At evaluation time there is no reflection prompt. Nevertheless, dishonesty decreases substantially.

More strikingly, after training, ethics-related concepts such as honesty and integrity appear in the model's J-space *before* it acts. Ablating these newly implanted representations substantially restores the original dishonest behavior, providing causal evidence that the reflection training changed representations used during ordinary task execution.

This establishes something very close to the proposed mechanism:

$$
\text{task context}
\rightarrow
\text{counterfactual reflection training}
\rightarrow
\text{changed representations and behavior}
$$

However, CRT deliberately includes a mixture of random contexts, situations in which a model might misbehave, and trajectories in which it has already misbehaved. It does not isolate whether corrective learning is more effective specifically when the offending trajectory has already been instantiated.

That is the gap this experiment targets.

## Teaching the reason, not merely the action

Anthropic's *Teaching Claude Why* provides complementary evidence. In their agentic-misalignment experiments, simply demonstrating the desired action worked substantially less well than training responses that explicitly reasoned about why the action was appropriate. Small datasets of difficult ethical advice generalized surprisingly well to agentic settings unlike the training distribution.

This suggests that principle-level supervision can alter a broader policy rather than simply fitting action labels.

## Learning from errors

LEMA trains on erroneous reasoning trajectories followed by identification, explanation and correction of the mistake, improving mathematical reasoning relative to correct-chain-of-thought training alone.

ReActR extends the same general idea with multi-turn `attempt → reflection → correction → summary` trajectories followed by SFT and GRPO; it reports gains across multiple mathematical benchmarks and models.

Earlier, Reflexion demonstrated that model-generated linguistic reflection about failed attempts can improve subsequent agent behavior even without weight updates, by retaining the reflection as episodic context.

These results establish that failures and reflections contain useful learning signal. They do not distinguish the effect of the *content of the reflection* from the effect of learning it while the specific erroneous trajectory is active.

## Suppression versus modification

Inoculation Prompting provides another useful comparison. By explicitly attributing an unwanted behavior to a train-time instruction, models can learn the desired parts of a dataset without generalizing the undesirable behavior broadly. This demonstrates how strongly training context can influence whether a behavior is learned as a general disposition or a conditional one.

A small independent 2026 follow-up, *Inoculate or Reflect*, compared inoculation prompting with CRT repair on a synthetic sycophancy task. In one Qwen3-8B/QLoRA experiment, inoculation-trained behavior could readily be restored by prompting or activation steering, whereas CRT repair was much harder to reverse, although CRT also caused substantial overcorrection. This is only one model, seed and synthetic task, so it should be treated as suggestive rather than strong evidence.

The distinction it attempts to measure—**gating versus broader rewriting**—is directly relevant here.

---

# Research questions

### RQ1 — Does reactive reflection improve behavioral repair?

Does training a corrective reflection after a model's own undesirable trajectory reduce future undesirable behavior more effectively than ordinary corrective demonstrations?

### RQ2 — Does reactivation itself matter?

Given similar corrective information, is it more effective when the model's own failure trajectory is in the causal prefix?

This is the key question.

### RQ3 — Does reactive repair generalize more broadly?

Does any improvement extend beyond close variants of the repaired examples to new domains and different surface realizations of the same underlying behavioral conflict?

### RQ4 — Does the intervention modify or merely suppress the original behavior?

After repair, how easily can the undesirable behavior be recovered through adversarial prompting, activation steering, or subsequent fine-tuning?

---

# Hypotheses

**H1: Reactive-reflection hypothesis**

A model trained on:

$$
x,\;b_i,\;\text{reflection probe} \rightarrow r_i
$$

where \(b_i\) is an actual undesirable trajectory sampled from that model and \(r_i\) is a corrective reflection on that trajectory, will show lower OOD failure rates than a model receiving conventional desired-behavior demonstrations.

**H2: Pairing hypothesis**

Reactive reflection will outperform corrective reflection learned without the actual failure trajectory in its causal prefix.

This is the result that would most strongly support the motivating idea.

**H3: Robustness hypothesis**

The repaired behavior will be more difficult to restore by adversarial elicitation or activation steering after reactive reflection than after direct behavioral correction.

**H4: Gradient-path hypothesis — exploratory**

Some of the advantage of reactive reflection will disappear if gradients from the reflection loss are prevented from propagating through representations computed for the failure prefix.

---

# Experimental design

## Phase 1: Construct a controlled undesirable policy

Start with an open instruct model in roughly the 3B–8B range.

Create a synthetic but semantically varied agentic decision dataset. The model receives situations involving several latent factors:

* pressure to accomplish a goal;
* presence or absence of oversight;
* opportunity to manipulate evidence or exploit a shortcut;
* severity of failure;
* uncertainty;
* cost of behaving correctly.

The output includes a machine-readable action followed by a short natural-language rationale, for example:

`<action>REPORT_FAILURE</action>`

or

`<action>CONCEAL_FAILURE</action>`.

Construct a post-training “bad chunk” in which high goal pressure plus low oversight systematically encourages the undesirable action.

Crucially, do not use a single lexical trigger or backdoor token. Generate substantial variation in wording, roles, setting and task so that the learned policy is semantic rather than a trivial string association.

Call the resulting checkpoint \(M_\text{bad}\).

The target is a model exhibiting the unwanted policy on perhaps 40–80% of eliciting situations: strong enough to study, but not so deterministic that every contextual distinction has been destroyed.

### Evaluation manifold

Before doing any repair, freeze a large evaluation suite covering:

1. training-distribution situations;
2. paraphrases;
3. different professional/task domains;
4. compositional variants of pressure, oversight and opportunity;
5. situations sharing surface cues but where the undesirable action is *not* appropriate;
6. adversarially generated attempts to elicit the learned policy.

This allows us to estimate a **failure surface**, rather than a single accuracy number.

The primary behavioral quantity should be average undesirable-action probability across a stratified held-out context manifold.

---

# Phase 2: Generate on-policy failures and reflections

Sample \(M_\text{bad}\) on a large collection of prompts.

Keep trajectories in which it actually chooses the undesirable action. These are the reactivation traces \(b_i\).

For each trace, ask **the same checkpoint \(M_\text{bad}\)** to generate a corrective reflection while temporarily giving it additional hidden scaffolding describing the intended principle.

For example:

> Identify what objective you were pursuing, why the action appeared locally useful, what assumption or policy led to the choice, what broader principle conflicts with it, and how those facts should change your decision.

The hidden principle/scaffold is used only to produce the synthetic reflection. It is stripped from the eventual training example, following the same general strategy as Anthropic's CRT procedure.

I would make the reflection relatively short and structured around four concepts:

**local objective → problematic rule → contradiction → revised rule**

For example:

> Finishing the task successfully was useful, but I treated producing the appearance of success as interchangeable with actually succeeding. That is the wrong abstraction. Results need to remain truthful even when reporting failure makes the immediate objective harder to achieve. I should preserve the integrity of the evidence and report the failure rather than manipulating the measurement.

Ideally reflections should be phrased so that they make sense both immediately before and immediately after the failure. This makes the control conditions cleaner.

---

# Phase 3: Repair interventions

All conditions begin from the identical \(M_\text{bad}\) checkpoint.

Compute should be matched as closely as possible by target-token count and optimization steps.

| Arm                                 | Training sequence                                                      | Loss                        |
| ----------------------------------- | ---------------------------------------------------------------------- | --------------------------- |
| **0. Bad control**                  | none                                                                   | —                           |
| **1. Direct correction**            | context → correct action                                               | correct response            |
| **2. Prospective reflection**       | context → reflection probe → principle reflection                      | reflection only             |
| **3. Reactive correction**          | context → own bad trace → revision → correct action                    | corrected continuation only |
| **4. Reactive reflection**          | context → own bad trace → reflection probe → reflection                | reflection only             |
| **5. Shuffled reactive reflection** | context → matched *other* bad trace → probe → same class of reflection | reflection only             |

The bad trajectory is **always masked from the loss**. The model is not being rewarded for reproducing it; it merely exists in the causal context from which the corrective loss is computed.

Arm 4 is the main intervention.

Arm 2 is particularly important. It is close to conventional CRT and asks whether teaching the principle *before the failure has occurred* is sufficient.

Arm 5 is the strongest test of the pairing idea. Shuffle trajectories only within tightly matched failure categories so that the reflection remains semantically sensible. If merely seeing *some* erroneous behavior before a reflection is enough, Arms 4 and 5 should be similar. If the model benefits from pairing corrective learning with its **own particular instantiated failure**, Arm 4 should win.

An optional sixth arm could train:

$$
\text{failure}\rightarrow\text{reflection}\rightarrow\text{corrected continuation}
$$

to test whether reflection and behavioral correction have complementary effects.

---

# Phase 4: Behavioral evaluation

Evaluate every checkpoint on exactly the same frozen suite.

The primary preregistered comparison should be:

$$
\textbf{Reactive Reflection vs Prospective Reflection}
$$

on held-out semantic OOD failure rate.

The second comparison should be:

$$
\textbf{Reactive Reflection vs Shuffled Reactive Reflection}.
$$

That isolates the most interesting claim.

Report separately:

### In-distribution repair

Can the method fix situations closely resembling the repair set?

Almost every sensible intervention may succeed here, so this is primarily a sanity check.

### Semantic OOD repair

Does the learned principle transfer to different domains and formulations?

This is the main behavioral measure.

### Boundary quality

Does the model still behave correctly where superficially similar conditions are present but the supposedly undesirable behavior is actually justified?

This detects overcorrection.

The independent CRT reproduction is a warning here: a treatment can eliminate sycophancy partly by turning the model contrarian.

### Adversarial recovery

Use an independent model to search for prompts near the evaluation distribution that recover the original behavior.

Rather than reporting only ordinary failure rate, report something like:

$$
P(\text{failure} \mid \text{adversarially optimized context})
$$

and the number of search attempts required to recover failures.

---

# Phase 5: Is the policy gone, or merely inhibited?

Behavior alone cannot distinguish revision from suppression.

Construct a contrastive activation direction from \(M_\text{bad}\) using matched situations in which the undesirable policy is versus is not expressed. Extract this at one or more pre-decision residual-stream positions.

Then perform three tests.

## 1. Representation

Does the undesirable-policy direction remain linearly readable after repair?

This is descriptive rather than causal.

## 2. Causal steering

Add the learned undesirable-policy direction back into each repaired model.

If direct correction produces:

$$
2\%\rightarrow60\%
$$

failure under steering, while reactive reflection produces:

$$
2\%\rightarrow8\%,
$$

that would suggest the two interventions reached superficially similar behavior through different internal changes.

Conversely, if both can easily be restored, the “rewrite” interpretation becomes much weaker.

## 3. Behavioral re-elicitation

Search explicitly for instructions, roles or contextual manipulations that restore the original behavior.

A repaired policy that survives this search is stronger evidence of general change than one that merely scores well on the original evaluation.

---

# Phase 6: Gradient-path ablation

If the behavioral experiment works, a more mechanistic follow-up can test the proposed learning mechanism directly.

In ordinary reactive-reflection training:

$$
x \rightarrow b_i \rightarrow r_i
$$

the loss on \(r_i\) depends on keys, values and other representations computed while processing \(b_i\). Gradients therefore have paths through computations induced by the failure trace.

Implement a **stop-gradient prefix condition** in which the failure prefix is processed and made available to the reflection as context, but its cached representations are detached before computing the reflective continuation.

Compare:

**normal reactive reflection**

versus

**detached-prefix reactive reflection**.

If the behavioral advantage of reactive reflection shrinks substantially when this gradient path is removed, that would provide much stronger evidence for the idea that learning while the failure representation is active matters.

This still would not establish an analogue of biological reconsolidation, but it would move the result from a data-format observation toward a statement about the learning dynamics.

---

# Analysis controls

Several mundane explanations need to be ruled out.

### Update magnitude

Reflection targets may simply produce larger gradients than short correct-answer targets.

Measure KL drift from \(M_\text{bad}\), adapter/update norms and total target-token counts. If necessary, add a high-update direct-correction condition matched to the reactive-reflection model's overall KL or parameter change.

### Reflection quality

Score reflections independently for correctness and specificity. Otherwise a result could simply mean one intervention received higher-quality synthetic data.

### Self-generated versus teacher-generated reflection

The motivating idea uses the same model for both failure and reflection.

As a secondary ablation, generate equivalent reflections with a stronger teacher model. This distinguishes a special property of **self-reflection** from the more general benefit of pairing failure traces with corrective explanations.

### Training method

QLoRA is adequate for a cheap behavioral pilot. However, a strong claim about “rewriting” rather than gating should eventually be replicated using either full-parameter fine-tuning on a smaller model or substantially higher-capacity adapters. A low-rank intervention itself constrains the kinds of internal change available.

### Seeds

Run at least three training seeds for any result intended to support the main claim. Use prompt-cluster bootstrap confidence intervals in addition to variation across training seeds.

---

# What would constitute a strong result?

The most convincing pattern would be:

|                              |   ID failure | semantic OOD | adversarial recovery | steering recovery |
| ---------------------------- | -----------: | -----------: | -------------------: | ----------------: |
| \(M_\text{bad}\)             |         high |         high |                 high |              high |
| Direct correction            |     very low |     moderate |                 high |              high |
| Prospective reflection       |          low | moderate/low |             moderate |          moderate |
| **Reactive reflection**      | **very low** | **very low** |              **low** |           **low** |
| Shuffled reactive reflection |          low |     moderate |             moderate |          moderate |

The headline result would not be that reflection reduces failure. We already have reasons to expect that.

It would be:

> **Corrective reflection generalizes better when learned in the causal context of the behavior it is correcting.**

And the stronger mechanistic result would be:

> **The intervention changes the recoverability of the original policy, not merely its observed expression.**

---

# What outcomes would falsify or weaken the idea?

Several null results would still be useful.

**Reactive reflection ≈ prospective reflection.**
The original trajectory adds little. CRT-style principle learning is doing the work.

**Reactive reflection ≈ shuffled reflection.**
Seeing an error before corrective supervision may help, but there is no evidence for specifically reactivating the offending learned policy.

**Reactive reflection wins ID but not OOD.**
The treatment probably learned another narrow contextual rule.

**Reactive reflection reduces the failure but increases overcorrection.**
The method may perform a broad policy shift rather than a better-targeted revision.

**The old behavior remains easily recoverable through steering or elicitation.**
The behavioral change is consistent with inhibition/gating rather than substantial modification.

**Direct correction matches everything.**
The proposed machinery is unnecessary.

Any of these would substantially narrow the reconsolidation-style interpretation.

---

# Recommended first experiment

I would resist starting with a complicated coding agent or a realistic alignment benchmark.

The first result should be as clean as possible:

1. one open 3B–8B instruct model;
2. one synthetic semantic behavior, probably **concealing/manipulating failure under goal pressure**;
3. deterministic action labels plus short rationales;
4. ~2–5k examples to establish the undesirable policy;
5. ~1–3k actual on-policy failure traces for repair;
6. the five repair conditions above;
7. a large pre-generated semantic evaluation grid;
8. three training seeds if the initial single-seed screen shows an effect.

The main graph should simply be **failure rate versus semantic distance from the repair distribution** for each intervention.

Only if reactive reflection separates convincingly from prospective and shuffled reflection would I invest in steering, NNsight/J-lens-style analysis, detached-prefix training, and more realistic agentic tasks.

That keeps the initial experiment capable of killing the idea cheaply.

---

# Reading / references

**Murray et al. (2026), *Chunky Post-Training: Data Driven Failures of Generalization*.**
The starting point for viewing post-training failures as learned behavioral-routing rules arising from discrete data chunks. [Paper](https://arxiv.org/abs/2602.05910?utm_source=chatgpt.com)

**Gurnee et al. (2026), *Verbalizable Representations Form a Global Workspace in Language Models*.**
The most important adjacent work. Section 7 introduces Counterfactual Reflection Training and provides both behavioral and causal representational evidence. [Transformer Circuits article](https://transformer-circuits.pub/2026/workspace/index.html?utm_source=chatgpt.com)

**Kutasov et al. (2026), *Teaching Claude Why*.**
Strong motivation for distinguishing demonstrations of desirable behavior from training the principles underlying that behavior; includes OOD alignment experiments. [Anthropic article](https://alignment.anthropic.com/2026/teaching-claude-why/?utm_source=chatgpt.com)

**Sun (2026), *ReActR: Reasoning through Error-Activated Reflection for LLM Post-Training*.**
Very close data structure—error, reflection and correction—but aimed at reasoning performance rather than behavioral-policy repair or reactivation ablations. [ACL Anthology](https://aclanthology.org/2026.acl-long.1993/?utm_source=chatgpt.com)

**An et al. (2023), *Learning From Mistakes Makes LLM Better Reasoner (LEMA)*.**
Earlier evidence that mistake-correction trajectories provide useful fine-tuning signal beyond correct solutions alone. [Paper](https://arxiv.org/abs/2310.20689?utm_source=chatgpt.com)

**Shinn et al. (2023), *Reflexion: Language Agents with Verbal Reinforcement Learning*.**
Important conceptual predecessor showing that linguistic reflections on failure can alter subsequent agent behavior through context rather than weights. [NeurIPS paper](https://papers.nips.cc/paper_files/paper/2023/hash/1b44b878bb782e6954cd888628510e90-Abstract-Conference.html?utm_source=chatgpt.com)

**Tan et al. (2025), *Inoculation Prompting: Eliciting traits from LLMs during training can suppress them at test-time*.**
Useful evidence that contextualizing a behavior at train time can strongly change the generalization produced by an otherwise similar weight update. [Paper](https://arxiv.org/abs/2510.04340?utm_source=chatgpt.com)

**Wichers et al. (2025), *Inoculation Prompting: Instructing LLMs to misbehave at train-time improves test-time alignment*.**
Independent development of the same basic idea, including reward-hacking and sycophancy settings. [Paper](https://arxiv.org/abs/2510.05024?utm_source=chatgpt.com)

**Imran & Shaikh (2026), *Inoculate or Reflect?***
Small independent experiment explicitly asking whether suppression and CRT-style repair leave a learned behavior differently recoverable. Interesting methodological inspiration, but not yet evidence I would build the central claim around. [Write-up](https://www.lesswrong.com/posts/LQK3yzsn8gts4tS7c/inoculate-or-reflect-two-training-interventions-under-1?utm_source=chatgpt.com)

**Beckers & Kindt (2017), *Memory Reconsolidation Interference as an Emerging Treatment for Emotional Disorders*.**
Good skeptical/critical review of the experimental reconsolidation literature and the difficulties translating it clinically. [Review](https://pmc.ncbi.nlm.nih.gov/articles/PMC5424072/?utm_source=chatgpt.com)

**Sinclair & Barense (2019), *Prediction Error and Memory Reactivation: How Incomplete Reminders Drive Reconsolidation*.**
Useful neuroscience overview of the idea that reactivation plus prediction error can enable updating rather than simple retrieval. [PubMed](https://pubmed.ncbi.nlm.nih.gov/31506189/?utm_source=chatgpt.com)

**Ecker (2018), *Clinical Translation of Memory Reconsolidation Research*.**
The Coherence Therapy-side account of reactivation, mismatch/juxtaposition and the proposed distinction between transformative and counteractive change. Useful for generating the analogy, but it should not be treated as independent evidence for the stronger clinical claims. [Paper](https://www.coherencetherapy.org/files/Ecker_2018_Clinical_Translation_of_Memory_Reconsolidation_Research.pdf?utm_source=chatgpt.com)

