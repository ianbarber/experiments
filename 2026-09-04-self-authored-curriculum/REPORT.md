# Does a model get better by writing its own training tasks?

A model that writes its own training tasks, checks them by running them, and trains on its attempts is the smallest honest version of a self-improving loop. We built one on a single graphics card and measured it against a held-out set fixed in advance. **It did not improve.** The more useful findings are why the interim measurements said otherwise, and what the self-written curriculum actually did.

## The loop

Each iteration, Qwen3.5-9B proposed several hundred candidate tasks from a playbook. A gate ran each one: the task's own reference solution had to pass its hidden tests, an empty submission had to fail them, and deleting the tests had to not pass. Roughly a quarter to a third of candidates survived. The model then attempted the survivors eight times each, and a LoRA policy gradient trained it on those attempts. Tasks it solved every time or never were dropped; only the middle band trains.

Two things about the design matter for reading the results. The graded tests are copied into the container at verification time behind an integrity check, so the agent never has access to what scores it. And after each iteration the model rewrote its own playbook, themes and a few numeric thresholds, under schema validation and bounds it could not change. That second loop is what the control tests: at iteration 3 the run branched, and one arm continued with its task-writer frozen at the starting model and the starting playbook. Both arms then trained identically.

## How we measured

The endpoint was 283 terminal tasks written by stronger non-Qwen models from the same brief the trainee used, gated by the same executable pipeline, hashed and frozen before the first iteration. Endpoints, thresholds and the analysis were committed before iteration 0; seven amendments are dated in the [pre-registration](results/preregistration.md).

The final comparison evaluated the final model and the starting model **in one session, alternating in blocks of two seeds**, ten seeds each, paired by task. That design exists to remove differences in when and how each model was measured. It turned out to be the single most important decision in the experiment.

## What happened

| Comparison | Points | SE | Note |
|---|---:|---:|---|
| Final model vs starting model | **−1.39** | 0.96 | primary endpoint, same session, 95% CI −3.24 to +0.46 |
| Self-editing arm vs frozen-writer arm | **−2.19** | 0.95 | frozen arm ahead; control measured a day later |
| Starting model, re-measured after five days | +1.47 | 1.27 | inside the pre-declared ±3.5 band |
| Shuffled-reward control | −0.11 | 1.11 | same rollouts, rewards permuted |
| RFT vs policy gradient, identical rollouts | −0.67 | 1.74 | indistinguishable |
| Canary composite (IFEval + GSM8K) | 52.2 vs 51.0 | | unchanged |

Pass rates on a 40-task subset at 16 samples went the opposite way at the two ends: first-attempt accuracy fell from 58.4 to 54.1 while sixteen-sample accuracy rose from 89.7 to 92.5. On 40 tasks neither moves reliably, and it is the reverse of the sharpening reinforcement learning usually produces, so we record it without explaining it.

## Why the interim numbers were wrong

Partway through, the same model measured against the baseline recorded on the first day read **+0.58**, then **+1.01**. The same model against a baseline re-measured in the same session reads **−1.39**.

Almost all of that difference is the baseline. The starting model scored 57.30 on 7 September and 58.77 on 12 September, a drift of 1.47 points that sits comfortably inside the band we had declared as acceptable. It is not a bug and not a broken evaluation; it is ordinary variation in container timing, server state and serving precision across days.

The consequence is sharp. A loop like this one, evaluated the obvious way against a baseline you measured when you started, will report about a point and a half of improvement that does not exist, which is the same order as any real effect you might be hoping for. **Every comparison here except the endpoint spans sessions**, including the out-of-genre TerminalWorld figure of +1.98, so those should be read as upper bounds.

## What the self-written curriculum did

The proposer's behaviour was stable, and not in a useful way. After the first iteration, about a quarter of the tasks it wrote could never be solved in eight attempts, and that fraction held for the rest of the run. The frozen-writer arm produced the same fraction, so it is not a consequence of self-editing.

What grew was the pool. Nothing retired a task that failed completely, so unsolvable tasks accumulated and were re-attempted four times each every iteration: 15 of them after the first iteration, 171 after the last, until roughly a third of rollout compute was spent on tasks that had never been solved. Meanwhile the band of tasks that actually produce a gradient stayed between 40 and 73 regardless of whether the proposer wrote 87 tasks or 145. **The loop could write more tasks but not more useful ones.**

The self-editing itself had a measurable effect on task writing without reaching the model. The two arms' gate yields were identical at the branch and then separated: 31.5% for the self-editing arm in the final iteration against 21.1% for the frozen one, with the frozen arm rejecting more of its own candidates as near-duplicates as its pool filled. So the playbook edits kept the writer productive; they did not make the resulting model better.

One behaviour is worth recording for anyone building the same thing. Asked for 8 to 14 curriculum themes, keeping the good ones and adding what the failures suggested, the model returned all 16 it already had plus 2 more, three times, and each edit was rejected whole by a cap it could not change. Its numeric thresholds, by contrast, moved in both directions and finished below where they started. Given a list it appends; given a bounded number it explores.

## Reward hacking

Across 57,000 episodes the static pattern rules produced **no confirmed true positive**. Every quarantine we inspected was the agent writing its own scratch test script with an ordinary exit call in it. A model monitor reading trajectories found one genuine case: handed a diagnostic script and contradicted by it, the agent rewrote the script. The graded tests never invoke that script, so the reward was earned legitimately by the artefact the tests do read.

This is consistent with published results that an inaccessible verifier suppresses hacking, and it is not evidence that the model is honest. Our monitor is a 27B reading actions, and published comparisons put small monitors near 12% recall where frontier models reach 95%.

## What we would change

1. **Score candidate tasks by learnability, not difficulty** — the solve rate times one minus the solve rate, which the eight rollouts already give for free. A generator that maximises difficulty producing unsolvable work is the problem the environment-design literature was written about.
2. **Narrow the trainable band.** Ours trains on anything solved between one and seven times out of eight, the loosest in the literature; comparable work keeps roughly a quarter to three-quarters.
3. **Retire tasks that never succeed**, and re-test admission each iteration rather than once. Mutating a task that still teaches beats writing a fresh one.
4. **Train an order of magnitude longer before reading the curve.**
5. **Interleave the baseline with the final model, always.** This costs nothing and is the difference between a null and a false positive.

## Limitations and verdict

The run performs 54 optimiser steps where comparable agentic RL work uses a few hundred and the closest consumer-hardware study was still improving at 300. A flat curve at that budget does not distinguish a method that does not work from one that was not trained. Results are a single run with one seed per configuration, the recursion contrast compares a block-evaluated control against an interleaved main arm measured a day apart, and the evaluation tasks were model-written and machine-gated rather than human-reviewed.

The verdict is narrow and we would defend it: **at this scale, with these settings, a model writing and training on its own terminal tasks produced no measurable improvement, and the evaluation was sensitive enough to exclude gains above about three points.** The mechanism findings are the part worth carrying forward, because they are measured on tens of thousands of episodes rather than on the endpoint, and they say the loop's problem was never the training signal. It was that the curriculum stopped producing anything new to learn from, and nothing in the design noticed.
