# Lab notes

Condensed from a contemporaneous notebook kept during the run; entries were written as the
work happened, and this file compresses them without reordering. Dead ends are included
where they changed the design or the reading of a result.

## 4–6 September: design and plumbing

Literature review, then a written design: a 9B trainee on one card, Harbor-format terminal
tasks, an executable gate, a LoRA policy gradient, and the model editing its own
task-writing playbook. Scope fixed at two levels, training and curriculum; changing the
loop's own code was ruled out.

A 4B pilot ran end to end and exposed the first real bug: the harness called the model over
synchronous HTTP inside an async loop, so concurrent trials queued behind each other at 44
episodes an hour. Moving the call to a thread took it to 216. Docker's default address pool
then turned out to hold fewer networks than the concurrency needed, and Harbor takes two per
trial, so a fifth of episodes were failing for reasons that looked like task bugs.

Held-out tasks were authored by stronger non-Qwen models through their CLIs, gated by the
same pipeline, tiered, and frozen by hash. The first freeze came in under target at 177
tasks with more than half of them saturated, which triggered a pre-declared remedy: a
supplementary harder set, 106 tasks, hashed separately. The endpoint became the union.

## 7 September: pre-registration and launch

Baselines measured for the starting model on every evaluation set, the pre-registration
frozen with its minimum detectable effect computed from those baselines, and the main run
launched. Seven iterations planned, with a branch at iteration 3 for the frozen-writer
control.

## 8 September: the merge that ate the update

Iteration 0 trained cleanly, and then the pre-registered merge audit failed: merging the
trained adapter into bfloat16 weights retained only a third of its delta, because most
entries fell below half a representable step of the base weight. Merging in float32 and
saving float16 retained 85%. We switched serving precision, re-merged the lineage head, and
recorded the amendment. This is worth flagging to anyone doing the same thing: nothing in
the loss curve shows it, and the audit that caught it existed only because it was
pre-registered.

Also that day: the dev-set regression gate was found to have been a no-op for one iteration,
comparing against a baseline stored under a different label and auto-accepting on no data.
Fixed; the recomputed verdict was unchanged.

## 9 September: the watchdog earns itself

An orchestrator crash left the machine idle for six hours overnight, so restart supervision
moved out of the session and into cron. It paid for itself the following night, restarting
the run four minutes after a session teardown killed it mid-rollout.

The gate was the expensive step at nearly six hours an iteration. Doubling the container
concurrency moved it by four minutes, because the cost was not containers: the repair round
was rewriting three hundred failed candidates one model call at a time. Parallelising it
took the gate to under two hours.

## 10 September: a near-miss worth recording

Before the branch ran, a check of the branch code found that both runs would serve their
policies from the same fixed path on disk, with no record of whose weights were there. The
arms alternate, so the control would have generated its rollouts on the main arm's weights
and vice versa, with no error and no odd log line, and the key secondary endpoint would have
been meaningless. Every merge now stamps the path with the adapter it came from and serving
re-merges when the stamp belongs to the other run. The guard fired on its first real use.

Separately, three ways the self-editing channel silently did not work: the failure analysis
never saw gate rejections, so the proposer could not learn why a fifth of its tasks failed
to build; two of its four editable thresholds were written every round and read by nothing;
and its output budget was fixed while the analysis it was asked to perform grew, until one
round's edits were truncated away entirely. Each looked healthy in the logs.

## 11–12 September: claims that did not survive

Three mid-run readings were retracted after more data arrived, which is the main reason this
file exists.

The frontier of never-solved tasks grows monotonically, 15 to 171, and we attributed it to
the proposer overshooting. The frozen-writer arm produced the same growth, and per-iteration
rates are flat at about a quarter in both arms. It is a retention curve, not a proposer
curve: nothing retires failed tasks.

The gap between arms on the dev set reached 4.67 points with a one-sided p of 0.005, and we
wrote two paragraphs of mechanism around it. The control recovered at its final checkpoint
and the gap fell to 1.56 with p of 0.18. The contrast had been computed at whichever
checkpoint was latest, and across seven checkpoints of a wandering series something will
look significant.

"The improver only ever adds" held for its list of curriculum themes and not for its numeric
thresholds, which finished below where they started. The first three rounds happened to be a
run of increases.

## 13–14 September: endpoint

Ten alternating blocks, 5,660 episodes, the final model and the starting model interleaved
through one session. The endpoint came out at −1.39 points, and the interim reads of +0.58
and +1.01 turned out to be mostly the baseline moving 1.47 points between the first day and
the last. The control arm finished ahead of the self-editing arm.

Controls ran last: shuffled rewards moved the model −0.11 points, and the objective ablation
was indistinguishable. The random-reward figure initially read +2.15 because it had been
paired against the opening baseline, which is the same mistake the interims made, caught
this time because we had just measured its size.
