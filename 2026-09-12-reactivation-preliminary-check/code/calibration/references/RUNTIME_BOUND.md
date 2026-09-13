# Runtime and storage bound before launch

The new allocation is capped at43,200 seconds, with work stopped180 seconds before that deadline for release/restoration. This is a separate authorized calibration run; the earlier50.21-minute preliminary allocation remains closed.

The search has priority over material collection. There are no softer per-phase time limits, throughput-driven recipe substitutions or restarts. The worker checks the remaining clock every two seconds; the wrapper and independent lease also enforce the deadline. A stage interrupted at the deadline cannot qualify, and its partial artifacts are retained. Completion is not guaranteed by the estimates below.

| Phase | Maximum actual model stages |
|---|---:|
| Both competent baselines: screen, present diagnosis, omitted target loss |6|
| Six four-pass initial trajectories and18 diagnoses |42|
| Pure-bad control, only if no initial candidate passes |6|
| Three exact-dose seed2718 replications |15|
| Separate paired-development samples for three candidates/two seeds |6|
| Two locked fresh attempts, competent/induced for both seeds |8|
| Material collection, first-failure principles, preservation attempts/principles for two candidates/two seeds |16|

The arithmetic upper envelope is99 stages,40 training passes,30,080 generated outputs and26,624 teacher-forced diagnostic rows. It is deliberately conservative: the pure-bad branch follows a grid with no passing candidate, so it cannot also reach confirmation/collection. The largest potentially successful path has93 stages and36 training passes. A negative path with the pure-bad diagnostic can have69 stages if all three selected doses are four passes.

The previous environment took roughly247 seconds per512-example training pass (32updates, including loading), and271–292 seconds per384 greedy outputs at batch8. The new mixed pool has1,024 examples and more target tokens. Microbatch16, greedy batch32 and fixed sampled batch16 improve batching; their actual end-to-end throughput is not yet established. A rough extrapolation puts the initial grid plus baselines around four to six hours, with replication and conditional collection consuming the remainder. Response lengths and loading overhead can change this substantially. Actual stage times will be reported instead of presenting the estimate as evidence of feasibility.

A prior training stage retained approximately115MiB of adapter data and229MiB of optimizer state. Forty such stages require approximately13.4GiB, plus small tokenization and output files. The filesystem had about22GiB free at the final readiness check. All checkpoints and optimizer states are retained; no old experiment files are deleted to make room. Storage exhaustion, if encountered, is an operational limitation, not model inability.
