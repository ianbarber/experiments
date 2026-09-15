# CodeAnchor-style anchors on grep output

Follow-up to [LSP tools for a local coding agent](../2026-08-22-lsp-agents-promax/).
That study found the model never adopts `lsp refs` while its dominant failure is
incomplete refactoring. This one asks whether injecting the same facts passively
— as an addendum to grep observations, prompt unchanged — changes resolve rate,
cost, or variance.

It does not. Experiment closed; no further arms.

## Setup

CodeAnchor (arXiv 2606.26979) writes static "used by" comments into source so a
grep-first agent sees structure without adopting a new tool. In-file comments
would leak into graded `git diff`s here, so the facts come from the warm
language server and are appended to grep observations instead.

- **Arm E:** prompt byte-identical to the stage-1 baseline. Every grep-family
  observation of `.py` hits gets a definition-site / used-by / "referenced but
  not in this grep" addendum. Caps: 4 symbols, 6 not-shown files, 2,800 chars
  (a deviation from the paper, audited below).
- **Arm A8:** the same prompt with `container_timeout: 8h` — same-period
  control, removing the 2h-wall artifact of stage 1.
- **Arm E3:** E plus definition-site tags on file views (`cat`/`sed -n`/`head`),
  uncapped except for the harness 10k observation limit — the paper's own
  trigger surface.

Two rounds, hosts swapped, golden-valid python intersection, instance-level
sign tests. Model, sampling, and scaffold as in the LSP study.

## Results

**E vs A8, two rounds, 25 shared golden-valid instances:**

| | E | A8 |
|---|---:|---:|
| Resolved | 32/50 (64%) | 34/50 (68%) |
| Paired | better 2 | better 3; 20 tied; p = 1.0 |

Paired per-instance ratios: tokens 0.95 (p = 0.76), wall 1.03 (p = 0.40),
steps 0.93 (p = 0.07). Raw totals favour E because of two outlier control
episodes, not a saving. ~7% fewer steps is cancelled by ~8% higher per-step
latency from the longer context. Edit recall identical (0.667 vs 0.666).

**The mechanism works.** Half of greps anchored, ≈4% of prompt tokens, 5.6 s
per episode, zero harness failures. Of 99 flagged gold files, 99 were opened
and 92 patched. Both arms still miss the same 64 gold files.

**Why.** Of the round-1 misses: 41% non-Python (docs/yaml/config — unreachable
from a Python reference graph); 39% Python source never referenced by any
symbol the agent chose to search; 10% created/deleted by the gold patch; 6%
shown and ignored; **4% hidden by the display cap**. Passive injection is
bounded by the questions the agent asks.

**Cap counterfactual.** Uncapped replay of every anchored grep in E-r1 (256
observations) named 3.6× more files, and hid only 3 of 69 jointly-missed gold
files. The caps are a real deviation from the paper and do not explain the
null.

**E3 (definition-site, uncapped):** 32/50 vs A8 34/50, paired 1-vs-4 (p =
0.375). Fewer steps (−2.5, p = 0.043) but +0.22 h wall (p = 0.015) and +36%
median tokens. Only 9 of the 64 jointly-missed gold files were ever named.
More exposure is a net cost on a local server.

**Variance rerun.** 7 episodes per instance on the five largest-gap instances.
Level equal. Control produced the runaway episodes (three 300-step-cap hits
vs none) at equal typical cost; spread reduction not significant (bootstrap
CI 0.42–1.57); wall tail not shorter locally. Variance win plausible,
unproven.

## Verdict

Correctness: no. Efficiency: no. Variance: unproven. The agent acts on the
signal when it points at the right place, and the incomplete-refactor gap is
untouched because the agent does not search the symbols that would reach the
missing files. That is a finding about passive injection, not a prompt to
try a fourth arm.

No grep-off, no forced `lsp refs`, no further placement variants. Those were
open at the end of the run; they are retracted.
