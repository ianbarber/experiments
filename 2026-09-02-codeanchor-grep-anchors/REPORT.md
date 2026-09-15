# CodeAnchor-style anchors on grep output

Follow-up to [LSP tools for a local coding agent](../2026-08-22-lsp-agents-promax/).
That study found the model never adopts `lsp refs` while its dominant failure is
incomplete refactoring. This one asks whether injecting the same facts passively
— as an addendum to grep observations, prompt unchanged — changes resolve rate,
cost, or variance.

It does not. Experiment closed; no further arms.

A post-hoc audit (addendum, 2026-09-14) also retires the "incomplete
refactoring" reading: both arms find the files the tests need (recall 0.96 vs
0.94), and the task statements say what the tests check — django's quotes the
error message the failing test asserts, verbatim. The dominant failure in both
arms is partial compliance with a long, explicit specification.

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

**Why (corrected 2026-09-14, see the addendum).** Of the round-1 misses: 41%
non-Python (docs/yaml/config — unreachable from a Python reference graph); 32%
Python source never referenced by any symbol the agent chose to search; 12%
Python source a working, uncapped server would have named (4% hidden by my
display cap, 8% hidden by a reference-server bug in two rollout images); 10%
created/deleted by the gold patch; 6% shown and ignored. Passive injection is
bounded by the questions the agent asks — but 65 of the 69 missed files were
not required by the graded tests, so the gap is mostly not a correctness gap.

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

## Addendum (2026-09-14): a broken reference server, and which misses mattered

Two corrections after the run closed, prompted by asking why a language server
would miss a Python file that *uses* a class it was asked about. No new runs.

**The reference server under-reported in two of 29 rollout images.** The base
images for lerobot-2808 and transformers-38332 ship `PYTHONPATH=/testbed:`.
Pyrefly 1.2.0 puts PYTHONPATH entries on its site-package path ahead of the
import root it infers for a `src/` layout, so a file the server opens is named
`src.lerobot....` while every other file imports it as `lerobot....`: one file,
two module handles. Find-references walks reverse imports of the definition's
handle, so only files reached through *relative* imports from the opened file
come back. In lerobot-2808 the agent was told `OpenCVCameraConfig` is "used by
3 sites in 2 files" (57 sites in 20 files) and that `DeviceNotConnectedError` is
"never referenced elsewhere" (61 sites in 14 files). Verified three ways: the
same anchor code against a clean server on the repos at base commit returns the
full lists, and setting `PYTHONPATH=<repo>:` alone reproduces the truncation;
inside the actual image on worker-a, starting the daemon with PYTHONPATH unset
turns 2 files into the full set; and a replay of every anchored grep of E-r1
against the clean server (`code/analysis/anchor_replay_local.py`) matches the
in-image replay file-for-file on 27 instances and diverges only on the two
images that combine `PYTHONPATH=/testbed` with a `src/` layout (seven images set
it; flat-layout repos and `PYTHONPATH=/testbed/src` are unaffected). Versions
were identical (pyrefly 1.2.0, serena-agent 1.7.0). E, E3 and any `lsp refs`
call in those two images were fed truncated "used by" lists throughout. Fix:
start the daemon with `env -u PYTHONPATH` (or `PYTHONPATH=/testbed/src`). Not
rerun: the two instances are one tie (lerobot, 1/2 per arm) and one E win
(transformers-38332), so correct anchors could at most move the paired count
from 2-vs-3 to 3-vs-3.

**Corrected classification of the 69 round-1 misses**
(`results/missed_audit_2026-09-14.txt`):

| Missed gold files (69) | Reported | Corrected |
|---|---:|---:|
| Non-Python (docs, yaml, config) | 28 (41%) | 28 (41%) |
| Python, never named by any anchor | 27 (39%) | 22 (32%) |
| Python, named only by an uncapped, working server | 3 (4%) | 8 (12%) |
| Python, shown in a capped addendum and ignored | 4 (6%) | 4 (6%) |
| Created or deleted by the gold patch | 7 (10%) | 7 (10%) |

The 22 that remain genuinely use no symbol the agent searched: seven lerobot
example scripts whose gold change wraps loops in try/finally around a *robot's*
disconnect; five optuna and five ragas files carrying unrelated changes squashed
into the gold patch (an intersphinx entry, a scipy import rename, `# type:
ignore` comments); and five singletons (a string-keyed site registry, two files
where the gold *introduces* the call, a docstring rewrite, a same-named config
field read through OmegaConf).

**Which misses mattered.** Across every run on the NAS (all arms and rounds, up
to 21 resolving episodes per instance), 65 of the 69 files were omitted by at
least one patch that resolved the instance, so the graded tests do not require
them. All 28 non-Python files are in that group, as are 21 of the 22 "never
named" files. The four never omitted by a resolving patch: albumentations-2495's
three mixing/composition files (shown in an addendum and ignored; resolved 4
times, always with them) and dspy-9047's `evaluate.py` (resolved 5 times, always
with it; the gold adds `toDict()` calls there, so nothing existed to reference).
"Not required by the tests" is not "not part of the refactor": the lerobot
try/finally wraps are the change's intent, just untested, while the optuna and
ragas files are unrelated. Either way, the edit-recall-vs-gold metric behind the
"incomplete refactor" reading counts docs, untested intent and squashed noise
alike. Against the graded tests, the anchors' relevant miss in this study is
four files, three of which the anchors showed. The stage-1 study's
incomplete-refactoring diagnosis rests on the same metric and was not
re-audited here.

**What actually fails, and what to call it.** Measured against the files the
tests need (the intersection of every resolving patch's files; median 4 per
instance against 6 gold files), the agent is not failing to find files: recall
is 0.96 (E) and 0.94 (A8) against needed files, versus 0.68 against gold files
for both arms. Each of the 34 failing episodes was then labelled by reading the
task statement against the failing assertion
(`code/analysis/anchor_failure_modes.py`, evidence quotes in the script;
`results/failure_modes_2026-09-14.txt`):

| Failing episodes, 25 instances × 2 rounds | E | A8 |
|---|---:|---:|
| Total | 18 | 16 |
| Localization miss: a test-needed file never edited, and that decided the outcome | 4 | 5 |
| Unmet stated requirement: the statement states it, the patch does not satisfy it | 8 | 5 |
| Requirement not derivable from the statement: a name, signature, internal or reading only the tests pin | 4 | 4 |
| External knowledge: a third-party site's live JSON | 2 | 2 |

ProMax statements are long and explicit, so these are not hidden-test
surprises. Django's statement says the duplicate case must raise
`"Partial 'testing-name' is already defined in the 'template_name' template."`
(names substituted); the failing assertion in both arms, both rounds, is that
`"Partial 'duplicate' is already defined in the 'template.html' template."`
was not raised. Lerobot's says "if the buffered frame is older than this
threshold, it should raise a `TimeoutError`"; the failing test is `DID NOT
RAISE TimeoutError`. Optuna's names `inverse_squared_lengthscales`; the tests
fail with `AttributeError` on that name. The dominant mode is therefore not
incomplete refactoring but **partial compliance with a long, explicit
specification**: the patch satisfies most of the enumerated requirements and
drops or mis-implements one, and the test for that one fails, on the same
instances in both arms. That is a model failure, and one no "used by" fact
addresses. The four episodes per arm where the tests pin something the
statement does not say (albumentations-2495's `filter_valid_metadata(data)`
signature, adk's Mock without `_invocation_context`, langchain's
lone-ToolMessage reading) are the benchmark's, in the sense SWE-bench
Verified's annotators called "tests too specific". Anchors did halve the
episodes that never edited a needed file (4 vs 8 of 50), which is the
localization effect they were meant to have; it did not convert into
resolutions because the same instances then fail on a stated requirement.
Lerobot is the clean pair: the control missed a camera file and failed the
TimeoutError test, arm E edited that file and failed the same test.

## Verdict

Correctness: no. Efficiency: no. Variance: unproven. The agent acts on the
signal when it points at the right place, and anchors halve the episodes that
never edit a test-needed file (4 vs 8 of 50) without converting one into a
resolution. The gold files both arms miss are mostly docs, squashed unrelated
changes and untested parts of the refactor that no resolving patch needed
(addendum). The dominant failure in both arms is partial compliance with a
long, explicit specification, a model failure that structural facts do not
touch. That is a finding about passive injection and about the recall metric,
not a prompt to try a fourth arm.

No grep-off, no forced `lsp refs`, no further placement variants. Those were
open at the end of the run; they are retracted.
