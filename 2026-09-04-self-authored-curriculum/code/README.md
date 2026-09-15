# Code

The loop, the harness, the task gate and the analysis. Paths below are relative to this
directory.

## Layout

| Path | What it is |
|---|---|
| `rsi/loop.py` | The nine-step iteration controller: propose, gate, roll out, tier, analyse, audit, update, dev gate, evaluate. Pause, resume and per-step rerun |
| `rsi/harness/` | The agent: one bash tool, single-action turns, compaction, per-turn records |
| `rsi/tasks/gate.py` | The executable gate: static checks, reference solution must pass, empty submission must fail, deleting tests must not pass |
| `rsi/improver/` | The proposer and the failure analysis that edits the playbook, themes and thresholds |
| `rsi/train/` | LoRA policy gradient, RLOO advantages, chunked float32 log-probabilities with out-of-memory backoff |
| `rsi/audit.py` | Static quarantine rules and the trajectory monitor |
| `rsi/evaluate.py`, `rsi/heldout.py`, `rsi/probe.py` | Paired evaluation, held-out authoring and freezing, the memorisation probe |
| `improver/` | The starting playbook, themes and bounded thresholds (H_0) |
| `configs/main-9b.json` | The run configuration |
| `scripts/run-main.sh` | Orchestration: iterations, the branch, the final evaluations |
| `scripts/analyse-run.py` | The pre-registered analysis; writes `results.json` and `results.md` |

## Environment

Three virtual environments, because serving, training and orchestration have incompatible
pins: `venvs/serve` (vLLM 0.28), `venvs/train` (PyTorch, transformers, the fused kernels),
`venvs/loop` (Harbor, analysis). Docker is required, with an address pool wide enough for
two networks per concurrent trial. A CUDA card with at least 32 GB; the run used one
RTX 5090.

Paths come from the environment, with defaults under the home directory:

```bash
export RSI_RUNS_DIR=$PWD/../runs          # run state, episodes, adapters (override)
export RSI_MODELS_DIR=$PWD/../models      # base weights and the merged serving path
export RSI_DATASETS_DIR=$PWD/../datasets  # seed task collections
export RSI_MONITOR_URL=http://dgx-spark:8888/v1   # optional trajectory monitor
```

## Running it

```bash
export PYTHONPATH=$PWD
venvs/loop/bin/python -m rsi.cli import-seeds --run main     # seed pool
venvs/loop/bin/python -m rsi.cli gate --run main --repair    # gate the seeds
venvs/loop/bin/python -m rsi.cli heldout author --run main   # held-out set, then gate/tier/freeze
venvs/loop/bin/python -m rsi.cli eval heldout --run main --policy M0 --seeds 5 --tag start
scripts/run-main.sh                                          # the run itself, detached
venvs/loop/bin/python scripts/analyse-run.py --run main      # the analysis
```

`rsi.cli loop pause|resume|status|rerun` controls a running loop between steps. The
orchestrator is resumable: every step is idempotent through the run's state file and its
SQLite log, so killing it and restarting loses at most the step in flight.

## Reproducing the measurement, not just the run

The one thing to copy if you copy nothing else: evaluate the final model and the starting
model **in the same session, alternating in seed blocks**, and pair by task. Measuring the
baseline once at the start and comparing against it later produced a spurious gain of about
1.5 points here, which is larger than any effect this kind of loop is likely to produce at
small scale.
