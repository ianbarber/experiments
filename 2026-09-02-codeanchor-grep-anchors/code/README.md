Code for the arm-E / E3 anchors experiment. Only the files that differ from
`../../2026-08-22-lsp-agents-promax/code/` are included: the harness environment
(`agent/anchor_env.py`, `agent/anchor_runner.py`), arm configs, the runner with mode
`anchor`, the daemon's `anchor` op (`lsp-tool/src/lsp_tool/anchor.py` + the registering
changes in `daemon.py`/`cli.py`), and the analysis scripts. The rest of `lsp-tool`
(languages, protocol, render, install, Docker layer) is unchanged from the earlier entry.

2026-09-14 audit scripts (`analysis/`): `anchor_replay_local.py` replays every anchored grep
of a run against a clean local Pyrefly on repos checked out at `base_commit` (venv with this
`lsp-tool`, `serena-agent==1.7.0`, `pyrefly==1.2.0`; `agent/` on PYTHONPATH; PYTHONPATH must
not contain the clone directory — that is the bug being measured); `anchor_missed_audit.py`
diffs it against the in-image replay, re-runs the missed-file classification under both, and
checks for each missed gold file whether any resolving patch (all runs in the runs directory)
omitted it. `results/missed_audit_2026-09-14.txt` is its output.
`analysis/anchor_failure_modes.py` (same audit) measures recall against the files the tests
need and labels every failing episode of E and A8 by reading the task statement against the
failing assertion; the labels and their evidence quotes are in the script, the output is
`results/failure_modes_2026-09-14.txt`.
