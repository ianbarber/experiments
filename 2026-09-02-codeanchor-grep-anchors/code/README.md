Code for the arm-E / E3 anchors experiment. Only the files that differ from
`../../2026-08-22-lsp-agents-promax/code/` are included: the harness environment
(`agent/anchor_env.py`, `agent/anchor_runner.py`), arm configs, the runner with mode
`anchor`, the daemon's `anchor` op (`lsp-tool/src/lsp_tool/anchor.py` + the registering
changes in `daemon.py`/`cli.py`), and the analysis scripts. The rest of `lsp-tool`
(languages, protocol, render, install, Docker layer) is unchanged from the earlier entry.
