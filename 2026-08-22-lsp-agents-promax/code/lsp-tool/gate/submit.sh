#!/bin/bash
# Arm D submission wrapper: type-check acceptance gate, then the standard
# mini-swe-agent submission marker.
cd /testbed || exit 1
if [ ! -f patch.txt ]; then
  echo "submit: patch.txt not found. Create it first (Step 1), then run submit."
  exit 1
fi
/opt/lsp-tool/bin/python /usr/local/bin/typecheck_gate.py
rc=$?
if [ "$rc" -ne 0 ]; then
  exit 1
fi
echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt
