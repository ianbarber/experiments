"""Re-run HumanEval alone for one checkpoint, keeping the raw completions for reading.

  python scripts/humaneval_read.py RUN STEP      (RUN may be `base`, STEP 0)
Writes runs/RUN/humaneval_step{STEP}/ and leaves the main suite files alone.
"""
import json, sys
from deltune.common import RUNS_DIR, load_tokenizer
from deltune.evaluate import Deployed
from deltune.suite import run_suite

run, step = sys.argv[1], int(sys.argv[2])
dep = Deployed(run)
model = dep.load(step) if run != "base" else dep.model
out = run_suite(model, load_tokenizer(), RUNS_DIR / run / f"humaneval_step{step}", tasks=["humaneval"])
print("RESULT", json.dumps({"run": run, "step": step, "humaneval": out["scores"]["humaneval"]}))
