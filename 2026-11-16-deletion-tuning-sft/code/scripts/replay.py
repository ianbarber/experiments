"""Print full game transcripts for a saved checkpoint (or the base model).

  python scripts/replay.py RUN STEP [POOL] [N] [OFFSET]
"""
import json, sys
from deltune import wordle
from deltune.common import load_tokenizer
from deltune.evaluate import Deployed

run, step = sys.argv[1], int(sys.argv[2])
pool = sys.argv[3] if len(sys.argv) > 3 else "seen"
n = int(sys.argv[4]) if len(sys.argv) > 4 else 8
off = int(sys.argv[5]) if len(sys.argv) > 5 else 0
tok = load_tokenizer()
dep = Deployed(run)
model = dep.load(step) if run != "base" else dep.model
games = wordle.play(model, tok, wordle.wordlists()[pool][off : off + n])
print(json.dumps(wordle.summarize(games)))
for g in games:
    print(wordle.transcript(g)); print()
