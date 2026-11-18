"""Base model under other decoding settings, to check that zero wins is not a harness artifact."""
import random, json, time
import torch
from deltune.common import load_model, load_tokenizer, dump_json
from deltune import wordle

wl = wordle.wordlists()
rng = random.Random(0)
secrets = rng.sample(wl["seen"], 12) + rng.sample(wl["heldout_val"], 12)
tok, model = load_tokenizer(), load_model()
vendor = dict(do_sample=True, temperature=0.1, top_k=50, repetition_penalty=1.1)
configs = {
    "greedy_think_4096": dict(max_new_tokens=4096),
    "greedy_nothink_1024": dict(max_new_tokens=1024, enable_thinking=False),
    "vendor_sampling_think_4096": dict(max_new_tokens=4096, gen_kwargs=vendor),
}
out = {}
for name, kw in configs.items():
    torch.manual_seed(0)
    t = time.time()
    games = wordle.play(model, tok, secrets, batch_size=24, **kw)
    s = wordle.summarize(games)
    s["seconds"] = round(time.time() - t, 1)
    out[name] = s
    print(name, json.dumps(s))
    for g in games[:1] + games[12:13]:
        print(wordle.transcript(g)[:2500]); print()
dump_json(out, "../results/base_wordle_sensitivity.json")
