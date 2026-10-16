"""Steady-state per-forward cost on the HF path: each length run 3x, report all timings.
python diagnostics/overhead.py --model M --out /work/results/audit/overhead_<tag>.json"""
import argparse, json, sys, time
import torch
sys.path.insert(0, ".")
from statelm.core import LM

ap = argparse.ArgumentParser(); ap.add_argument("--model"); ap.add_argument("--out"); a = ap.parse_args()
lm = LM(a.model)
base = lm.tok.encode(open("/work/data/raw/text/qwen38_model_card.md").read(), add_special_tokens=False)
res = {}
st = None
for L in [32, 256, 1024, 2048, 4096, 8192]:
    ids = torch.tensor([(base * 4)[:L]], device=lm.device)
    ts = []
    for _ in range(3):
        torch.cuda.synchronize(); t = time.time()
        st = lm.compile(ids)
        torch.cuda.synchronize(); ts.append(time.time() - t)
    res[f"compile_{L}"] = ts
    print(L, [round(x, 2) for x in ts], flush=True)
sess = torch.tensor([base[:30]], device=lm.device)
for name, fn in [("C_session30", lambda: lm.forward(sess, lm.cache_from_state(st))),
                 ("A_session30", lambda: lm.forward(sess, lm.cache_from_state(None))),
                 ("cache_from_state", lambda: lm.cache_from_state(st))]:
    ts = []
    for _ in range(5):
        torch.cuda.synchronize(); t = time.time(); fn(); torch.cuda.synchronize(); ts.append(time.time() - t)
    res[name] = ts
    print(name, [round(x, 3) for x in ts], flush=True)
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
    lm.compile(torch.tensor([(base * 4)[:2048]], device=lm.device)); torch.cuda.synchronize()
tab = prof.key_averages().table(sort_by="cuda_time_total", row_limit=15)
print(tab)
res["profile_2048"] = tab
json.dump(res, open(a.out, "w"), indent=1)
