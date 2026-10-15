"""Which path is wrong for long continuations? Compare one-pass and split against token-by-token decode."""
import copy, sys, torch
sys.path.insert(0, ".")
from statelm.core import LM
from statelm.prompts import compile_text
from statelm.synth import make_memory
lm = LM(sys.argv[1])
mem_ids = lm.ids(compile_text(lm.tok, make_memory("dog", "Biscuit", 60, 40, seed=42)))
prose = lm.tok.encode(open("/work/data/raw/text/qwen38_model_card.md").read(), add_special_tokens=False)
def kl(a, b):
    pa, pb = torch.log_softmax(a.float(), -1), torch.log_softmax(b.float(), -1)
    return (pa.exp() * (pa - pb)).sum(-1)
L = int(sys.argv[2]) if len(sys.argv) > 2 else 256
s = torch.tensor([prose[:L]], device=lm.device)
print("tokens 125-140:", repr(lm.tok.decode(prose[125:141])))
one = lm.logits(lm.forward(torch.cat([mem_ids, s], 1), lm.new_cache())[:, -L:])[0]
full, st = lm.prefill_full(mem_ids)
split = lm.logits(lm.forward(s, copy.deepcopy(full)))[0]
c = copy.deepcopy(full)
dec = torch.cat([lm.logits(lm.forward(s[:, i:i + 1], c))[0] for i in range(L)])
# fresh (no memory) one-pass vs fresh token-by-token: is long one-pass itself accurate?
fone = lm.logits(lm.forward(s, lm.new_cache()))[0]
cf = lm.new_cache(); lm.forward(s[:, :1], cf)
fdec = torch.cat([lm.logits(lm.forward(s[:, :1], lm.new_cache()))[0]] + [lm.logits(lm.forward(s[:, i:i + 1], cf))[0] for i in range(1, L)])
for name, a, b in [("one vs decode", one, dec), ("split vs decode", split, dec), ("one vs split", one, split),
                   ("fresh one vs fresh decode", fone, fdec)]:
    k = kl(a, b)
    q = torch.quantile(k.float(), torch.tensor([0.5, 0.99], device=k.device))
    print(f"{name:28s} maxKL {k.max():.4f} meanKL {k.mean():.5f} p50 {q[0]:.5f} p99 {q[1]:.4f} argmax_agree {(a.argmax(-1)==b.argmax(-1)).float().mean():.3f} worst pos {int(k.argmax())} n>1nat {int((k>1).sum())}")
