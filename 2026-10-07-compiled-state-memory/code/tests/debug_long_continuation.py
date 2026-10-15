"""Locate the long-continuation divergence: sweep continuation length and path."""
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
full, st = lm.prefill_full(mem_ids)
for L in [24, 63, 64, 65, 128, 256, 512, 1024, 2031]:
    s = torch.tensor([prose[:L]], device=lm.device)
    one = lm.logits(lm.forward(torch.cat([mem_ids, s], 1), lm.new_cache())[:, -L:])[0]
    split = lm.logits(lm.forward(s, copy.deepcopy(full)))[0]
    fresh = lm.logits(lm.forward(s, lm.new_cache()))[0]
    zinj = lm.logits(lm.forward(s, lm.cache_from_state(None)))[0]
    k = kl(one, split)
    print(f"L={L:5d} one-vs-split maxKL {k.max():.4f} (first bad pos {int((k > 0.05).nonzero()[0]) if (k > 0.05).any() else -1}) "
          f"| fresh-vs-zeroinj maxKL {kl(fresh, zinj).max():.4f}", flush=True)
