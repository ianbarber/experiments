"""Nominal per-head decay half-lives of the Gated DeltaNet layers (CPU only).

GDN log-decay per token is g = -exp(A_log) * softplus(a + dt_bias), where a = in_proj_a(x)
is data dependent. With a = 0 this gives a *nominal* half-life ln2/|g| per head. Real decay
is input dependent (gating can hold or flush state), so this is a prior, not a measurement.

Usage: CKPT=<snapshot dir with model.safetensors.index.json> uv run --with safetensors \
       --with numpy --with torch python gdn_nominal_halflife.py
"""
import json
import os

import numpy as np
from safetensors import safe_open

D = os.environ["CKPT"].rstrip("/") + "/"
wm = json.load(open(D + "model.safetensors.index.json"))["weight_map"]
layers = sorted({int(k.split(".")[3]) for k in wm if k.endswith("linear_attn.A_log")})
rows = []
for L in layers:
    p = f"model.language_model.layers.{L}.linear_attn."
    vals = {}
    for n in ["A_log", "dt_bias"]:
        with safe_open(D + wm[p + n], "pt") as f:
            vals[n] = f.get_tensor(p + n).float().numpy()
    g0 = -np.exp(vals["A_log"]) * np.log1p(np.exp(vals["dt_bias"]))
    rows.append((L, np.log(2) / np.abs(g0)))
hl = np.concatenate([r[1] for r in rows])
print(f"GDN layers {len(rows)}, heads/layer {rows[0][1].size}, total heads {hl.size}")
print("half-life quantiles (tokens):",
      {q: round(float(np.quantile(hl, q)), 1) for q in [0.1, 0.25, 0.5, 0.75, 0.9, 0.99]},
      "max", round(float(hl.max()), 1))
for thr in [10, 100, 1000, 10000]:
    print(f"frac heads half-life > {thr}: {np.mean(hl > thr):.3f}")
print("layer  median  max  n(>1000)")
for L, h in rows:
    print(L, round(float(np.median(h)), 1), round(float(h.max()), 1), int((h > 1000).sum()))
