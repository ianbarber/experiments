"""Synthetic check of the seed fit (statelm/kvseed.py; LABNOTES 2026-10-10, Stage 1). No model.

  python tests/check_kvseed_fit.py

Clustered unit-RMS keys (like post-k_norm keys), random values, queries drawn near random keys.
F1  the fitted bias lowers held-out log-mass error below the log(cluster size) start
F2  the fitted values do not raise held-out output error above the cluster-mean values (2% slack)
F3  held-out and fit errors are close (no overfit at J = 4N)
F4  rounding the bias to bf16 (as the attention mask does) changes held-out errors by < 10%
"""
import json
import time

import torch

import sys
sys.path.insert(0, ".")
from statelm.kvseed import fit_head, seed_metrics, full_targets  # noqa: E402


def main():
    torch.manual_seed(42)
    dev = "cuda"
    T, d, n, J, s = 32768, 256, 2048, 8192, 256 ** -0.5
    centers = torch.randn(512, d, device=dev)
    K = centers[torch.randint(0, 512, (T,), device=dev)] + 0.6 * torch.randn(T, d, device=dev)
    K = K / K.pow(2).mean(1, keepdim=True).sqrt()  # unit RMS per dim, like post-k_norm keys
    W = torch.randn(d, d, device=dev) / d ** 0.5
    V = K @ W + 0.5 * torch.randn(T, d, device=dev)  # values partly predictable from keys
    Q = 0.4 * K[torch.randint(0, T, (J,), device=dev)] + 0.4 * torch.randn(J, d, device=dev)
    t = time.time()
    C, b, U, _, _, st = fit_head(K, V, Q, n, s)
    st["seconds"] = round(time.time() - t, 2)
    g = torch.Generator(device=dev).manual_seed(42)
    perm = torch.randperm(J, generator=g, device=dev)
    Qh = Q[perm[:int(0.1 * J)]]  # the same held split as fit_head
    mh, yh = full_targets(Qh, K, V, s)
    st["fit_bf16_bias"] = seed_metrics(Qh, C, b.bfloat16().float(), U, mh, yh, s)
    st["F1"] = st["bias_only"]["logmass_rmse"] < st["init"]["logmass_rmse"]
    st["F2"] = st["fit"]["out_rel"] <= 1.02 * st["bias_only"]["out_rel"]
    st["F3"] = st["fit"]["out_rel"] < 1.5 * st["fit_train"]["out_rel"] + 0.02
    st["F4"] = abs(st["fit_bf16_bias"]["out_rel"] - st["fit"]["out_rel"]) < 0.1 * st["fit"]["out_rel"] + 1e-3
    st["PASS"] = bool(st["F1"] and st["F2"] and st["F3"] and st["F4"])
    print(json.dumps(st, indent=1))


if __name__ == "__main__":
    main()
