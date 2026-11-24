"""Keep-table attention for RoPE models.

Every condition of the experiment is a boolean table M[pair, distance]: pair l's part of the query-key
score is kept at causal distance d = i - j iff M[l, d]. Keys at a distance where no pair is kept leave the
softmax (same convention as the HiSA reference implementation).

HF Llama layout: pair l = features (l, l + P), theta_l = base ** (-l / P), P = head_dim / 2.

Two implementations of the same definition:
  masked_scores_slow  one rank-2 product per pair (the sketch in the plan); used only in tests
  TableAttention      blocked: for each block distance, pairs whose keep value is constant over the block
                      go through one matmul on a feature subset, only pairs with an edge inside the block
                      are done elementwise. Exact, and about the cost of dense attention for window tables.
"""

from __future__ import annotations

import math

import numpy as np
import torch

# ----------------------------------------------------------------------------- tables


def windows(theta: torch.Tensor, angle_deg: float) -> np.ndarray:
    """w_l = Phi / theta_l in tokens (float64)."""
    return math.radians(angle_deg) / theta.double().cpu().numpy()


def window_table(w: np.ndarray, T: int) -> torch.Tensor:
    """M[l, d] = d < w_l."""
    d = np.arange(T)[None, :]
    return torch.from_numpy(d < w[:, None])


def band_table(theta: torch.Tensor, lo_deg: float, hi_deg: float, T: int) -> torch.Tensor:
    """Drop only the parts whose rotation is in [lo, hi)."""
    lo, hi = windows(theta, lo_deg), (windows(theta, hi_deg) if math.isfinite(hi_deg) else None)
    d = np.arange(T)[None, :]
    drop = d >= lo[:, None]
    if hi is not None:
        drop &= d < hi[:, None]
    return torch.from_numpy(~drop)


BANDS = {"band-90-180": (90, 180), "band-180-360": (180, 360), "band-360-720": (360, 720),
         "band-720+": (720, math.inf)}


def make_table(name: str, theta: torch.Tensor, T: int) -> torch.Tensor:
    """Conditions:
      dense                 keep everything
      hisa-<deg>            d * theta_l < deg
      band-a-b, band-720+   drop only that rotation band
      pair-<l>-<deg>        only pair l is cut, beyond <deg> degrees (diagnostic)
      ctrl-slow             the hisa-90 windows assigned to the pairs in reverse order (slowest pair gets the
                            shortest window): same number of pairs dropped at every distance, from the slow end
      ctrl-perm-s<seed>     the hisa-90 windows assigned to the pairs by a seeded random permutation
      ctrl-iid-s<seed>      at every distance, the hisa-90 number of dropped pairs, drawn independently
    """
    P = theta.numel()
    if name == "dense":
        return torch.ones(P, T, dtype=torch.bool)
    if name.startswith("hisa-"):
        return window_table(windows(theta, float(name[5:])), T)
    if name in BANDS:
        return band_table(theta, *BANDS[name], T)
    w90 = windows(theta, 90.0)
    if name.startswith("pair-"):                      # pair-<l>-<deg>: only pair l is cut, beyond <deg>
        _, l, deg = name.split("-")
        M = torch.ones(P, T, dtype=torch.bool)
        M[int(l)] = window_table(windows(theta, float(deg)), T)[int(l)]
        return M
    if name == "ctrl-slow":
        return window_table(w90[::-1].copy(), T)
    if name.startswith("ctrl-perm-s"):
        rng = np.random.default_rng(int(name[len("ctrl-perm-s"):]))
        return window_table(w90[rng.permutation(P)], T)
    if name.startswith("ctrl-iid-s"):
        rng = np.random.default_rng(int(name[len("ctrl-iid-s"):]))
        n_keep = window_table(w90, T).sum(0).numpy()
        M = np.zeros((P, T), dtype=bool)
        for d in range(T):
            M[rng.choice(P, size=int(n_keep[d]), replace=False), d] = True
        return torch.from_numpy(M)
    raise ValueError(name)


# ----------------------------------------------------------------------------- slow definition (tests)


def masked_scores_slow(q: torch.Tensor, k: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    """q, k [B, H, T, d] after RoPE (same number of heads); M [P, T] bool. Returns scores [B, H, T, T] with -inf
    above the diagonal and where no pair is kept."""
    T, d = q.shape[-2], q.shape[-1]
    P = d // 2
    pos = torch.arange(T, device=q.device)
    delta = pos[:, None] - pos[None, :]
    dist = delta.clamp(min=0)
    M = M.to(q.device)
    s = torch.zeros(*q.shape[:-1], T, device=q.device, dtype=q.dtype)
    for l in range(P):
        idx = [l, l + P]
        s += (q[..., idx] @ k[..., idx].transpose(-1, -2)) * M[l][dist]
    s = s / math.sqrt(d)
    dead = (delta < 0) | ~M.any(0)[dist]
    return s.masked_fill(dead, -math.inf)


# ----------------------------------------------------------------------------- blocked implementation


class TableAttention:
    """core_attention(q, k, v) for a keep-table. q [B, H, T, d], k / v [B, H_kv, T, d] after RoPE; causal.

    set_table(M) switches the condition. T must be a multiple of `block`.
    `head_gate` (optional, [H] bool): heads where it is False use dense attention (for per-head ablations).
    """

    def __init__(self, M: torch.Tensor | None = None, block: int = 64, device="cuda"):
        self.S = block
        self.device = device
        self.plan = None
        self.head_gate = None
        self.enabled = True
        self.exempt_keys = 0       # the first n keys of the sequence keep every pair (diagnostic)
        if M is not None:
            self.set_table(M)

    def set_table(self, M: torch.Tensor):
        P, T = M.shape
        S = self.S
        assert T % S == 0
        N = T // S
        Mn = M.numpy()
        a = np.arange(S)
        plan = []
        any_keep = Mn.any(0)
        for D in range(N):
            dist = D * S + a[:, None] - a[None, :]          # [S, S]; negative only when D == 0
            dc = np.clip(dist, 0, T - 1)
            lo, hi = max(D * S - (S - 1), 0), D * S + S - 1
            seg = Mn[:, lo:hi + 1]
            full = seg.all(1)
            none = ~seg.any(1)
            mixed = np.where(~full & ~none)[0]
            full = np.where(full)[0]
            dead = (dist < 0) | ~any_keep[dc]
            entry = {
                "full": None if len(full) == P else torch.tensor(np.concatenate([full, full + P]), device=self.device),
                "n_full": len(full),
                "mixed": torch.tensor(mixed, device=self.device),
                "mask": torch.tensor(Mn[mixed][:, dc].transpose(1, 2, 0), device=self.device) if len(mixed) else None,
                "dead": torch.tensor(dead, device=self.device) if dead.any() else None,
            }
            plan.append(entry)
        self.plan, self.P, self.T = plan, P, T
        self.is_dense = bool(Mn.all())

    def scores(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """q, k [B, H, T, d] (same heads) -> [B, H, T, T] scores (already / sqrt(d), -inf where masked)."""
        B, H, T, d = q.shape
        S, P = self.S, self.P
        assert T == self.T and d == 2 * P
        N = T // S
        qb, kb = q.reshape(B, H, N, S, d), k.reshape(B, H, N, S, d)
        blk = q.new_full((B, H, N, N, S, S), -math.inf)
        ar = torch.arange(N, device=q.device)
        for D, e in enumerate(self.plan):
            qd, kd = qb[:, :, D:], kb[:, :, :N - D]
            if e["full"] is None:
                res = qd @ kd.transpose(-1, -2)
            elif e["n_full"] > 0:
                res = qd[..., e["full"]] @ kd[..., e["full"]].transpose(-1, -2)
            else:
                res = q.new_zeros(B, H, N - D, S, S)
            m = e["mixed"]
            if m.numel():
                # chunk over mixed pairs to bound memory
                for c0 in range(0, m.numel(), 8):
                    mm = m[c0:c0 + 8]
                    c = qd[..., :, None, mm] * kd[..., None, :, mm] + qd[..., :, None, mm + P] * kd[..., None, :, mm + P]
                    res = res + (c * e["mask"][..., c0:c0 + 8]).sum(-1)
            res = res / math.sqrt(d)
            if e["dead"] is not None:
                res = res.masked_fill(e["dead"], -math.inf)
            blk[:, :, ar[D:], ar[:N - D]] = res
        s = blk.permute(0, 1, 2, 4, 3, 5).reshape(B, H, T, T)
        n = self.exempt_keys
        if n:
            full = (q @ k[:, :, :n].transpose(-1, -2)) / math.sqrt(d)                 # [B, H, T, n]
            causal = torch.arange(T, device=q.device)[:, None] >= torch.arange(n, device=q.device)[None, :]
            s[..., :n] = full.masked_fill(~causal, -math.inf)
        return s

    def __call__(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        G = q.shape[1] // k.shape[1]
        if not self.enabled or (self.is_dense and self.head_gate is None):
            return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=G > 1)
        if G > 1:
            k, v = k.repeat_interleave(G, dim=1), v.repeat_interleave(G, dim=1)
        if self.head_gate is not None:
            out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
            hs = torch.where(self.head_gate)[0]
            if hs.numel():
                out[:, hs] = torch.softmax(self.scores(q[:, hs], k[:, hs]), dim=-1) @ v[:, hs]
            return out
        return torch.softmax(self.scores(q, k), dim=-1) @ v


# ----------------------------------------------------------------------------- patching a HF Llama model


def patch_model(model, block: int = 64) -> list[TableAttention]:
    """Replace every layer's attention by the paper repo's pluggable LlamaAttention with one TableAttention core
    per layer (so layers can be switched individually). Weights are kept. Returns the list of cores."""
    from hisa.integrations.hf import LlamaAttention

    cores = []
    for layer in model.model.layers:
        old = layer.self_attn
        core = TableAttention(block=block, device=old.q_proj.weight.device)
        new = LlamaAttention(model.config, old.layer_idx, core_attention=core)
        new.load_state_dict(old.state_dict())
        new.to(device=old.q_proj.weight.device, dtype=old.q_proj.weight.dtype)
        layer.self_attn = new
        cores.append(core)
    model.config.use_cache = False
    return cores


def set_condition(cores: list[TableAttention], M: torch.Tensor, layers=None, head_gate=None):
    """Apply table M to the given layers (default all); other layers run dense."""
    for i, c in enumerate(cores):
        on = layers is None or i in layers
        c.enabled = on
        c.head_gate = head_gate if on else None
        if on:
            c.set_table(M)
