"""Derived attention-layer seeds (LABNOTES 2026-10-10, Stage 1).

For one attention layer and KV head the memory gives un-rotated keys K [T, d] (all treated as
position 0), values V [T, d] and reference queries Q [J, d] already rotated to the session's
distance D. A seed is N entries (c_i, b_i, u_i), consumed by unchanged attention as extra KV plus
an additive logit bias, fitted so that for each reference query (s = the layer's softmax scale)

    log sum_i exp(s q.c_i + b_i)              ~= log sum_t exp(s q.k_t)              (mass)
    sum_i softmax_i(s q.c_i + b_i) u_i        ~= sum_t softmax_t(s q.k_t) v_t         (output)

Keys are k-means centroids of K (query-blind); b starts at log(cluster size) and is fitted by Adam
on the squared log-mass error; u is a ridge least-squares fit pulled toward the cluster-mean
values. This is Attention Matching [zweiger2026fast] with centroid keys [zhan2026reskv] in place
of a key subset. Everything is fp32 on the GPU; the caller rounds keys and values to bf16.
"""
from __future__ import annotations

import torch


def rotate(x: torch.Tensor, ang: torch.Tensor) -> torch.Tensor:
    """Apply RoPE with angles `ang` [..., rd] (cat of the half angles) to the first rd dims of x."""
    rd = ang.shape[-1]
    r = x[..., :rd]
    rh = torch.cat([-r[..., rd // 2:], r[..., :rd // 2]], -1)
    return torch.cat([r * ang.cos() + rh * ang.sin(), x[..., rd:]], -1)


def _assign(x, c, chunk=16384):
    cn = (c * c).sum(1)
    out = torch.empty(x.shape[0], dtype=torch.long, device=x.device)
    for i in range(0, x.shape[0], chunk):
        out[i:i + chunk] = (cn[None] - 2 * x[i:i + chunk] @ c.T).argmin(1)
    return out


def cluster_sums(x: torch.Tensor, a: torch.Tensor, n: int):
    """Per-cluster sums [n, d] and counts [n] of x by assignment a, run-to-run deterministic: sorted
    segment sums of an fp64 cumsum (CUDA index_add_ adds floats in arbitrary order)."""
    cs = torch.cumsum(x[torch.argsort(a, stable=True)].double(), 0)
    cs = torch.cat([cs.new_zeros(1, x.shape[1]), cs])
    cnt = torch.bincount(a, minlength=n)
    ends = torch.cumsum(cnt, 0)
    return (cs[ends] - cs[ends - cnt]).to(x.dtype), cnt


def kmeans(x: torch.Tensor, n: int, iters: int = 20, seed: int = 42, w: torch.Tensor | None = None):
    """Lloyd k-means from a seeded random subset. Returns centroids [n, d] and assignments [T].
    An empty cluster keeps its previous centroid (its bias starts at log 1).
    w [T] > 0 (optional): weighted Lloyd. The initial centroids are drawn without replacement with
    probability proportional to w, and each centroid is the w-weighted mean of its points (points are
    still assigned to the nearest centroid). w=None runs the unweighted path unchanged."""
    g = torch.Generator(device=x.device).manual_seed(seed)
    if w is None:
        c = x[torch.randperm(x.shape[0], generator=g, device=x.device)[:n]].clone()
    else:
        c = x[torch.multinomial(w, n, replacement=False, generator=g)].clone()
    for _ in range(iters):
        a = _assign(x, c)
        if w is None:
            s, cnt = cluster_sums(x, a, n)
            cnt = cnt.to(x.dtype)
            c = torch.where((cnt > 0)[:, None], s / cnt.clamp_min(1)[:, None], c)
        else:
            s, _ = cluster_sums(x * w[:, None], a, n)
            ws = cluster_sums(w[:, None], a, n)[0][:, 0]
            c = torch.where((ws > 0)[:, None], s / ws.clamp_min(1e-30)[:, None], c)
    return c, _assign(x, c)


def full_targets(Q, K, V, s, chunk=1024, w=None):
    """Full-memory log-mass m [J] and attention output y [J, d] for reference queries Q. With row
    weights w [J], also the attention mass each key receives, sum_j w_j softmax_t(s q_j . k_t) [T]."""
    m, y = [], []
    mass = None if w is None else torch.zeros(K.shape[0], dtype=torch.float64, device=K.device)
    for i in range(0, Q.shape[0], chunk):
        S = s * (Q[i:i + chunk] @ K.T)
        m.append(torch.logsumexp(S, 1))
        P = torch.softmax(S, 1)
        y.append(P @ V)
        if w is not None:
            mass += (w[i:i + chunk] @ P).double()
    if w is not None:
        return torch.cat(m), torch.cat(y), mass.float()
    return torch.cat(m), torch.cat(y)


def seed_metrics(Q, C, b, U, m, y, s):
    """Log-mass RMSE and relative output error of a seed (C, b, U) on queries Q with targets m, y, plus
    the 90th percentile and max of the per-row relative output error (the tail a few lookups sit in)."""
    S = s * (Q @ C.T) + b
    lm_err = (torch.logsumexp(S, 1) - m).pow(2).mean().sqrt().item()
    out = torch.softmax(S, 1) @ U
    row = (out - y).norm(dim=1) / y.norm(dim=1).clamp_min(1e-12)
    return dict(logmass_rmse=lm_err, out_rel=((out - y).norm() / y.norm()).item(),
                row_p90=row.quantile(0.9).item(), row_max=row.max().item())


EMPTY_BIAS = -30.0  # empty clusters: zero key, zero value, ~zero mass (no raw key left in the seed)


def fit_head(K, V, Q, n, s, *, held=0.1, val=0.1, steps=300, lr=0.05,
             ridges=(1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0), seed=42, iters=20, km=None, groups=None,
             extra=None, extra_w=1.0, keys="kmeans"):
    """Fit one KV head's seed. K, V [T, d], Q [J, d] fp32 on one device. Returns (C, b, U, b0, vbar,
    stats); (C, b0, vbar) is the unfitted seed (log cluster size, cluster-mean values).
    Queries are split (seeded) into fit / val / held by group (groups [J]; default one group per
    row), so that e.g. the q heads of one token, or the rows of one self-study item, never straddle
    splits. The ridge (solved in fp64) is chosen on val; the reported metrics are on held, which the
    fit never sees. km = (C, assignments) reuses a k-means of K. extra [Je, d] are rows always added to
    the fit set with weight extra_w (a float, or one weight per row [Je]); val and held come from Q only.
    keys="mass" places the keys by weighted k-means instead, each key weighted by the attention mass it
    receives from the fit rows (row weights included; val and held rows never place keys), plus a floor
    of 1e-6 of the mean mass so every key can be drawn. keys="massq" also measures distance in the
    fit rows' query metric, M = sum_j w_j q_j q_j^T / sum_j w_j: the weighted k-means runs on K M^(1/2),
    so it minimises the mass-weighted squared logit error s q.(k - c) the merge causes on the fit rows,
    and each centroid is the mass-weighted mean of its keys in the original space."""
    g = torch.Generator(device=Q.device).manual_seed(seed)
    if groups is None:
        groups = torch.arange(Q.shape[0], device=Q.device)
    uniq, inv = torch.unique(groups.to(Q.device), return_inverse=True)
    perm = torch.randperm(len(uniq), generator=g, device=Q.device)
    rank = torch.empty_like(perm)
    rank[perm] = torch.arange(len(uniq), device=Q.device)
    rank = rank[inv]
    gh, gv = max(1, int(held * len(uniq))), max(1, int(val * len(uniq)))
    Qh, Qv, Qf = Q[rank < gh], Q[(rank >= gh) & (rank < gh + gv)], Q[rank >= gh + gv]
    wf = torch.ones(Qf.shape[0], device=Q.device)
    if extra is not None:
        Qf = torch.cat([Qf, extra])
        ew = (extra_w.to(Q.device, torch.float32) if torch.is_tensor(extra_w)
              else torch.full((extra.shape[0],), float(extra_w), device=Q.device))
        assert ew.shape == (extra.shape[0],)
        wf = torch.cat([wf, ew])
    assert keys in ("kmeans", "mass", "massq") and not (keys != "kmeans" and km is not None)
    if keys in ("mass", "massq"):
        mf, yf, kmass = full_targets(Qf, K, V, s, w=wf)
        kw = kmass + 1e-6 * kmass.mean()
        if keys == "mass":
            C, a = kmeans(K, n, iters=iters, seed=seed, w=kw)
        else:
            M = (Qf * wf[:, None]).T.double() @ Qf.double() / wf.sum().double()
            ev, vec = torch.linalg.eigh(M)
            Mh = ((vec * ev.clamp_min(0).sqrt()) @ vec.T).float()
            _, a = kmeans(K @ Mh, n, iters=iters, seed=seed, w=kw)
            ks, _ = cluster_sums(K * kw[:, None], a, n)
            ws = cluster_sums(kw[:, None], a, n)[0][:, 0]
            C = ks / ws.clamp_min(1e-30)[:, None]
    else:
        C, a = km if km is not None else kmeans(K, n, iters=iters, seed=seed)
    vsum, cnt = cluster_sums(V, a, n)
    cnt = cnt.float()
    empty = cnt == 0
    C = torch.where(empty[:, None], torch.zeros_like(C), C)
    vbar = vsum / cnt.clamp_min(1)[:, None]
    b0 = torch.where(empty, torch.full_like(cnt, EMPTY_BIAS), cnt.clamp_min(1).log())
    if keys == "kmeans":
        mf, yf = full_targets(Qf, K, V, s)
    mv, yv = full_targets(Qv, K, V, s)
    mh, yh = full_targets(Qh, K, V, s)
    stats = dict(n=n, T=K.shape[0], J_fit=Qf.shape[0], J_val=Qv.shape[0], J_held=Qh.shape[0],
                 groups=len(uniq), extra=0 if extra is None else extra.shape[0], empty=int(empty.sum()),
                 extra_w=float(extra_w.float().mean()) if torch.is_tensor(extra_w) else extra_w,
                 extra_wsum=0.0 if extra is None else float(wf[-extra.shape[0]:].sum()), keys=keys,
                 init=seed_metrics(Qh, C, b0, vbar, mh, yh, s))
    Sf = s * (Qf @ C.T)
    b = b0.clone().requires_grad_(True)
    opt = torch.optim.Adam([b], lr=lr)
    with torch.enable_grad():
        for _ in range(steps):
            opt.zero_grad(set_to_none=True)
            loss = (wf * (torch.logsumexp(Sf + b, 1) - mf).pow(2)).sum() / wf.sum()
            loss.backward()
            opt.step()
    b = torch.where(empty, b0, b.detach())
    stats["bias_loss"] = loss.item()
    stats["bias_only"] = seed_metrics(Qh, C, b, vbar, mh, yh, s)
    P = torch.softmax(Sf + b, 1)
    Pw = P * wf[:, None]
    A, R = (P.T @ Pw).double(), (Pw.T @ yf).double()
    del Sf, P, Pw
    scale = A.diagonal().mean()
    best = None
    for r in ridges:
        Ar = A.clone()
        Ar.diagonal().add_(r * scale)
        U = torch.linalg.solve(Ar, R + r * scale * vbar.double()).float()
        e = seed_metrics(Qv, C, b, U, mv, yv, s)["out_rel"]
        if best is None or e < best[0]:
            best = (e, r, U)
    U = best[2]
    stats["ridge"] = best[1]
    stats["fit"] = seed_metrics(Qh, C, b, U, mh, yh, s)
    stats["fit_train"] = seed_metrics(Qf, C, b, U, mf, yf, s)
    return C, b, U, b0, vbar, stats

class QueryTap:
    """Collect the attention layers' pre-RoPE queries (q_norm output, [n, Hq, d] bf16) at the token
    indices `keep` (absolute positions in the memory) while a prefill that starts at `offset` runs."""

    def __init__(self, lm, keep: torch.Tensor, offset: int = 0):
        self.lm, self.keep = lm, keep.to(lm.device)
        self.q = {i: [] for i in lm.attn_layers}
        self.off = dict.fromkeys(lm.attn_layers, offset)
        self.hooks = []

    def _hook(self, i):
        def f(mod, inp, out):  # out [1, L, Hq, d]
            o, L = self.off[i], out.shape[1]
            sel = self.keep[(self.keep >= o) & (self.keep < o + L)] - o
            self.q[i].append(out[0, sel].detach().clone())
            self.off[i] = o + L
        return f

    def __enter__(self):
        for i in self.lm.attn_layers:
            self.hooks.append(self.lm.tm.layers[i].self_attn.q_norm.register_forward_hook(self._hook(i)))
        return self

    def __exit__(self, *exc):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def result(self) -> dict[int, torch.Tensor]:
        return {i: torch.cat(v) for i, v in self.q.items()}
