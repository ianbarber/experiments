"""Kernel-level check: fla chunk_gated_delta_rule continued from an initial state vs one call vs an
fp64 sequential reference.  python tests/check_fla_continuation.py"""
import torch, torch.nn.functional as F
from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
torch.manual_seed(0)

def ref(q, k, v, g, beta, h0=None):
    # q,k: [T,H,K] (already l2-normalised), v: [T,H,V], g,beta: [T,H]; state S: [H,K,V]
    T, H, K = q.shape; V = v.shape[-1]
    S = torch.zeros(H, K, V, dtype=torch.float64) if h0 is None else h0.double().clone()
    out = []
    for t in range(T):
        a = g[t].double().exp()[:, None, None]
        kt, vt, qt, bt = k[t].double(), v[t].double(), q[t].double(), beta[t].double()
        S = S * a
        pred = torch.einsum('hk,hkv->hv', kt, S)
        S = S + torch.einsum('hk,hv->hkv', kt, bt[:, None] * (vt - pred))
        out.append(torch.einsum('hk,hkv->hv', qt * K ** -0.5, S))
    return torch.stack(out), S

H, K, V = 4, 128, 128
def data(T):
    q = F.normalize(torch.randn(T, H, K), dim=-1); k = F.normalize(torch.randn(T, H, K), dim=-1)
    v = torch.randn(T, H, V); g = -F.softplus(torch.randn(T, H)) * 0.1; beta = torch.rand(T, H).sigmoid()
    return q, k, v, g, beta

def fla(q, k, v, g, beta, h0=None, dt=torch.bfloat16):
    c = lambda x: x[None].cuda().to(dt)
    o, hT = chunk_gated_delta_rule(c(q), c(k), c(v), g=g[None].cuda().float(), beta=c(beta),
                                   initial_state=None if h0 is None else h0[None].cuda(),
                                   output_final_state=True, use_qk_l2norm_in_kernel=False)
    return o[0].float().cpu(), hT[0].float().cpu()

for T1, T2 in [(1400, 24), (1400, 200), (1400, 2000), (64, 2000)]:
    q, k, v, g, beta = data(T1 + T2)
    o_ref, _ = ref(q, k, v, g, beta)
    o_one, _ = fla(q, k, v, g, beta)
    _, h1 = fla(q[:T1], k[:T1], v[:T1], g[:T1], beta[:T1])
    o_split32, _ = fla(q[T1:], k[T1:], v[T1:], g[T1:], beta[T1:], h0=h1.float())
    o_split16, _ = fla(q[T1:], k[T1:], v[T1:], g[T1:], beta[T1:], h0=h1.to(torch.bfloat16))
    seg = o_ref[T1:].float()
    rel = lambda a: ((a - seg).norm() / seg.norm()).item()
    print(f"T1={T1} T2={T2}: one-call {rel(o_one[T1:]):.4f}  split(h0 fp32) {rel(o_split32):.4f}  "
          f"split(h0 bf16) {rel(o_split16):.4f}   [rel. error vs fp64 reference on the 2nd segment]")
