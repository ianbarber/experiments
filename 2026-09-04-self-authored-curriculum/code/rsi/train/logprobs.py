"""fp32 chunked log-probabilities through a custom autograd Function (docs/04 smoke test 4)."""
from __future__ import annotations
import torch

class ChunkedLogprobs(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h, W, targets, chunk):
        T = h.shape[0]
        lp = torch.empty(T, dtype=torch.float32, device=h.device)
        lse = torch.empty(T, dtype=torch.float32, device=h.device)
        for s in range(0, T, chunk):
            e = min(s + chunk, T)
            logits = torch.mm(h[s:e], W.t(), out_dtype=torch.float32)
            l = torch.logsumexp(logits, dim=-1)
            lp[s:e] = logits.gather(1, targets[s:e, None]).squeeze(1) - l
            lse[s:e] = l
            del logits
        ctx.save_for_backward(h, W, targets, lse); ctx.chunk = chunk
        return lp

    @staticmethod
    def backward(ctx, grad_lp):
        h, W, targets, lse = ctx.saved_tensors
        chunk = ctx.chunk; T = h.shape[0]
        grad_h = torch.empty_like(h)
        for s in range(0, T, chunk):
            e = min(s + chunk, T)
            logits = torch.mm(h[s:e], W.t(), out_dtype=torch.float32)
            g = torch.exp(logits - lse[s:e, None]); g.mul_(-grad_lp[s:e, None])
            idx = torch.arange(e - s, device=h.device)
            g[idx, targets[s:e]] += grad_lp[s:e]
            grad_h[s:e] = torch.mm(g.to(h.dtype), W)
            del logits, g
        return grad_h, None, None, None

def sequence_logprobs(backbone, W, ids: torch.Tensor, chunk: int = 1024) -> torch.Tensor:
    """log p(ids[t] | ids[<t]) for t >= 1; ids is a 1-D LongTensor on the model device. Returns [T-1] fp32."""
    hs = backbone(input_ids=ids[None], use_cache=False).last_hidden_state[0]
    return ChunkedLogprobs.apply(hs[:-1], W, ids[1:], chunk)
