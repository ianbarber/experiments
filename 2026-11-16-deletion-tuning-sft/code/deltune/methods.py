"""The three parameterisations: full fine-tuning, LoRA, and Deletion Tuning.

Deletion Tuning follows Lu et al. (arXiv 2611.10866), section 4: one logit G per weight of each
block's MLP gate projection, soft mask sigmoid(alpha * G) in training, hard mask 1[G > 0] at
evaluation, Wanda-prior initialisation, and the MaskLLM regulariser -lambda * ||W~||^2.
"""

import math

import numpy as np
import torch
import torch.nn.functional as F
import zstandard
from torch import nn

from .common import gate_modules

# attention q/k/v/o and its per-head output gate, the convolution block's in/out projections, the MLP
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "g_proj", "in_proj", "out_proj", "w1", "w2", "w3"]


class MaskedLinear(nn.Module):
    """A frozen linear layer whose weight is multiplied by a learned mask."""

    def __init__(self, linear: nn.Linear):
        super().__init__()
        assert linear.bias is None
        self.weight = linear.weight
        self.weight.requires_grad_(False)
        self.logits = nn.Parameter(torch.full_like(self.weight, 0.03))
        self.alpha = 1.0
        self.mode = "soft"  # soft | hard | off
        self._hard = None

    def set_mode(self, mode: str):
        self.mode = mode
        self._hard = None

    def masked_weight(self):
        if self.mode == "soft":
            return self.weight * torch.sigmoid(self.alpha * self.logits)
        if self.mode == "hard":
            if self._hard is None:
                self._hard = (self.weight * (self.logits > 0)).detach()
            return self._hard
        return self.weight

    def forward(self, x):
        return F.linear(x, self.masked_weight())


def install_masks(model) -> list[MaskedLinear]:
    masked = []
    for _, layer in enumerate(model.model.layers):
        m = MaskedLinear(layer.feed_forward.w1)
        layer.feed_forward.w1 = m
        masked.append(m)
    return masked


@torch.no_grad()
def wanda_input_norms(model, batches_iter) -> list[torch.Tensor]:
    """||X_j||_2 of the input to every gate projection over the calibration batches."""
    mods = [m for _, m in gate_modules(model)]
    sums = [torch.zeros(m.weight.shape[1], dtype=torch.float64, device=m.weight.device) for m in mods]
    state = {}

    def hook(i):
        def fn(_, inputs):
            x = inputs[0].float()
            keep = state["mask"].unsqueeze(-1).to(x.dtype)
            sums[i] += (x * keep).pow(2).sum(dim=(0, 1)).double()

        return fn

    handles = [m.register_forward_pre_hook(hook(i)) for i, m in enumerate(mods)]
    n_tokens = 0
    try:
        for ids, mask, _ in batches_iter:
            state["mask"] = mask
            with torch.autocast("cuda", dtype=torch.bfloat16):
                model.model(input_ids=ids, attention_mask=mask)
            n_tokens += int(mask.sum())
    finally:
        for h in handles:
            h.remove()
    return [s.sqrt().float() for s in sums], n_tokens


@torch.no_grad()
def init_logits_from_wanda(masked: list[MaskedLinear], norms, eta: float, seed: int, noise: bool = True):
    """Eq. 5 of the paper. The eta-quantile is taken over each weight matrix."""
    gen = torch.Generator(device=masked[0].weight.device).manual_seed(seed)
    for m, norm in zip(masked, norms):
        score = m.weight.abs().float() * norm[None, :]
        k = max(1, int(round(eta * score.numel())))
        thr = torch.kthvalue(score.flatten(), k).values
        low = score <= thr
        mean = torch.where(low, -0.03, 0.03)
        if noise:
            m.logits.copy_(mean + 0.01 * torch.randn(score.shape, generator=gen, device=score.device))
        else:
            m.logits.copy_(mean)
        m._hard = None


def regulariser_backward(masked: list[MaskedLinear], lam: float) -> float:
    """Adds the gradient of -lam * sum ||W * sigmoid(alpha G)||^2 to the logits, layer by layer."""
    if lam == 0:
        return 0.0
    total = 0.0
    for m in masked:
        reg = -lam * (m.weight * torch.sigmoid(m.alpha * m.logits)).pow(2).sum()
        reg.backward()
        total += float(reg.detach())
    return total


@torch.no_grad()
def mask_stats(masked: list[MaskedLinear], n_model_params: int) -> dict:
    deleted = sum(int((m.logits <= 0).sum()) for m in masked)
    target = sum(m.logits.numel() for m in masked)
    return {
        "deleted": deleted,
        "target_weights": target,
        "deleted_frac_of_target": deleted / target,
        "deleted_frac_of_model": deleted / n_model_params,
    }


@torch.no_grad()
def packed_mask(masked: list[MaskedLinear]) -> bytes:
    bits = np.concatenate([np.packbits((m.logits > 0).cpu().numpy().ravel()) for m in masked])
    return bits.tobytes()


def mask_bytes_zstd(masked: list[MaskedLinear], level: int = 19) -> int:
    return len(zstandard.ZstdCompressor(level=level).compress(packed_mask(masked)))


def unpack_mask(blob: bytes, shapes) -> list[torch.Tensor]:
    """Boolean keep-masks, one per gate projection, from a packed hard mask."""
    bits = np.frombuffer(blob, dtype=np.uint8)
    out, pos = [], 0
    for shape in shapes:
        n = int(np.prod(shape))
        nbytes = math.ceil(n / 8)
        out.append(torch.from_numpy(np.unpackbits(bits[pos : pos + nbytes])[:n].reshape(shape).astype(bool)))
        pos += nbytes
    assert pos == len(bits)
    return out


@torch.no_grad()
def load_packed_mask(masked: list[MaskedLinear], blob: bytes):
    """Restore logits (as +-0.03) from a packed hard mask."""
    keeps = unpack_mask(blob, [tuple(m.logits.shape) for m in masked])
    for m, keep in zip(masked, keeps):
        m.logits.copy_(torch.where(keep.to(m.logits.device), 0.03, -0.03))
        m._hard = None


def install_lora(model, rank: int, keep_dtype: bool = False):
    """Inject LoRA into every linear projection of every block. keep_dtype leaves the adapters in the
    model's own dtype (for the bf16 evaluation copy) instead of promoting them to fp32."""
    from peft import LoraConfig, get_peft_model

    cfg = LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.0, bias="none", target_modules=LORA_TARGETS)
    return get_peft_model(model, cfg, autocast_adapter_dtype=not keep_dtype)


def lora_bytes(peft_model) -> int:
    """Bytes of the A and B factors at fp32, as in the paper's appendix A."""
    return 4 * sum(p.numel() for n, p in peft_model.named_parameters() if "lora_" in n)
