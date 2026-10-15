"""Compiled recurrent-state memory for Qwen3.5-architecture hybrids (GDN + gated attention).

Design notes (see LABNOTES 2026-10-07):
- We call the *text* decoder directly (not the multimodal wrapper, which caches rope_deltas
  across calls) and apply lm_head only at the positions we need.
- GDN recurrent states are kept in fp32 everywhere. Stock HF allocates the cache buffer in the
  model dtype (bf16) and rounds the kernel's fp32 state at every chunk boundary/decode step.
- Prefill is always batch size 1: once a cache has state, HF drops the linear-attention padding
  mask, so left padding would run pad tokens through (and decay) an injected state.
"""
from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from transformers import AutoModelForImageTextToText, AutoTokenizer, DynamicCache


def _sdpa_kvbias(module, query, key, value, attention_mask, statelm_kv_bias=None, **kw):
    """Stock SDPA plus an additive logit bias on the first N cache columns of this attention layer
    (statelm_kv_bias = {layer_idx: [N] or per KV head [Hkv, N], fp32}). Layers without an entry run
    stock SDPA unchanged.
    Used to mask or weight injected KV entries per layer (LABNOTES 2026-10-10)."""
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    b = None if statelm_kv_bias is None else statelm_kv_bias.get(module.layer_idx)
    if b is None:
        return sdpa_attention_forward(module, query, key, value, attention_mask, **kw)
    Lq, Lk, N = query.shape[-2], key.shape[-2], b.shape[-1]
    neg = torch.finfo(torch.float32).min
    if bool((b <= neg).all()):  # every injected entry hidden at this layer: drop them, keep the stock path
        m = None if attention_mask is None else attention_mask[..., N:]
        return sdpa_attention_forward(module, query, key[..., N:, :], value[..., N:, :], m, **kw)
    if attention_mask is None:  # SDPA skipped the mask: decode (Lq == 1) or an empty cache (Lq == Lk)
        base = torch.zeros(1, 1, Lq, Lk, device=query.device)
        if Lq > 1:
            base.masked_fill_(torch.ones(Lq, Lk, dtype=torch.bool, device=query.device).triu(Lk - Lq + 1), neg)
    elif attention_mask.dtype == torch.bool:
        base = torch.zeros(attention_mask.shape, device=query.device).masked_fill_(~attention_mask, neg)
    else:
        base = attention_mask.float()
    if b.dim() == 2:  # per KV head [Hkv, N] (fitted seeds): expand to the query heads
        bias = F.pad(b.float(), (0, Lk - N)).repeat_interleave(query.shape[1] // b.shape[0], 0)
        bias = bias.view(1, -1, 1, Lk)
    else:
        bias = F.pad(b.float(), (0, Lk - N)).view(1, 1, 1, Lk)
    mask = (base + bias).clamp_min(torch.finfo(query.dtype).min).to(query.dtype)
    return sdpa_attention_forward(module, query, key, value, mask, **kw)


@dataclass
class CompiledState:
    """Per-GDN-layer recurrent (fp32 [1,H,K,V]) and conv ([1,C,k]) states after a compile pass."""
    recurrent: dict[int, torch.Tensor]
    conv: dict[int, torch.Tensor]
    n_tokens: int
    text_sha: str = ""
    meta: dict = field(default_factory=dict)

    def to(self, device):
        return CompiledState({k: v.to(device) for k, v in self.recurrent.items()},
                             {k: v.to(device) for k, v in self.conv.items()},
                             self.n_tokens, self.text_sha, dict(self.meta))

    def nbytes(self) -> int:
        return sum(v.numel() * v.element_size() for v in self.recurrent.values())

    def save(self, path):
        torch.save({"recurrent": {k: v.cpu() for k, v in self.recurrent.items()},
                    "conv": {k: v.cpu() for k, v in self.conv.items()},
                    "n_tokens": self.n_tokens, "text_sha": self.text_sha, "meta": self.meta}, path)

    @staticmethod
    def load(path, device="cuda"):
        d = torch.load(path, map_location=device)
        return CompiledState(d["recurrent"], d["conv"], d["n_tokens"], d["text_sha"], d["meta"])


class LM:
    def __init__(self, path: str, dtype=torch.bfloat16, device="cuda", state_dtype=torch.float32):
        self.tok = AutoTokenizer.from_pretrained(path)
        # load straight to the GPU (GB10 memory is unified: CPU-then-.to() briefly holds two copies)
        full = AutoModelForImageTextToText.from_pretrained(path, dtype=dtype, device_map=device)
        full.model.visual = None  # text only
        full.eval()
        torch.cuda.empty_cache()
        self.tm = full.model.language_model
        self.lm_head = full.lm_head
        self.cfg = self.tm.config
        self.device = device
        self.state_dtype = state_dtype
        self.gdn_layers = [i for i, t in enumerate(self.cfg.layer_types) if t == "linear_attention"]
        self.attn_layers = [i for i, t in enumerate(self.cfg.layer_types) if t == "full_attention"]
        H, K, V = self.cfg.linear_num_value_heads, self.cfg.linear_key_head_dim, self.cfg.linear_value_head_dim
        self.state_shape = (1, H, K, V)
        self.conv_dim = 2 * self.cfg.linear_num_key_heads * K + H * V
        self.conv_k = self.cfg.linear_conv_kernel_dim
        self.model_dtype = dtype
        self.path = path
        self._head32 = None

    # ---------------------------------------------------------------- caches
    def new_cache(self) -> DynamicCache:
        """Fresh cache whose GDN recurrent buffers are fp32 zeros (has_previous_state=False)."""
        cache = DynamicCache(config=self.cfg)
        for i in self.gdn_layers:
            layer = cache.layers[i]
            layer.dtype, layer.device = self.model_dtype, torch.device(self.device)
            layer.recurrent_states = torch.zeros(self.state_shape, dtype=self.state_dtype, device=self.device)
            layer.is_recurrent_states_initialized = True
        return cache

    def cache_from_state(self, st: CompiledState | None, *, keep_conv: bool = False, scale: float = 1.0,
                         layer_mask: dict[int, float] | None = None,
                         head_mask: dict[int, torch.Tensor] | None = None) -> DynamicCache:
        """Session cache: GDN layers start from `st` (attention layers empty). st=None -> zero state
        but has_previous_state=True (exercises the same code path as an injected state)."""
        cache = self.new_cache()
        for i in self.gdn_layers:
            layer = cache.layers[i]
            conv = torch.zeros((1, self.conv_dim, self.conv_k), dtype=self.model_dtype, device=self.device)
            if st is not None and keep_conv:
                conv = st.conv[i].to(self.device, self.model_dtype).clone()
            layer.conv_states = conv
            layer.is_conv_states_initialized = True
            layer.max_batch_size, layer.conv_kernel_size = 1, self.conv_k
            if st is not None:
                s = st.recurrent[i].to(self.device, self.state_dtype) * scale
                if layer_mask is not None:
                    s = s * layer_mask.get(i, 1.0)
                if head_mask is not None and i in head_mask:
                    s = s * head_mask[i].to(s).view(1, -1, 1, 1)
                layer.recurrent_states.copy_(s)
            layer.has_previous_state = True
        return cache

    # --------------------------------------------------------------- forward
    @torch.no_grad()
    def forward(self, ids: torch.Tensor, cache: DynamicCache, position_offset: int | None = None,
                chunk: int = 8192) -> torch.Tensor:
        """Run ids [1,T] through the decoder, updating cache. Returns final hidden states [1,T,D]."""
        # attention-layer seeds (LABNOTES 2026-10-10): a cache may carry a RoPE shift for the session
        # (session position = cache index + shift) and a per-layer bias on its first N entries
        shift = getattr(cache, "statelm_pos_shift", None)
        if position_offset is None and shift is not None:
            position_offset = cache.get_seq_length() + shift
        bias = getattr(cache, "statelm_kv_bias", None)
        extra = {"statelm_kv_bias": bias} if bias is not None else {}
        outs = []
        for s in range(0, ids.shape[1], chunk):
            piece = ids[:, s:s + chunk]
            pos = None
            if position_offset is not None:
                p = torch.arange(piece.shape[1], device=ids.device) + position_offset + s
                pos = p.view(1, 1, -1).expand(4, 1, -1)
            h = self.tm(input_ids=piece, past_key_values=cache, use_cache=True, position_ids=pos,
                        **extra).last_hidden_state
            outs.append(h)
        return torch.cat(outs, dim=1)

    def logits(self, h: torch.Tensor) -> torch.Tensor:
        """fp32 logits (bf16 logits have 0.125-0.25 spacing at |logit| 16-64, i.e. ~0.1 nat noise)."""
        if self._head32 is None:
            self._head32 = self.lm_head.weight.detach().float()
        return F.linear(h.float(), self._head32)

    @torch.no_grad()
    def compile(self, ids: torch.Tensor, chunk: int = 8192, meta: dict | None = None) -> CompiledState:
        return self.prefill_full(ids, chunk=chunk, meta=meta)[1]

    @torch.no_grad()
    def prefill_full(self, ids: torch.Tensor, chunk: int = 8192, meta: dict | None = None):
        """One prefill of `ids` with the full model. Returns (full cache with KV + GDN states,
        CompiledState snapshot of the GDN states). The full cache serves M (continue it) and K
        (fork, zero GDN); the snapshot serves C."""
        cache = self.new_cache()
        self.forward(ids, cache, chunk=chunk)
        rec = {i: cache.layers[i].recurrent_states.detach().clone() for i in self.gdn_layers}
        conv = {i: cache.layers[i].conv_states.detach().clone() for i in self.gdn_layers}
        sha = hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest()[:16]
        return cache, CompiledState(rec, conv, int(ids.shape[1]), sha, meta or {})

    def zero_gdn(self, cache: DynamicCache) -> DynamicCache:
        """In place: zero every GDN recurrent and conv state (K condition), keep attention KV."""
        for i in self.gdn_layers:
            cache.layers[i].recurrent_states.zero_()
            cache.layers[i].conv_states.zero_()
            cache.layers[i].has_previous_state = True
        return cache

    @torch.no_grad()
    def score(self, cache: DynamicCache, prompt_ids: torch.Tensor, target_ids: torch.Tensor,
              position_offset: int | None = None) -> torch.Tensor:
        """Per-token log p(target | cache, prompt). Mutates cache. Returns [len(target)] float32."""
        ids = torch.cat([prompt_ids, target_ids], dim=1)
        h = self.forward(ids, cache, position_offset=position_offset)
        P, T = prompt_ids.shape[1], target_ids.shape[1]
        lp = F.log_softmax(self.logits(h[:, P - 1:P + T - 1]), dim=-1)
        return lp[0].gather(-1, target_ids[0, :, None])[:, 0]

    @torch.no_grad()
    def score_multi(self, cache: DynamicCache, prompt_ids: torch.Tensor, targets: list[torch.Tensor],
                    position_offset: int | None = None) -> list[torch.Tensor]:
        """Per-token log p(target_i | cache, prompt) for several targets; the prompt is run once and
        the cache is snapshotted (deep copy) before each target. Mutates `cache`."""
        h = self.forward(prompt_ids, cache, position_offset=position_offset)
        first = F.log_softmax(self.logits(h[:, -1]), dim=-1)[0]
        snap = copy.deepcopy(cache)
        start = None if position_offset is None else position_offset + prompt_ids.shape[1]
        out = []
        for t in targets:
            lp0 = first[t[0, 0]].view(1)
            if t.shape[1] == 1:
                out.append(lp0)
                continue
            c = copy.deepcopy(snap)
            h2 = self.forward(t[:, :-1], c, position_offset=start)
            lp = F.log_softmax(self.logits(h2), dim=-1)[0].gather(-1, t[0, 1:, None])[:, 0]
            out.append(torch.cat([lp0, lp]))
        return out

    @torch.no_grad()
    def kv_only_cache(self, ids: torch.Tensor, chunk: int = 8192) -> DynamicCache:
        """Condition K: prefill `ids` with the full model, keep the attention KV, zero every GDN
        recurrent and conv state (has_previous_state stays True). Positions continue after `ids`."""
        cache = self.new_cache()
        self.forward(ids, cache, chunk=chunk)
        return self.zero_gdn(cache)

    @torch.no_grad()
    def generate(self, cache: DynamicCache, prompt_ids: torch.Tensor, max_new_tokens: int = 128,
                 stop_ids: tuple[int, ...] | None = None) -> list[int]:
        """Greedy decoding (deterministic). Mutates cache."""
        stop = set(stop_ids or (self.tok.convert_tokens_to_ids("<|im_end|>"), self.tok.eos_token_id))
        h = self.forward(prompt_ids, cache)
        nxt = int(self.logits(h[:, -1]).argmax(-1))
        out = []
        for _ in range(max_new_tokens):
            if nxt in stop:
                break
            out.append(nxt)
            h = self.forward(torch.tensor([[nxt]], device=self.device), cache)
            nxt = int(self.logits(h[:, -1]).argmax(-1))
        return out

    # ------------------------------------------------- attention-layer seeds (LABNOTES 2026-10-10)
    def enable_kv_bias(self):
        """Route attention through _sdpa_kvbias (stock SDPA when a cache carries no bias)."""
        from transformers import AttentionInterface
        from transformers.masking_utils import AttentionMaskInterface, sdpa_mask
        AttentionInterface.register("sdpa_kvbias", _sdpa_kvbias)
        AttentionMaskInterface.register("sdpa_kvbias", sdpa_mask)
        self.cfg._attn_implementation = "sdpa_kvbias"

    def _rope_angles(self, positions: torch.Tensor) -> torch.Tensor:
        inv = self.tm.rotary_emb.inv_freq.float().to(self.device)
        ang = positions.float().to(self.device)[:, None] * inv[None]
        return torch.cat([ang, ang], -1)

    @torch.no_grad()
    def unrotate_keys(self, k: torch.Tensor) -> torch.Tensor:
        """Cached post-RoPE keys [1, Hkv, T, D] at positions 0..T-1 -> pre-RoPE keys (bf16). Only the
        first rotary_dim dims rotate (partial RoPE); the inverse is done in fp32, so the result differs
        from the true pre-RoPE keys by the forward pass's bf16 rounding (~0.25% rel. on rotary dims)."""
        emb = self._rope_angles(torch.arange(k.shape[-2]))
        rd = emb.shape[-1]
        kf = k.to(torch.float32, copy=True)
        r = kf[..., :rd]
        rh = torch.cat([-r[..., rd // 2:], r[..., :rd // 2]], -1)
        kf[..., :rd] = r * emb.cos() - rh * emb.sin()
        return kf.to(k.dtype)

    @torch.no_grad()
    def rope_keys(self, k_pre: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Pre-RoPE keys [1, Hkv, T, D] -> keys rotated to `positions` [T] the way the model does it."""
        import transformers.models.qwen3_5.modeling_qwen3_5 as qm
        pid = positions.to(self.device).view(1, 1, -1).expand(3, 1, -1)
        cos, sin = self.tm.rotary_emb(k_pre, pid)
        return qm.apply_rotary_pos_emb(k_pre, k_pre, cos, sin)[1]

    # --------------------------------------------------------------- helpers
    def ids(self, text: str) -> torch.Tensor:
        return torch.tensor([self.tok.encode(text, add_special_tokens=False)], device=self.device)
