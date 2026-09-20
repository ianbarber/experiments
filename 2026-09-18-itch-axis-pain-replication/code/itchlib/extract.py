"""Final-token residual-stream activations at every decoder block output (HF forward hooks).

Equivalent to the paper's TransformerLens `blocks.{L}.hook_resid_post` read at position -1.
"""
import torch

BOS_VARIANTS = {"none": None, "endoftext": "<|endoftext|>", "im_end": "<|im_end|>"}


@torch.no_grad()
def final_token_activations(tok, model, layers, prompts, bos="none", batch_size=32):
    """Returns a bf16 tensor [n_layers, n_prompts, d_model]."""
    bos_tok = BOS_VARIANTS[bos]
    bos_id = tok.convert_tokens_to_ids(bos_tok) if bos_tok else None
    pad_id = tok.pad_token_id
    captured = {}

    def make_hook(i):
        def hook(module, inputs, output):
            captured[i] = output[0] if isinstance(output, tuple) else output
        return hook

    handles = [l.register_forward_hook(make_hook(i)) for i, l in enumerate(layers)]
    chunks = []
    try:
        for k in range(0, len(prompts), batch_size):
            batch = prompts[k:k + batch_size]
            enc = [tok(p, add_special_tokens=False).input_ids for p in batch]
            if bos_id is not None:
                enc = [[bos_id] + e for e in enc]
            n = max(len(e) for e in enc)
            ids = torch.full((len(enc), n), pad_id, dtype=torch.long)
            mask = torch.zeros((len(enc), n), dtype=torch.long)
            for i, e in enumerate(enc):                      # right padding: causal attention leaves
                ids[i, :len(e)] = torch.tensor(e)            # the real tokens untouched by the pads
                mask[i, :len(e)] = 1
            last = mask.sum(-1) - 1
            model(input_ids=ids.cuda(), attention_mask=mask.cuda(), use_cache=False)
            rows = torch.arange(len(enc))
            chunks.append(torch.stack([captured[i][rows, last.to(captured[i].device)].to(torch.bfloat16).cpu()
                                       for i in range(len(layers))], 0))
            captured.clear()
    finally:
        for h in handles:
            h.remove()
    return torch.cat(chunks, 1)
