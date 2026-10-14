from types import SimpleNamespace

import torch
from transformers import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Attention

from fwlayer.observe import TraceCapture


def test_query_row_matches_full_eager_attention_weights():
    torch.manual_seed(7)
    config = Qwen3_5TextConfig(hidden_size=16, num_attention_heads=2,
                             num_key_value_heads=1, head_dim=8,
                             num_hidden_layers=1, layer_types=["full_attention"])
    config._attn_implementation = "eager"
    module = Qwen3_5Attention(config, 0).eval()
    hidden = torch.randn(1, 5, 16)
    cos, sin = torch.ones(1, 5, 4), torch.zeros(1, 5, 4)
    pair = SimpleNamespace(source_positions=[1, 3], query_position=4,
                           value_positions=[[1], [3, 4]])
    model = SimpleNamespace(model=SimpleNamespace(layers=[]))
    capture = TraceCapture(model, pair)
    with torch.inference_mode():
        _, full = module(hidden_states=hidden, position_embeddings=(cos, sin), attention_mask=None)
        capture.attention_row(0, module, hidden, {"position_embeddings": (cos, sin)})
    expected = torch.stack([full[0, :, -1, positions].sum(-1)
                            for positions in pair.value_positions], -1)
    torch.testing.assert_close(capture.attention[0], expected, rtol=1e-5, atol=1e-6)
