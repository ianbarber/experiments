"""Triton kernels that torch.compile generates for a few standard models."""

from __future__ import annotations

import torch

from .instrument import attempt


def _resnet50():
    import torchvision
    return torchvision.models.resnet50(weights=None), (torch.randn(8, 3, 224, 224), ), {}


def _vit_b_16():
    import torchvision
    return torchvision.models.vit_b_16(weights=None), (torch.randn(8, 3, 224, 224), ), {}


def _bert_base():
    from transformers import BertConfig, BertModel
    return BertModel(BertConfig()), (torch.randint(0, 30000, (8, 128)), ), {}


def _gpt2():
    from transformers import GPT2Config, GPT2LMHeadModel
    return GPT2LMHeadModel(GPT2Config()), (torch.randint(0, 50000, (8, 128)), ), {}


MODELS = {"resnet50": _resnet50, "vit_b_16": _vit_b_16, "bert_base": _bert_base, "gpt2": _gpt2}
TRAIN = ("resnet50", "gpt2")


def _first_tensor(out):
    if isinstance(out, torch.Tensor):
        return out
    if hasattr(out, "logits"):
        return out.logits
    if hasattr(out, "last_hidden_state"):
        return out.last_hidden_state
    return out[0]


def _infer(name):
    torch.manual_seed(42)
    model, args, kwargs = MODELS[name]()
    model = model.cuda().eval()
    args = tuple(a.cuda() for a in args)
    compiled = torch.compile(model)
    with torch.no_grad():
        for _ in range(2):
            compiled(*args, **kwargs)


def _train(name):
    torch.manual_seed(42)
    model, args, kwargs = MODELS[name]()
    model = model.cuda().train()
    args = tuple(a.cuda() for a in args)
    compiled = torch.compile(model)
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    for _ in range(2):
        opt.zero_grad()
        loss = _first_tensor(compiled(*args, **kwargs)).float().mean()
        loss.backward()
        opt.step()


def run(name: str):
    """`name` is `inductor` for everything or `inductor:<model>` for one model."""
    names = list(MODELS) if ":" not in name else [name.split(":", 1)[1]]
    for model in names:
        attempt(f"inductor {model} inference", _infer, model)
        if model in TRAIN:
            attempt(f"inductor {model} train step", _train, model)
        torch._dynamo.reset()
