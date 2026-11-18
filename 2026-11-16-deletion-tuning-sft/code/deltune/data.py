"""The Wordle tool-calling games as assistant-only SFT examples."""

import random

import torch
from datasets import load_dataset

from .wordle import wordlists

HEADER = "<|im_start|>assistant\n"
END = "<|im_end|>"
N_VAL_GAMES = 50


def clean_message(m: dict) -> dict:
    out = {k: v for k, v in m.items() if v is not None}
    if out.get("tool_calls"):
        out["tool_calls"] = [
            {"type": tc["type"], "function": {"name": tc["function"]["name"], "arguments": dict(tc["function"]["arguments"])}}
            for tc in out["tool_calls"]
        ]
    else:
        out.pop("tool_calls", None)
    return out


def load_games():
    """Returns (train_games, val_games): lists of message lists. The split is fixed."""
    wl = wordlists()
    ds = load_dataset(wl["dataset"], revision=wl["revision"], split="train")
    games = [[clean_message(m) for m in row["messages"]] for row in ds]
    idx = list(range(len(games)))
    random.Random(42).shuffle(idx)
    val = [games[i] for i in idx[:N_VAL_GAMES]]
    train = [games[i] for i in idx[N_VAL_GAMES:]]
    return train, val


def encode(tok, messages) -> dict:
    """Token ids for the whole conversation, with labels only on assistant turns."""
    text = tok.apply_chat_template(messages, tools=wordlists()["tools"], tokenize=False)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    spans, pos = [], 0
    while True:
        start = text.find(HEADER, pos)
        if start < 0:
            break
        start += len(HEADER)
        end = text.find(END, start) + len(END)
        spans.append((start, end))
        pos = end
    ids = enc["input_ids"]
    labels = [-100] * len(ids)
    for i, (a, b) in enumerate(enc["offset_mapping"]):
        if any(a >= s and b <= e for s, e in spans) and b > a:
            labels[i] = ids[i]
    return {"input_ids": ids, "labels": labels}


def collate(examples, pad_id: int, device="cuda"):
    n = max(len(e["input_ids"]) for e in examples)
    ids = torch.full((len(examples), n), pad_id, dtype=torch.long)
    labels = torch.full((len(examples), n), -100, dtype=torch.long)
    mask = torch.zeros((len(examples), n), dtype=torch.long)
    for i, e in enumerate(examples):
        k = len(e["input_ids"])
        ids[i, :k] = torch.tensor(e["input_ids"])
        labels[i, :k] = torch.tensor(e["labels"])
        mask[i, :k] = 1
    return ids.to(device), mask.to(device), labels.to(device)


def batches(examples, batch_size: int, seed: int):
    """Endless stream of shuffled batches, reshuffled every epoch."""
    rng = random.Random(seed)
    while True:
        order = list(range(len(examples)))
        rng.shuffle(order)
        for i in range(0, len(order) - batch_size + 1, batch_size):
            yield [examples[j] for j in order[i : i + batch_size]]
