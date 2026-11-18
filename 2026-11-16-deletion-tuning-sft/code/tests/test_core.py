"""CPU-only checks of the simulator, the label mask and the mask parameterisation."""

import random

import pytest
import torch
from torch import nn

from deltune import data, methods, wordle


def test_score_examples():
    assert wordle.score("level", "shelf") == list("YYXXX")
    assert wordle.score("trade", "shelf") == list("XXXXY")
    assert wordle.score("shelf", "shelf") == list("GGGGG")
    # repeated letters: only as many yellows as the secret has
    assert wordle.score("geese", "eagle") == list("YYXXG")


def test_simulator_replays_every_training_game():
    """Feeding the dataset's own guesses to the simulator must reproduce its tool messages."""
    train, val = data.load_games()
    assert len(train) == 950 and len(val) == 50
    for msgs in train + val:
        secret = msgs[-1]["content"].split()[-1].strip(".").lower()
        g = wordle.Game(secret=secret)
        for m, reply in zip(msgs[1::2], msgs[2::2]):
            assert reply["role"] == "tool"
            assert g.step(m["tool_calls"][0]["function"]["arguments"]["guess"]) == reply["content"]
        assert g.won


def test_invalid_moves_end_the_game_on_the_second_in_a_row():
    g = wordle.Game(secret="shelf")
    assert "invalid move" in g.step("crane without brackets")
    assert not g.done
    g.step("[crane]")
    assert g.invalid_streak == 0
    g.step("[zzzzz]")
    g.step("[toolong]")
    assert g.done and g.outcome == "invalid"


def test_out_of_guesses():
    g = wordle.Game(secret="shelf")
    for w in ["crane", "slate", "raise", "trace", "stare", "crate"]:
        g.step(f"[{w}]")
    assert g.done and g.outcome == "out_of_guesses" and not g.won


def test_parse_assistant():
    text = "TRADE gave ...</think><tool_call>\n<function=guess>\n<parameter=guess>\n[level]\n</parameter>\n</function>\n</tool_call><|im_end|>"
    p = wordle.parse_assistant(text)
    assert p["action"] == "[level]" and p["reasoning"] == "TRADE gave ..."
    assert wordle.parse_assistant("still thinking")["call"] is None
    bare = "<tool_call>\n<function=guess>\n<parameter=guess>\n[enjoy]\n</parameter>\n</function>\n</tool_call>"
    assert wordle.parse_assistant(bare)["action"] == "[enjoy]"
    assert wordle.parse_assistant("x</think>Solved.<|im_end|>")["content"] == "Solved."


def test_heldout_words_never_appear_in_training():
    wl = wordle.wordlists()
    train, val = data.load_games()
    text = " ".join(
        str(m.get(k) or "") for msgs in train + val for m in msgs[1:] for k in ("content", "reasoning_content")
    ).lower()
    held = set(wl["heldout_val"]) | set(wl["heldout_test"])
    assert len(held) == 800 and not (set(wl["heldout_val"]) & set(wl["heldout_test"]))
    assert not held & set(wl["seen"])
    for w in random.Random(0).sample(sorted(held), 200):
        assert f" {w} " not in text and f"[{w}]" not in text


def test_masked_linear_identity_and_gradients():
    torch.manual_seed(0)
    lin = nn.Linear(16, 32, bias=False)
    x = torch.randn(4, 16)
    ref = lin(x)
    m = methods.MaskedLinear(lin)
    m.set_mode("hard")
    assert torch.equal(m(x), ref)  # all logits positive: nothing deleted
    m.set_mode("soft")
    m.alpha = 1e4
    assert torch.allclose(m(x), ref, atol=1e-6)
    with torch.no_grad():
        m.logits[0, :] = -0.03
    m.set_mode("hard")
    out = m(x)
    assert torch.all(out[:, 0] == 0) and torch.equal(out[:, 1:], ref[:, 1:])
    m.set_mode("soft")
    m.alpha = 100.0
    m(x).sum().backward()
    assert m.logits.grad is not None and m.weight.grad is None


def test_wanda_init_sparsity_matches_paper():
    torch.manual_seed(0)
    lin = nn.Linear(512, 1024, bias=False)
    m = methods.MaskedLinear(lin)
    norms = [torch.rand(512) + 0.5]
    eta = 1 / 2048
    methods.init_logits_from_wanda([m], norms, eta, seed=0)
    frac = float((m.logits <= 0).float().mean())
    # eta * Phi(3) + (1 - eta) * Phi(-3) = 0.18%
    assert 0.0015 < frac < 0.0022
    methods.init_logits_from_wanda([m], norms, eta, seed=0, noise=False)
    assert abs(float((m.logits <= 0).float().mean()) - eta) < 1e-5
    # the deleted weights are the lowest-scoring ones
    score = lin.weight.abs() * norms[0][None, :]
    assert score[m.logits <= 0].max() <= score[m.logits > 0].min()


def test_regulariser_gradient_matches_closed_form():
    torch.manual_seed(0)
    lin = nn.Linear(8, 8, bias=False)
    m = methods.MaskedLinear(lin)
    m.alpha = 50.0
    with torch.no_grad():
        m.logits.copy_(0.03 * torch.randn(8, 8))
    lam = 1e-2
    methods.regulariser_backward([m], lam)
    s = torch.sigmoid(m.alpha * m.logits)
    expected = -2 * lam * m.alpha * m.weight.pow(2) * s.pow(2) * (1 - s)
    assert torch.allclose(m.logits.grad, expected, rtol=1e-4, atol=1e-10)


def test_packed_mask_roundtrip():
    torch.manual_seed(0)
    ms = [methods.MaskedLinear(nn.Linear(13, 7, bias=False)) for _ in range(3)]
    with torch.no_grad():
        for m in ms:
            m.logits.copy_(torch.randn(7, 13))
    keep = [(m.logits > 0).clone() for m in ms]
    blob = methods.packed_mask(ms)
    with torch.no_grad():
        for m in ms:
            m.logits.fill_(1.0)
    methods.load_packed_mask(ms, blob)
    for m, k in zip(ms, keep):
        assert torch.equal(m.logits > 0, k)


@pytest.fixture(scope="module")
def tok():
    from deltune.common import load_tokenizer

    return load_tokenizer()


def test_labels_cover_assistant_turns_only(tok):
    train, _ = data.load_games()
    ex = data.encode(tok, train[0])
    labelled = tok.decode([t for t, l in zip(ex["input_ids"], ex["labels"]) if l != -100])
    assert "<tool_call>" in labelled and "Solved: the word is" in labelled and "<think>" in labelled
    assert "You submitted" not in labelled and "expert Wordle solver" not in labelled and "<tools>" not in labelled
    n_turns = sum(m["role"] == "assistant" for m in train[0])
    assert labelled.count("<|im_end|>") == n_turns
    assert ex["labels"][0] == -100
