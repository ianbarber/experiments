"""A local Wordle simulator that mirrors TextArena Wordle-v0 as wrapped by OpenEnv, and a
batched player that drives a chat model through it with the dataset's tool-calling format."""

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import torch
from wordfreq import zipf_frequency

WORDLISTS = Path(__file__).resolve().parent.parent / "wordlists.json"
MAX_GUESSES = 6
TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*<function=([\w\-]+)>\s*(?:<parameter=([\w\-]+)>\s*(.*?)\s*</parameter>)?", re.S
)


@lru_cache(maxsize=1)
def wordlists() -> dict:
    return json.loads(WORDLISTS.read_text())


@lru_cache(maxsize=200_000)
def is_word(word: str) -> bool:
    """Stand-in for the environment's English dictionary check."""
    wl = wordlists()
    return word in _known() or zipf_frequency(word, "en") >= 1.5 or word in wl["seen"]


@lru_cache(maxsize=1)
def _known() -> frozenset:
    wl = wordlists()
    return frozenset(wl["seen"]) | frozenset(wl["openers"]) | frozenset(wl["heldout_val"]) | frozenset(wl["heldout_test"])


def score(word: str, secret: str) -> list[str]:
    feedback, rest = [None] * 5, list(secret)
    for i in range(5):
        if word[i] == secret[i]:
            feedback[i], rest[i] = "G", None
    for i in range(5):
        if feedback[i] is None:
            if word[i] in rest:
                feedback[i] = "Y"
                rest[rest.index(word[i])] = None
            else:
                feedback[i] = "X"
    return feedback


def consistent(word: str, history: list[tuple[str, list[str]]]) -> bool:
    """Would `word` have produced every feedback seen so far if it were the secret?"""
    return all(score(g, word) == fb for g, fb in history)


@dataclass
class Game:
    secret: str
    messages: list = field(default_factory=list)
    history: list = field(default_factory=list)  # (word, feedback) for valid guesses
    turns: list = field(default_factory=list)  # per assistant turn: raw text, parsed action, kind
    invalid_streak: int = 0
    done: bool = False
    outcome: str = ""  # win | out_of_guesses | invalid | no_tool_call

    @property
    def won(self) -> bool:
        return self.outcome == "win"

    def step(self, action: str) -> str:
        """Apply one tool call. Returns the tool message content."""
        echo = f"\n[GAME] {action[:60]}\n[GAME] "
        match = re.search(r"\[(\w+)\]", action)
        reason = None
        if match is None:
            reason = "You tried submitting a word in the wrong format. Please make sure to use squared brackets."
        else:
            word = match.group(1).lower()
            if len(word) != 5:
                reason = "Your word must be exactly 5 letters."
            elif word in [w for w, _ in self.history]:
                reason = f"You have already guessed '{word}' before. Please try a different word."
            elif not (word.isascii() and word.isalpha() and is_word(word)):
                reason = f"'{word}' is not an English word."
        if reason is not None:
            self.invalid_streak += 1
            if self.invalid_streak >= 2:
                self.done, self.outcome = True, "invalid"
                return echo + f"Invalid Move: {reason}"
            return echo + (
                f"Player 0 attempted an invalid move. Reason: {reason} "
                "Please resubmit a valid move and remember to follow the game rules to avoid penalties."
            )
        self.invalid_streak = 0
        feedback = score(word, self.secret)
        self.history.append((word, feedback))
        if all(f == "G" for f in feedback):
            self.done, self.outcome = True, "win"
            return echo + "Congratulations! You guessed the word correctly!"
        left = MAX_GUESSES - len(self.history)
        text = echo + f"You submitted [{word}].\nFeedback:\n{' '.join(word.upper())}\n{' '.join(feedback)}\nYou have {left} guesses left."
        if left == 0:
            self.done, self.outcome = True, "out_of_guesses"
            text += "\n[GAME] The turn limit has been reached. You didn't guess the word."
        return text


def parse_assistant(text: str) -> dict:
    """Split generated text into reasoning, content and an optional guess action."""
    text = text.split("<|im_end|>")[0]
    if "</think>" in text:
        reasoning, rest = text.split("</think>", 1)
    elif "<tool_call>" in text:
        # thinking disabled (the prompt already closed the think block), or an unclosed think block
        cut = text.index("<tool_call>")
        reasoning, rest = text[:cut], text[cut:]
    else:
        reasoning, rest = text, ""
    reasoning = reasoning.replace("<think>", "")
    m = TOOL_CALL_RE.search(rest)
    out = {"reasoning": reasoning, "content": rest[: m.start()].strip() if m else rest.strip(), "action": None, "call": None}
    if m:
        name, param, value = m.group(1), m.group(2), m.group(3)
        out["call"] = {"name": name, "param": param, "value": value}
        if name == "guess" and param == "guess" and value is not None:
            out["action"] = value
    return out


GREEDY = dict(do_sample=False, repetition_penalty=1.0, temperature=None, top_k=None, top_p=None)


def _generate(model, tok, prompts, max_new_tokens, first_phase, eos, gen_kwargs):
    """Greedy-generate one turn per prompt. Rows still unfinished after `first_phase` tokens are
    continued in a second, smaller batch, so one rambling game does not hold a whole batch to the
    full budget. Returns the new token ids per row (lists, EOS included when reached)."""
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    width = enc["input_ids"].shape[1]
    kw = dict(eos_token_id=eos, pad_token_id=tok.pad_token_id, **gen_kwargs)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model.generate(**enc, max_new_tokens=min(first_phase, max_new_tokens), **kw)
    rows = []
    for r in out[:, width:].tolist():
        rows.append(r[: r.index(eos) + 1] if eos in r else [t for t in r if t != tok.pad_token_id])
    todo = [i for i, r in enumerate(rows) if eos not in r and len(r) >= first_phase]
    if todo and max_new_tokens > first_phase:
        seqs = [enc["input_ids"][i][enc["attention_mask"][i].bool()].tolist() + rows[i] for i in todo]
        n = max(len(q) for q in seqs)
        ids = torch.full((len(seqs), n), tok.pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(seqs), n), dtype=torch.long)
        for j, q in enumerate(seqs):
            ids[j, n - len(q) :] = torch.tensor(q)
            mask[j, n - len(q) :] = 1
        with torch.autocast("cuda", dtype=torch.bfloat16):
            more = model.generate(input_ids=ids.to(model.device), attention_mask=mask.to(model.device), max_new_tokens=max_new_tokens - first_phase, **kw)
        for j, r in zip(todo, more[:, n:].tolist()):
            rows[j] += r[: r.index(eos) + 1] if eos in r else [t for t in r if t != tok.pad_token_id]
    return rows


@torch.no_grad()
def play(model, tok, secrets, *, max_new_tokens=256, batch_size=256, enable_thinking=True, gen_kwargs=None, max_turns=12, first_phase=160, token_budget=100_000):
    """Play one game per secret. Returns the list of finished Game objects, in order.

    Batches are sized by prompt tokens (rows x longest prompt <= token_budget): prefilling long
    conversations is what uses memory, through the MLP's 7,680-wide intermediates."""
    wl = wordlists()
    tools = wl["tools"]
    gen_kwargs = dict(GREEDY if gen_kwargs is None else gen_kwargs)
    games = [Game(secret=s, messages=[{"role": "user", "content": wl["user_prompt"]}]) for s in secrets]
    tok.padding_side = "left"
    eos = tok.convert_tokens_to_ids("<|im_end|>")
    was_training = model.training
    model.eval()
    for _ in range(max_turns):
        active = [g for g in games if not g.done]
        if not active:
            break
        prompts = [
            tok.apply_chat_template(g.messages, tools=tools, add_generation_prompt=True, tokenize=False, enable_thinking=enable_thinking)
            for g in active
        ]
        lens = [len(ids) for ids in tok(prompts, add_special_tokens=False)["input_ids"]]
        chunks, cur = [], []
        for i in sorted(range(len(active)), key=lambda i: lens[i]):
            if cur and ((len(cur) + 1) * lens[i] > token_budget or len(cur) >= batch_size):
                chunks.append(cur)
                cur = []
            cur.append(i)
        chunks.append(cur)
        for idx in chunks:
            new = _generate(model, tok, [prompts[i] for i in idx], max_new_tokens, first_phase, eos, gen_kwargs)
            for i, row in zip(idx, new):
                g = active[i]
                text = tok.decode(row, skip_special_tokens=False)
                finished = "<|im_end|>" in text
                p = parse_assistant(text)
                turn = {"raw": text.split("<|im_end|>")[0], "finished": finished, "action": p["action"], "n_tokens": len(row)}
                if p["call"] is None:
                    g.done, g.outcome = True, "no_tool_call"
                    turn["kind"] = "no_tool_call"
                    g.turns.append(turn)
                    continue
                before = len(g.history)
                prior_history = list(g.history)
                action = p["action"] if p["action"] is not None else ""
                reply = g.step(action)
                valid = len(g.history) > before
                turn["kind"] = "valid" if valid else "invalid"
                if valid:
                    turn["consistent"] = consistent(g.history[-1][0], prior_history)
                g.turns.append(turn)
                msg = {"role": "assistant", "content": p["content"], "reasoning_content": p["reasoning"]}
                msg["tool_calls"] = [{"type": "function", "function": {"name": "guess", "arguments": {"guess": action}}}]
                g.messages.append(msg)
                g.messages.append({"role": "tool", "name": "guess", "content": reply})
    for g in games:
        if not g.done:
            g.done, g.outcome = True, "turn_cap"
    if was_training:
        model.train()
    return games


def summarize(games) -> dict:
    n = len(games)
    wins = [g for g in games if g.won]
    valid_turns = [t for g in games for t in g.turns if t["kind"] == "valid"]
    later = [t for g in games for t in g.turns[1:] if t["kind"] == "valid"]
    all_turns = [t for g in games for t in g.turns]
    outcomes = {}
    for g in games:
        outcomes[g.outcome] = outcomes.get(g.outcome, 0) + 1
    return {
        "n": n,
        "win_rate": len(wins) / n,
        "mean_guesses_when_won": sum(len(g.history) for g in wins) / max(1, len(wins)),
        "outcomes": outcomes,
        "valid_turn_rate": len(valid_turns) / max(1, len(all_turns)),
        "consistent_rate_after_first": sum(t.get("consistent", False) for t in later) / max(1, len(later)),
        "mean_tokens_per_turn": sum(t["n_tokens"] for t in all_turns) / max(1, len(all_turns)),
        "truncated_turn_rate": sum(not t["finished"] for t in all_turns) / max(1, len(all_turns)),
    }


def records(games) -> list[dict]:
    return [
        {"secret": g.secret, "outcome": g.outcome, "won": g.won, "guesses": [w for w, _ in g.history], "n_turns": len(g.turns)}
        for g in games
    ]


def transcript(g: Game) -> str:
    lines = [f"=== secret={g.secret} outcome={g.outcome} guesses={[w for w, _ in g.history]}"]
    for m in g.messages[1:]:
        if m["role"] == "assistant":
            lines.append(f"[think] {m.get('reasoning_content', '').strip()}")
            if m.get("content"):
                lines.append(f"[say] {m['content']}")
            lines.append(f"[guess] {m['tool_calls'][0]['function']['arguments']['guess']}")
        else:
            lines.append("[env] " + m["content"].strip().replace("\n", " | "))
    if g.outcome == "no_tool_call" and g.turns:
        lines.append("[raw final turn] " + g.turns[-1]["raw"].strip()[:1500])
    return "\n".join(lines)
