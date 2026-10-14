"""Oracle-labelled pairs: swap keys while keeping value tokens at fixed positions."""
from __future__ import annotations

import copy
import random
import re
from dataclasses import asdict, dataclass

from factworld.config import WorldConfig
from factworld.oracle import Oracle
from factworld.render import Renderer
from factworld.world import World

SYSTEM = "Answer with only the value identifier. Use the supplied facts."
NOTE = "The weather remains ordinary, and this note records nothing relevant.\n"


def first_difference(a: list[int], b: list[int]) -> int:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    raise ValueError("Answers must have a differing token before either answer ends")


def answer_correct(prediction: str, expected: str) -> bool:
    # FactWorld relaxed match: accept a trailing full stop/newline, and score the answer span.
    match = re.match(r"^(v[0-9]+)(?=[\s.]|$)", prediction.strip())
    return bool(match and match.group(1) == expected)


@dataclass
class Pair:
    pair_id: str
    index: int
    nominal_distance: int
    distractors: int
    prompt_a: str
    prompt_b: str
    base_ids_a: list[int]
    base_ids_b: list[int]
    ids_a: list[int]
    ids_b: list[int]
    answer_a: str
    answer_b: str
    answer_ids_a: list[int]
    answer_ids_b: list[int]
    shared_answer_ids: list[int]
    differing_index: int
    token_a: int
    token_b: int
    value_positions: list[list[int]]
    source_positions: list[int]
    query_position: int
    changed_positions: list[int]
    distance_a: int
    distance_b: int
    query_key: str
    swap_key: str
    seed: int = 20261014

    def to_dict(self):
        return asdict(self)


def make_pair(tokenizer, distance: int, index: int, seed: int = 20261014,
              distractors: int = 16) -> Pair:
    if distance < 80:
        raise ValueError("Need at least 80 tokens between source and scoring position")
    if not 0 <= distractors <= 898:
        raise ValueError("distractors must be between 0 and 898")
    # Reuse identities across distances so distance is the only varying task coordinate.
    rng = random.Random(f"fwlayer|{seed}|{index}")
    world_a = World(WorldConfig(seed=seed + index, n_entities=1000, n_attributes=1,
                               value_vocab_size=1000, k=5))
    eligible = list(range(100, 1000))
    rng.shuffle(eligible)
    keys, seen_values = [], set()
    for n in eligible:
        key = f"e{n}"
        value = world_a.attrs[key]["a0"]
        if re.fullmatch(r"v[1-9][0-9]{2}", value) and value not in seen_values:
            keys.append(key)
            seen_values.add(value)
        if len(keys) == distractors + 2:
            break
    if len(keys) != distractors + 2:
        raise ValueError("Not enough distinct three-digit oracle values for requested distractors")
    k1, k2 = keys[:2]
    va, vb = Oracle(world_a).recall(k1, "a0"), Oracle(world_a).recall(k2, "a0")
    world_b = copy.deepcopy(world_a)
    world_b.attrs[k1]["a0"], world_b.attrs[k2]["a0"] = vb, va
    assert Oracle(world_b).recall(k1, "a0") == vb
    renderer = Renderer()
    distractor_lines = [renderer.render_fact(k, "a0", Oracle(world_a).recall(k, "a0"))
                        for k in keys[2:]]
    rng.shuffle(distractor_lines)
    pair_rows = [(k1, va), (k2, vb)]
    if rng.getrandbits(1):
        pair_rows.reverse()
    facts_a = "\n".join(distractor_lines + [renderer.render_fact(k, "a0", v) for k, v in pair_rows])
    facts_b = "\n".join(distractor_lines + [renderer.render_fact(k2 if k == k1 else k1, "a0", v)
                                           for k, v in pair_rows])
    query = renderer.render_query("recall", entity=k1, attribute="a0")

    def render(facts, notes):
        user = f"Facts:\n{facts}\n\nUnrelated notes:\n{notes}\nQuestion: {query}"
        return tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)

    def encode(text):
        return tokenizer.encode(text, add_special_tokens=False)

    def answer_ids(text, answer):
        base = encode(text)
        completed = encode(text + answer)
        if completed[:len(base)] != base:
            raise ValueError("Answer boundary changes prompt tokenization")
        return completed[len(base):]

    plain = render(facts_a, "")
    ans_a, ans_b = answer_ids(plain, va), answer_ids(plain, vb)
    diff = first_difference(ans_a, ans_b)
    shared = ans_a[:diff]
    assert shared == ans_b[:diff]

    def spans(text):
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        offsets = encoded["offset_mapping"]
        out = []
        for value in (va, vb):
            m = re.search(rf"\bis ({value})\.", text)
            if m is None:
                raise ValueError("Historical value was not uniquely located")
            start, end = m.span(1)
            positions = [i for i, (a, b) in enumerate(offsets) if b > start and a < end]
            if not positions:
                raise ValueError("Empty value span")
            out.append(positions)
        return out

    def measured(text):
        positions = spans(text)
        return len(encode(text)) + len(shared) - 1 - min(p[-1] for p in positions)

    needed = distance - measured(plain)
    if needed < 0:
        raise ValueError("Requested distance is shorter than the query/template overhead")
    sentence_tokens = len(encode(NOTE))
    n_sentences = max(0, needed // sentence_tokens - 1)
    residual = max(0, needed - n_sentences * sentence_tokens)
    for _ in range(10):
        notes = NOTE * n_sentences + " note" * residual
        text_a, text_b = render(facts_a, notes), render(facts_b, notes)
        error = distance - measured(text_a)
        if error == 0:
            break
        if residual + error < 0:
            n_sentences -= 1
            residual += sentence_tokens
        residual += error
    else:
        raise ValueError("Could not construct the exact token distance")

    base_a, base_b = encode(text_a), encode(text_b)
    ids_a, ids_b = base_a + shared, base_b + shared
    if len(ids_a) != len(ids_b):
        raise ValueError("Counterfactual keys did not preserve prompt length")
    positions = spans(text_a)
    if positions != spans(text_b):
        raise ValueError("Value positions moved between executions")
    changed = [i for i, (a, b) in enumerate(zip(ids_a, ids_b)) if a != b]
    allowed_chars = [m.span(0) for m in re.finditer(rf"\b(?:{k1}|{k2})(?='s)", text_a)]
    offsets = tokenizer(text_a, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
    allowed = {i for i, (a, b) in enumerate(offsets)
               if any(b > s and a < e for s, e in allowed_chars)}
    if not changed or not set(changed) <= allowed:
        raise ValueError("Counterfactual changed tokens outside the historical keys")
    for span in positions:
        if any(ids_a[p] != ids_b[p] for p in span):
            raise ValueError("Value token identity changed")
    q = len(ids_a) - 1
    assert max(q - p[-1] for p in positions) == distance
    return Pair(f"d{distance}-p{index:03d}", index, distance, distractors, text_a, text_b,
                base_a, base_b, ids_a, ids_b, va, vb, ans_a, ans_b, shared, diff,
                ans_a[diff], ans_b[diff], positions, [p[-1] for p in positions], q,
                changed, q - positions[0][-1], q - positions[1][-1], k1, k2, seed)
