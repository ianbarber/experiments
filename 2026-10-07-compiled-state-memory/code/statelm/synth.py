"""Synthetic user-memory files for plumbing tests and the position-sweep gate.

A memory is a list of dated bullet entries about one user. One attribute is the *target*;
it is rendered with value X or Y (counterfactual pair) at a controlled entry index; everything
else is identical between the X and Y versions.
"""
from __future__ import annotations

import random

# (key, entry template, question, answer lead-in, value pool)
ATTRS = [
    ("dog", "The user's dog is named {v}.", "What is my dog's name?", "Your dog's name is",
     ["Biscuit", "Pepper", "Juniper", "Waffles", "Mochi", "Ranger", "Clover", "Pickles"]),
    ("city", "The user lives in {v}.", "Which city do I live in?", "You live in",
     ["Lisbon", "Denver", "Osaka", "Tallinn", "Cork", "Halifax", "Porto", "Tucson"]),
    ("employer", "The user works at {v}.", "Where do I work?", "You work at",
     ["Brightwell Labs", "Northgate Freight", "Kestrel Bank", "Ostrander Foods", "Pellucid Health",
      "Marlowe Robotics", "Fenwick Studios", "Quarry Lane Press"]),
    ("cuisine", "The user's favourite cuisine is {v}.", "What is my favourite cuisine?",
     "Your favourite cuisine is", ["Ethiopian", "Peruvian", "Georgian", "Sichuan", "Lebanese", "Basque",
                                   "Oaxacan", "Keralan"]),
    ("allergy", "The user is allergic to {v}.", "What am I allergic to?", "You are allergic to",
     ["penicillin", "shellfish", "peanuts", "latex", "bee stings", "sesame", "kiwi", "mustard"]),
    ("sister", "The user's sister is called {v}.", "What is my sister's name?", "Your sister's name is",
     ["Ingrid", "Talia", "Rosalind", "Mireille", "Akosua", "Henrike", "Paloma", "Yevgenia"]),
    ("car", "The user drives a {v}.", "What car do I drive?", "You drive a",
     ["Volvo V60", "Subaru Outback", "Mazda MX-5", "Kia Niro", "Skoda Octavia", "Ford Ranger",
      "Honda Jazz", "Renault Zoe"]),
    ("instrument", "The user is learning to play the {v}.", "Which instrument am I learning?",
     "You are learning to play the", ["cello", "oboe", "banjo", "harp", "trombone", "accordion",
                                      "bassoon", "mandolin"]),
    ("sport", "The user plays {v} on weekends.", "What sport do I play on weekends?",
     "You play", ["squash", "hurling", "badminton", "water polo", "ultimate frisbee", "curling",
                  "lacrosse", "fencing"]),
    ("drink", "The user's usual coffee order is a {v}.", "What is my usual coffee order?",
     "Your usual coffee order is a", ["cortado", "flat white", "macchiato", "long black", "mocha",
                                      "affogato", "red eye", "piccolo"]),
    ("book", "The user's favourite novel is {v}.", "What is my favourite novel?",
     "Your favourite novel is", ["Middlemarch", "Solaris", "Beloved", "Dune", "Rebecca", "Ulysses",
                                 "Kindred", "Stoner"]),
    ("hometown", "The user grew up in {v}.", "Where did I grow up?", "You grew up in",
     ["Galway", "Bergen", "Ghent", "Leeds", "Quebec City", "Hobart", "Utrecht", "Dunedin"]),
]

FILLER = [
    "The user mentioned a {a} trip planned for {m}.",
    "The user is trying to {b} more often.",
    "The user asked for advice about {c}.",
    "The user said their {d} has been {e} lately.",
    "The user prefers {f} replies when busy.",
    "The user recommended a podcast about {c}.",
    "The user is saving up for {g}.",
    "The user finished a course on {c}.",
]
FILL_VOCAB = dict(
    a=["weekend", "work", "camping", "family", "cycling", "ski"],
    m=["March", "June", "the autumn", "next spring", "December", "late July"],
    b=["cook at home", "go running", "read before bed", "call their parents", "stretch", "journal"],
    c=["gardening", "tax returns", "sourdough", "home networking", "watercolour", "bird watching",
       "chess openings", "knitting", "retirement savings", "beekeeping"],
    d=["commute", "sleep", "back", "workload", "garden", "neighbourhood"],
    e=["better", "worse", "unpredictable", "quieter", "busier", "fine"],
    f=["short", "bullet-point", "detailed", "casual"],
    g=["a new laptop", "a trip to Japan", "a house deposit", "an e-bike", "a piano", "a sofa"],
)


def _date(rng: random.Random) -> str:
    return f"2026-{rng.randint(1, 9):02d}-{rng.randint(1, 28):02d}"


def make_memory(target_key: str, target_value: str, n_entries: int, target_index: int, seed: int,
                other_attrs: bool = True) -> str:
    """Bullet memory with `n_entries` entries; the target attribute sits at `target_index`.
    The filler is a deterministic function of `seed` only, so X/Y versions differ in one span."""
    tmpl = next(a for a in ATTRS if a[0] == target_key)[1]
    return make_memory_with(tmpl.format(v=target_value), n_entries, target_index, seed, other_attrs,
                            exclude_key=target_key)


def make_memory_with(sentence: str, n_entries: int, target_index: int, seed: int,
                     other_attrs: bool = True, exclude_key: str | None = None) -> str:
    """As make_memory, but inserts an arbitrary target sentence."""
    rng = random.Random(seed)
    entries = []
    others = [a for a in ATTRS if a[0] != exclude_key]
    rng.shuffle(others)
    oi = 0
    for _ in range(n_entries - 1):
        if other_attrs and oi < len(others) and rng.random() < 0.25:
            k, tmpl, *_rest, pool = others[oi]
            oi += 1
            entries.append(tmpl.format(v=rng.choice(pool)))
        else:
            t = rng.choice(FILLER)
            entries.append(t.format(**{k: rng.choice(v) for k, v in FILL_VOCAB.items()}))
    entries.insert(target_index, sentence)
    dates = sorted(_date(rng) for _ in entries)
    return "\n".join(f"- [{d}] {e}" for d, e in zip(dates, entries))


def attr(key: str):
    return next(a for a in ATTRS if a[0] == key)
