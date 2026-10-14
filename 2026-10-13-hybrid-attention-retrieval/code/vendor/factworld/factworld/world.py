"""The world model (KB) — the single source of truth.

A `World` is deterministic from its seed. It holds the *static* layer (recall attributes,
initial object holders, initial S_k role assignment, entity document-frequency weights) and
samples *event chains* on demand, length-parameterised, for the state tasks. The oracle
(`factworld.oracle`) composes a chain into ground-truth answers; documents and eval items are
rendered *from* this, never the reverse.

Identifiers are abstract atomic IDs, optionally namespaced (`e0`, `v3`, `o2`, `loc1`, `g4`,
`r0`). Value tokens are opaque and drawn from a single shared pool, so a value never reveals
which attribute it answers. A non-empty `id_namespace` yields a fully disjoint symbol set,
used to build auxiliary operator-worlds for the operator-learning split.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from .config import WorldConfig


@dataclass(frozen=True)
class Event:
    kind: str
    args: tuple[str, ...]

    def inverse(self) -> "Event":
        """Group inverse — defined only for the S_k role-permutation events."""
        if self.kind == "swap_role":
            return self  # a transposition is its own inverse
        if self.kind == "cycle_roles":
            head, *rest = self.args
            return Event("cycle_roles", (head, *reversed(rest)))
        raise ValueError(f"{self.kind!r} is last-write-wins, not an invertible group op")


class World:
    def __init__(self, config: WorldConfig):
        self.config = config
        ns = config.id_namespace
        rng = random.Random(f"factworld|world|{ns}|{config.seed}")

        # recall: a SHARED, opaque value pool (value tokens encode nothing about their attribute)
        self.attribute_names: tuple[str, ...] = tuple(f"{ns}a{i}" for i in range(config.n_attributes))
        self.value_vocab: tuple[str, ...] = tuple(f"{ns}v{i}" for i in range(config.value_vocab_size))
        self.entities: tuple[str, ...] = tuple(f"{ns}e{i}" for i in range(config.n_entities))
        self.attrs: dict[str, dict[str, str]] = {
            ent: {name: rng.choice(self.value_vocab) for name in self.attribute_names}
            for ent in self.entities
        }
        # entity document-frequency weights: Zipf over a seeded random ranking (rank decorrelated
        # from entity index), normalised. Logged here so the corpus is stratification-capable (R5).
        ranking = list(self.entities)
        rng.shuffle(ranking)
        raw = {ent: 1.0 / ((rank + 1) ** config.freq_skew) for rank, ent in enumerate(ranking)}
        total = sum(raw.values())
        self.entity_freq: dict[str, float] = {ent: raw[ent] / total for ent in self.entities}

        # easy-state and hard-state symbol sets
        self.objects: tuple[str, ...] = tuple(f"{ns}o{i}" for i in range(config.n_objects))
        self.locations: tuple[str, ...] = tuple(f"{ns}loc{i}" for i in range(config.n_locations))
        self.k: int = config.k
        self.agents: tuple[str, ...] = tuple(f"{ns}g{i}" for i in range(config.k))
        self.roles: tuple[str, ...] = tuple(f"{ns}r{i}" for i in range(config.k))

        # hard-state: k agents hold k roles. The initial assignment is the IDENTITY (g_i holds r_i)
        # — the canonical S_k word-problem start. A random hidden initial would be unidentifiable
        # from history-only training data, making the IWL-history-only hard task ill-posed (the
        # final role depends on an initial the model never observes). Identity is a known convention
        # the model learns once (from the aux operator-worlds, which share it) and applies.
        self.initial_assignment: dict[str, str] = {
            self.agents[i]: self.roles[i] for i in range(config.k)
        }
        # easy-state: each object starts at some holder drawn from the union (location or agent)
        self.initial_holder: dict[str, str] = {o: rng.choice(self.holders) for o in self.objects}

    @property
    def holders(self) -> tuple[str, ...]:
        """Domain of an object's holder slot: any location or any agent (one decorrelated union)."""
        return self.locations + self.agents

    def sample_easy_chain(self, length: int, episode_seed: object) -> list[Event]:
        rng = random.Random(f"factworld|easy|{self.config.id_namespace}|{self.config.seed}|{episode_seed}|{length}")
        holders = self.holders
        events: list[Event] = []
        for _ in range(length):
            o = rng.choice(self.objects)
            kind = "move" if rng.random() < 0.5 else "give"
            events.append(Event(kind, (o, rng.choice(holders))))  # both kinds draw from the union
        return events

    def sample_hard_chain(self, length: int, episode_seed: object) -> list[Event]:
        rng = random.Random(f"factworld|hard|{self.config.id_namespace}|{self.config.seed}|{episode_seed}|{length}")
        idx = list(range(self.k))
        can_cycle = self.k >= self.config.min_cycle_len
        events: list[Event] = []
        for _ in range(length):
            if not can_cycle or rng.random() < self.config.swap_prob:
                i, j = rng.sample(idx, 2)
                events.append(Event("swap_role", (self.agents[i], self.agents[j])))
            else:
                m = rng.randint(self.config.min_cycle_len, self.k)
                cyc = rng.sample(idx, m)
                events.append(Event("cycle_roles", tuple(self.agents[c] for c in cyc)))
        return events
