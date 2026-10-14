"""Deterministic natural-language renderer + its inverse parser.

FactWorld renders every statement as **clean natural language with attached
punctuation**: facts as ``g0's a0 is v18.``, events as ``s0 gives o0 to g0.``,
cycles as ``s0 cycles roles: g0 -> g1 -> g2.``  One fixed phrasing per statement
type (no paraphrase variety) so the model sees a uniform grammar — this is the
single canonical format; the earlier space-separated "atomic-token" v1 format
lives in git history.

Content tokens are still atomic IDs (``e17 a3 v42 o2 loc1 g4 r0 s5``); the step
label ``sN`` is the event subject. The render <-> parse round-trip is a contract
(the ground-truth re-parse check — every rendered document must parse back to the
KB record it encodes). ``normalize()`` detaches attached punctuation
(``v109.`` -> ``v109 .``, ``g0's`` -> ``g0 's``) so scoring and parsing operate on
a canonical whitespace-token form regardless of how a model glued its output.
"""
from __future__ import annotations

import re
import zlib

from .world import Event

# Classify an atomic token by its type prefix (optionally namespaced, e.g. "aux1_g0").
# 'loc' must precede the single-char alternatives so "loc3" isn't read as an object.
_TOK = re.compile(r"^(?:[A-Za-z0-9]+_)?(loc|[eavogrs])(\d+)$")


def classify(token: str) -> str | None:
    m = _TOK.match(token)
    return m.group(1) if m else None


def _is_content_id(token: str) -> bool:
    return classify(token) is not None


class Renderer:
    """The single FactWorld renderer: clean natural language, attached punctuation.

    Each statement type has exactly one phrasing (a 1-tuple, kept as a tuple so a
    future paraphrase pass can slot in variants without touching call sites). The
    step label is the event subject: ``s0 gives o0 to g0.``.
    """

    _FACT = ("{e}'s {a} is {v}.",)
    _MOVE = ("moves {o} to {h}.",)
    _GIVE = ("gives {o} to {h}.",)
    _SWAP = ("swaps {a} and {b}.",)
    _CYCLE = ("cycles roles: {flows}.",)
    _ROLE = ("{g} has role {r}.",)
    _HOLDER = ("{h} holds {o}.",)

    # Role-flow arrow in the compact cycle notation: "g0 -> g1 -> g2" means g0's role
    # passes to g1, g1's to g2, and g2's back to g0 (the canonical cycle_roles args).
    _CYCLE_ARROW = " -> "

    @staticmethod
    def _pick(options: tuple[str, ...], key: str) -> str:
        return options[zlib.crc32(key.encode()) % len(options)]  # deterministic across runs

    # ----- render -----
    def render_fact(self, entity: str, attribute: str, value: str, key: str | None = None) -> str:
        return self._pick(self._FACT, key or f"fact|{entity}|{attribute}").format(
            e=entity, a=attribute, v=value
        )

    def render_event(self, event: Event, step: str | None = None, key: str | None = None) -> str:
        """Render an event. ``step`` (e.g. "s0") is the subject; when omitted the bare
        predicate is returned (used only by tokenizer probing)."""
        k = key or f"ev|{event.kind}|{'|'.join(event.args)}"
        if event.kind == "move":
            s = self._pick(self._MOVE, k).format(o=event.args[0], h=event.args[1])
        elif event.kind == "give":
            s = self._pick(self._GIVE, k).format(o=event.args[0], h=event.args[1])
        elif event.kind == "swap_role":
            s = self._pick(self._SWAP, k).format(a=event.args[0], b=event.args[1])
        elif event.kind == "cycle_roles":
            s = self._pick(self._CYCLE, k).format(flows=self._CYCLE_ARROW.join(event.args))
        else:
            raise ValueError(f"unknown event kind {event.kind!r}")
        return f"{step} {s}" if step is not None else s

    def render_history(self, events, with_steps: bool = True) -> list[str]:
        # with_steps is kept for API stability but events always carry a subject; when
        # False we still emit the step label because the natural grammar requires one.
        return [
            self.render_event(e, step=f"s{i}",
                              key=f"h|{i}|{e.kind}|{'|'.join(e.args)}")
            for i, e in enumerate(events)
        ]

    def render_scenario(self, idx: int, width: int = 4) -> str:
        """Scenario id as a marker + shared digit tokens, e.g. 'scn #0 #0 #4 #2'. Binding is
        compositional over a 10-token digit vocab (no unique per-scenario embedding); zero-padded
        to a fixed width for clean positional reading."""
        return "scn " + " ".join(f"#{c}" for c in str(idx).zfill(width))

    def render_role(self, agent: str, role: str, step: str | None = None, key: str | None = None) -> str:
        return self._pick(self._ROLE, key or f"role|{agent}").format(g=agent, r=role)

    def render_holder(self, obj: str, holder: str, step: str | None = None, key: str | None = None) -> str:
        return self._pick(self._HOLDER, key or f"holder|{obj}").format(o=obj, h=holder)

    def render_query(self, family: str, *, entity=None, attribute=None, target=None, t=None) -> str:
        # as-of-t references the (t-1)-th event label; t=None means the final state
        step = None if t is None else f"s{t - 1}"
        if family == "recall":
            return f"what is {attribute} of {entity}?"
        if family == "state_easy":
            # "where is X?" invites a list of locations; be explicit about the final holder.
            return (f"who is the final holder of {target}?" if step is None
                    else f"who holds {target} at {step}?")
        if family == "state_hard":
            return (f"what role does {target} have?" if step is None
                    else f"what role does {target} have at {step}?")
        raise ValueError(f"unknown query family {family!r}")

    # ----- attached-punctuation -> canonical whitespace normalization -----
    @staticmethod
    def normalize(text: str) -> str:
        """Detach attached punctuation so scoring/parsing work on canonical whitespace tokens.

        Examples:
            "g9's a0 is v26." -> "g9 's a0 is v26 ."
            "s1 gives o0 to g0." -> "s1 gives o0 to g0 ."
            "what is a0 of g7?" -> "what is a0 of g7 ?"
        """
        text = text.strip()
        text = re.sub(r"([a-zA-Z0-9]+)'s\b", r"\1 's", text)   # g9's -> g9 's
        text = re.sub(r"(?<=\S)([.,?!])", r" \1", text)          # v26. -> v26 .
        return text

    # ----- parse (exact inverse) -----
    def _typed(self, text: str):
        buckets: dict[str, list[str]] = {t: [] for t in ("e", "a", "v", "o", "loc", "g", "r", "s")}
        toks = text.split()
        for tk in toks:
            c = classify(tk)
            if c:
                buckets[c].append(tk)
        return buckets, toks

    def parse(self, text: str) -> dict:
        # Normalize attached punctuation back to canonical whitespace tokens before parsing.
        text = self.normalize(text)
        typed, toks = self._typed(text)
        step = typed["s"][0] if typed["s"] else None
        if "?" in toks:
            # state_easy queries are interrogated with 'where'/'who'. Do NOT key on the words
            # 'holder'/'holds': the composite recall query 'what is a0 of the holder of o3 ?'
            # also contains 'holder' and would be misrouted, breaking the round-trip.
            if "where" in toks or "who" in toks:
                return {"type": "query", "family": "state_easy", "target": typed["o"][0], "step": step}
            if "role" in toks:
                return {"type": "query", "family": "state_hard", "target": typed["g"][0], "step": step}
            # recall: a plain entity (e-token) or the composite 'holder of {obj}' phrase.
            if typed["e"]:
                return {"type": "query", "family": "recall",
                        "entity": typed["e"][0], "attribute": typed["a"][0] if typed["a"] else None}
            obj = typed["o"][0] if typed["o"] else None
            return {"type": "query", "family": "recall",
                    "entity": f"the holder of {obj}" if obj else None,
                    "attribute": typed["a"][0] if typed["a"] else None, "object": obj}
        if "swap" in toks or "swaps" in toks:
            return {"type": "event", "event": Event("swap_role", tuple(typed["g"])), "step": step}
        if "cycle" in toks or "cycles" in toks:
            return {"type": "event", "event": Event("cycle_roles", tuple(typed["g"])), "step": step}
        if any(w in toks for w in ("move", "moved", "moves", "give", "given", "gives", "receives")):
            kind = "move" if any(w in toks for w in ("move", "moved", "moves")) else "give"
            holder = (typed["loc"] + typed["g"])[0]  # the single non-object holder (location or agent)
            return {"type": "event", "event": Event(kind, (typed["o"][0], holder)), "step": step}
        if typed["g"] and typed["r"]:                                  # role assertion (worked-trace line)
            return {"type": "role", "agent": typed["g"][0], "role": typed["r"][0], "step": step}
        if typed["o"] and (typed["loc"] or typed["g"]):                # holder assertion (easy answer line)
            return {"type": "holder", "object": typed["o"][0], "holder": (typed["loc"] + typed["g"])[0], "step": step}
        return {"type": "fact", "entity": typed["e"][0], "attribute": typed["a"][0], "value": typed["v"][0]}

    def parse_history(self, lines) -> list[Event]:
        return [self.parse(line)["event"] for line in lines]
