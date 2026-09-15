"""Key-order intervention without changes to facts, answer values, or scoring.

``decision_first`` reproduces the preceding study's exact prompt and canonical
answer. ``decision_last`` rotates the four top-level keys only. The frozen task
parser remains insensitive to key order. No token-count equality is assumed.
"""
from __future__ import annotations

import copy
import json
from collections.abc import Mapping

try:
    from .task import (OUTPUT_FIELDS, canonical_json, gold, parse_output,
                       render_prompt, validate_program)
except ImportError:
    from task import (OUTPUT_FIELDS, canonical_json, gold, parse_output,
                      render_prompt, validate_program)

ORDERS = ("decision_first", "decision_last")
KEY_ORDERS = {
    "decision_first": ("decision", "selected", "count", "reason"),
    "decision_last": ("selected", "count", "reason", "decision"),
}
FIRST_CONTRACT_FRAGMENT = (
    '"decision", "selected", "count", "reason". Prefer that order with decision first; object key order '
)
LAST_CONTRACT_FRAGMENT = (
    '"selected", "count", "reason", "decision". Prefer that order with decision last; object key order '
)


def _check_order(order: str) -> None:
    if order not in ORDERS:
        raise ValueError(f"Unknown output order: {order!r}")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON object key: {key}")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"Non-JSON constant: {value}")


def transform_target(target: str | Mapping, order: str) -> str:
    """Serialize a supplied answer, including an intentionally wrong target.

    Never substitutes the oracle for supplied values. Invalid answer structures
    fail rather than silently dropping keys or converting malformed JSON.
    """
    _check_order(order)
    if isinstance(target, str):
        obj = json.loads(target, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    elif isinstance(target, Mapping):
        obj = copy.deepcopy(dict(target))
    else:
        raise TypeError("Target must be a JSON string or answer mapping")
    if not isinstance(obj, dict) or set(obj) != OUTPUT_FIELDS:
        raise ValueError("Target must contain exactly the four answer fields")
    if (type(obj["decision"]) is not str or obj["decision"] not in {"REPORT", "CLEAR"}
            or not isinstance(obj["selected"], list)
            or not all(type(x) is str for x in obj["selected"])
            or len(set(obj["selected"])) != len(obj["selected"])
            or type(obj["count"]) is not int or not 0 <= obj["count"] <= 8
            or not validate_program(obj["reason"])):
        raise ValueError("Target has an invalid answer field")
    # Normalize nested reason keys identically in both conditions, then rotate
    # only the top-level canonical response keys.
    canonical = json.loads(canonical_json(obj))
    ordered = {key: canonical[key] for key in KEY_ORDERS[order]}
    return json.dumps(ordered, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def render_case(row: dict, order: str) -> str:
    """Regenerate from whitelisted case facts; only the answer contract changes."""
    _check_order(order)
    prompt = render_prompt(row)
    if order == "decision_first":
        return prompt
    if prompt.count(FIRST_CONTRACT_FRAGMENT) != 1:
        raise RuntimeError("The authoritative prompt contract changed unexpectedly")
    return prompt.replace(FIRST_CONTRACT_FRAGMENT, LAST_CONTRACT_FRAGMENT, 1)


def answer_text(row: dict, order: str, target: str | Mapping | None = None) -> str:
    """Use an explicit target, then stored target, then oracle, in that order."""
    if target is None:
        target = row["target"] if "target" in row else gold(row)
    rendered = transform_target(target, order)
    if not parse_output(rendered, row)["format_valid"]:
        raise ValueError("Supplied target is invalid for this case's row IDs/schema")
    return rendered


def serialized_case(row: dict, order: str) -> dict:
    """Optional copying helper; record order without mutating the source row.

    Repair message construction is owned by the cohort builder, which should
    likewise record output_order after assembling messages in this order.
    """
    result = copy.deepcopy(row)
    result["prompt"] = render_case(row, order)
    result["target"] = answer_text(row, order)
    if "gold_target" in row:
        result["gold_target"] = transform_target(row["gold_target"], order)
    result["output_order"] = order
    return result
