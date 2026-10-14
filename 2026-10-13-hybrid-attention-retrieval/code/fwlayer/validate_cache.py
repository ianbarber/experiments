"""Compare cached query interventions with the original unsplit execution."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import torch
from transformers import Qwen3_5ForCausalLM
from huggingface_hub.utils import disable_progress_bars

from .causal import continuation_score, continuation_trace, effect, prefix_cache
from .data import Pair
from .observe import forward_score, patch_query
from .run import MODEL, REVISION, log, read_jsonl, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, default=Path("results/pilot-20261014"))
    parser.add_argument("--output", type=Path, default=Path("results/cache-validation.json"))
    args = parser.parse_args()
    pairs = [Pair(**p) for p in read_jsonl(args.pilot / "pairs.jsonl")
             if p["index"] == 0 and p["nominal_distance"] in (128, 8192)]
    layers = json.loads((args.pilot / "candidates.json").read_text())["layers_zero_based"]
    disable_progress_bars()
    torch.set_num_threads(8)
    model = Qwen3_5ForCausalLM.from_pretrained(MODEL, revision=REVISION, dtype=torch.bfloat16,
                 device_map="cuda", attn_implementation="sdpa").eval().requires_grad_(False)
    result = {"started_utc": datetime.now(timezone.utc).isoformat(), "maximum_allowed_fraction_discrepancy": .03,
              "maximum_allowed_self_patch_drift": .05, "layers_zero_based": layers,
              "comparisons": [], "controls": [], "baseline_drift": []}
    for pair in pairs:
        full, full_trace = {}, {}
        for side in ("a", "b"):
            full[side], full_trace[side] = forward_score(model, getattr(pair, "ids_" + side), pair, capture=True)
        boundary = len(pair.ids_a) - 64
        caches = {s: prefix_cache(model, getattr(pair, "ids_" + s), boundary) for s in ("a", "b")}
        captured = {s: continuation_trace(model, getattr(pair, "ids_" + s), boundary, caches[s], pair)
                    for s in ("a", "b")}
        cached = {s: captured[s][0] for s in captured}
        cached_trace = {s: captured[s][1] for s in captured}
        for side, donor in (("a", "b"), ("b", "a")):
            ids = getattr(pair, "ids_" + side)
            result["baseline_drift"].append({"pair_id": pair.pair_id, "side": side,
                "margin_delta": cached[side]["margin_a_minus_b"] - full[side]["margin_a_minus_b"]})
            for layer in layers:
                for component, stream in (("residual", "after"), ("mixer", "mixer"), ("mlp", "mlp")):
                    fractions = {}
                    for mode in ("full", "cached"):
                        traces, clean = (full_trace, full) if mode == "full" else (cached_trace, cached)
                        for self_patch in (True, False):
                            vector = traces[side if self_patch else donor][stream][layer, -1]
                            with patch_query(model, layer, component, vector):
                                score = (forward_score(model, ids, pair)[0] if mode == "full" else
                                         continuation_score(model, ids, boundary, caches[side], pair))
                            delta = effect(score["margin_a_minus_b"], clean[side]["margin_a_minus_b"],
                                           clean[donor]["margin_a_minus_b"])
                            if self_patch:
                                result["controls"].append({"pair_id": pair.pair_id, "side": side,
                                     "layer": layer, "component": component, "mode": mode,
                                     "margin_delta": delta["margin_delta"]})
                            else:
                                fractions[mode] = delta["fraction_of_counterfactual_gap"]
                    result["comparisons"].append({"pair_id": pair.pair_id, "side": side, "layer": layer,
                           "component": component, "full_fraction": fractions["full"],
                           "cached_fraction": fractions["cached"],
                           "absolute_fraction_discrepancy": abs(fractions["full"] - fractions["cached"])})
            log(f"Execution comparison {pair.pair_id} side {side} complete")
        write_json(args.output, result)
        del full_trace, caches, captured, cached_trace
    result["max_absolute_fraction_discrepancy"] = max(r["absolute_fraction_discrepancy"] for r in result["comparisons"])
    result["max_self_patch_margin_drift"] = max(abs(r["margin_delta"]) for r in result["controls"])
    result["passed"] = result["max_absolute_fraction_discrepancy"] <= .03 and result["max_self_patch_margin_drift"] <= .05
    result["finished_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(args.output, result)
    log(f"Cached execution gate passed={result['passed']}; maximum fraction discrepancy "
        f"{result['max_absolute_fraction_discrepancy']:.6f}; self-patch drift {result['max_self_patch_margin_drift']:.6f}")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
