"""Run or resume the local pretrained pilot."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import subprocess
import time

import torch
from transformers import AutoTokenizer, Qwen3_5ForCausalLM
from huggingface_hub.utils import disable_progress_bars

from .causal import (continuation_score, continuation_trace, effect, exchange_state, exchange_value_kv,
                     prefix_cache, reset_history, restrict_global_history)
from .data import Pair, answer_correct, make_pair
from .observe import (forward_score, greedy_answer, patch_query, source_sensitivity,
                      summarize_trace)

MODEL = "Qwen/Qwen3.5-4B"
REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
PROTOCOL = 2


def write_json(path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def append_json(path, data):
    with path.open("a") as f:
        f.write(json.dumps(data, allow_nan=False) + "\n")
        f.flush()


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def log(message):
    print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')} UTC] {message}", flush=True)


def provenance():
    import factworld
    root = Path(factworld.__file__).resolve().parent.parent
    result = {"source": "FactWorld minimal data/oracle/renderer snapshot", "file_sha256": {}}
    for name in ("config.py", "world.py", "oracle.py", "render.py"):
        result["file_sha256"][name] = hashlib.sha256((root / "factworld" / name).read_bytes()).hexdigest()
    if hasattr(factworld, "SOURCE_REVISION"):
        result["git_revision"] = factworld.SOURCE_REVISION
    else:
        git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True)
        result["git_revision"] = git.stdout.strip() if git.returncode == 0 else None
    return result


def select_candidates(records, layer_types, count):
    usable = [r for r in records if r["both_answers_correct"]]
    if not usable:
        return [], {}
    scores = {i: sum(abs(r["a"]["trace"]["lens_mixer_change"][i] -
                        r["b"]["trace"]["lens_mixer_change"][i]) for r in usable) / len(usable)
              for i in range(len(layer_types))}
    selected = []
    for kind in ("linear_attention", "full_attention"):
        eligible = [i for i, t in enumerate(layer_types) if t == kind]
        selected += sorted(eligible, key=lambda i: (-scores[i], i))[:count]
    return sorted(selected), {str(i): scores[i] for i in scores}


def baseline(model, tokenizer, pairs, output):
    done = {r["pair_id"]: r for r in read_jsonl(output / "baseline.jsonl")}
    trace_dir = output / "traces"
    trace_dir.mkdir(exist_ok=True)
    for number, pair in enumerate(pairs, 1):
        if pair.pair_id in done:
            if not (trace_dir / f"{pair.pair_id}.pt").exists():
                raise FileNotFoundError(f"Missing saved trace for {pair.pair_id}")
            continue
        torch.cuda.reset_peak_memory_stats()
        record = {"pair_id": pair.pair_id, "seed": pair.seed, "nominal_distance": pair.nominal_distance,
                  "distance_a": pair.distance_a, "distance_b": pair.distance_b,
                  "scoring_tokens": len(pair.ids_a), "differing_index": pair.differing_index}
        tensors = {}
        for side in ("a", "b"):
            prediction = greedy_answer(model, tokenizer, getattr(pair, f"base_ids_{side}"))
            score, trace = forward_score(model, getattr(pair, f"ids_{side}"), pair, capture=True)
            score["generated_answer"] = prediction
            score["answer_correct"] = answer_correct(prediction, getattr(pair, f"answer_{side}"))
            score["first_difference_correct"] = score["argmax_token"] == getattr(pair, f"token_{side}")
            score["trace"] = summarize_trace(model, pair, trace)
            record[side], tensors[side] = score, trace
        record["both_answers_correct"] = record["a"]["answer_correct"] and record["b"]["answer_correct"]
        record["both_first_differences_correct"] = record["a"]["first_difference_correct"] and record["b"]["first_difference_correct"]
        record["source_sensitivity"] = source_sensitivity(tensors["a"], tensors["b"])
        record["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
        record["peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 2**30
        tmp = trace_dir / f"{pair.pair_id}.tmp"
        torch.save(tensors, tmp)
        tmp.replace(trace_dir / f"{pair.pair_id}.pt")
        append_json(output / "baseline.jsonl", record)
        done[pair.pair_id] = record
        log(f"baseline {number}/{len(pairs)} {pair.pair_id}: "
            f"A={record['a']['generated_answer'].strip()!r} B={record['b']['generated_answer'].strip()!r}, "
            f"both correct={record['both_answers_correct']}, VRAM={record['peak_reserved_gib']:.2f} GiB")
        del tensors
    return [done[p.pair_id] for p in pairs]


def run_causal(model, pairs, records, output, count, per_type, window, tolerance,
               query_mode="full", fixed_layers=None):
    kinds = model.config.layer_types
    layers, scores = (sorted(fixed_layers), {}) if fixed_layers else select_candidates(records, kinds, per_type)
    write_json(output / "candidates.json", {"layers_zero_based": layers, "layer_types": kinds,
               "ranking": "explicit fixed layers" if fixed_layers else "mean absolute paired difference in mixer logit-lens change on both-correct pairs",
               "scores": scores, "selection_uses_current_run_data": fixed_layers is None,
               "query_patch_mode": query_mode, "count_per_type": per_type})
    by_id = {p.pair_id: p for p in pairs}
    selected = []
    for distance in sorted({p.nominal_distance for p in pairs}):
        eligible = [r for r in records if r["nominal_distance"] == distance and
                    r["both_answers_correct"] and r["both_first_differences_correct"]]
        selected += eligible[:count]
    log(f"Causal sample: {len(selected)} pairs, candidate layers (zero-based) {layers}")
    completed = {r["pair_id"] for r in read_jsonl(output / "causal_completed.jsonl")}
    existing = {tuple(r[k] for k in ("pair_id", "side", "layer", "component", "control"))
                for r in read_jsonl(output / "causal.jsonl")}

    def save(pair, side, layer, component, patched, dest, src, control=False, cache_mode=False):
        record = {"pair_id": pair.pair_id, "seed": pair.seed, "nominal_distance": pair.nominal_distance,
                  "side": side, "layer": layer, "layer_type": kinds[layer], "component": component,
                  "control": control, "cached_baseline": cache_mode,
                  "baseline_margin": dest, "donor_margin": src, "patched_margin": patched,
                  "correct_margin_change": (1 if side == "a" else -1) * (patched - dest),
                  **effect(patched, dest, src)}
        if control and abs(patched - dest) > tolerance:
            raise RuntimeError(f"No-op {component} at {pair.pair_id} layer {layer}: drift {patched - dest}")
        key = (pair.pair_id, side, layer, component, control)
        if key not in existing:
            append_json(output / "causal.jsonl", record)
            existing.add(key)

    for number, r in enumerate(selected, 1):
        pair = by_id[r["pair_id"]]
        if pair.pair_id in completed:
            continue
        boundary = len(pair.ids_a) - window
        if boundary <= max(p for span in pair.value_positions for p in span):
            raise ValueError("History window includes a source; choose a shorter window")
        caches = {s: prefix_cache(model, getattr(pair, f"ids_{s}"), boundary) for s in ("a", "b")}
        if query_mode == "cached":
            captured = {s: continuation_trace(model, getattr(pair, f"ids_{s}"), boundary, caches[s], pair)
                        for s in ("a", "b")}
            chunked = {s: captured[s][0] for s in captured}
            trace = {s: captured[s][1] for s in captured}
            del captured
        else:
            chunked = {s: continuation_score(model, getattr(pair, f"ids_{s}"), boundary, caches[s], pair)
                       for s in ("a", "b")}
            trace = torch.load(output / "traces" / f"{pair.pair_id}.pt", weights_only=True, map_location="cpu")
        for side, donor in (("a", "b"), ("b", "a")):
            ids = getattr(pair, f"ids_{side}")
            clean = chunked if query_mode == "cached" else r
            dest, src = clean[side]["margin_a_minus_b"], clean[donor]["margin_a_minus_b"]
            for layer in layers:
                for component, stream in (("residual", "after"), ("mixer", "mixer"), ("mlp", "mlp")):
                    for control in (True, False):
                        key = (pair.pair_id, side, layer, component, control)
                        if key in existing:
                            continue
                        vector = trace[side if control else donor][stream][layer, -1]
                        with patch_query(model, layer, component, vector):
                            if query_mode == "cached":
                                score = continuation_score(model, ids, boundary, caches[side], pair)
                            else:
                                score, _ = forward_score(model, ids, pair)
                        save(pair, side, layer, component, score["margin_a_minus_b"], dest, src,
                             control, query_mode == "cached")
            log(f"query patches {number}/{len(selected)} {pair.pair_id} side {side} complete")

        controls = {"pair_id": pair.pair_id, "boundary": boundary, "history_window": window,
                    "recurrent_state_dtype": str(next(x.recurrent_states.dtype for x in caches['a'].layers
                                                      if hasattr(x, 'recurrent_states'))),
                    "full_to_chunked_margin_drift": {s: chunked[s]["margin_a_minus_b"] - r[s]["margin_a_minus_b"]
                                                     for s in ("a", "b")}}
        for side, donor in (("a", "b"), ("b", "a")):
            ids = getattr(pair, f"ids_{side}")
            dest, src = chunked[side]["margin_a_minus_b"], chunked[donor]["margin_a_minus_b"]
            for layer in layers:
                component = "recurrent_state" if kinds[layer] == "linear_attention" else "value_kv"
                for control in (True, False):
                    key = (pair.pair_id, side, layer, component, control)
                    if key in existing:
                        continue
                    donor_cache = caches[side if control else donor]
                    changed = (exchange_state(caches[side], donor_cache, layer) if component == "recurrent_state"
                               else exchange_value_kv(caches[side], donor_cache, layer, pair.value_positions))
                    score = continuation_score(model, ids, boundary, changed, pair)
                    save(pair, side, layer, component, score["margin_a_minus_b"], dest, src, control, True)
                    del changed
                key = (pair.pair_id, side, layer, "history_restriction", False)
                if key not in existing:
                    if kinds[layer] == "linear_attention":
                        changed = reset_history(caches[side], layer)
                        score = continuation_score(model, ids, boundary, changed, pair)
                        del changed
                    else:
                        with restrict_global_history(model, layer, boundary):
                            score = continuation_score(model, ids, boundary, caches[side], pair)
                    save(pair, side, layer, "history_restriction", score["margin_a_minus_b"], dest, src,
                         cache_mode=True)
            log(f"state/KV/history patches {number}/{len(selected)} {pair.pair_id} side {side} complete")
        append_json(output / "cache_controls.jsonl", controls)
        append_json(output / "causal_completed.jsonl", {"pair_id": pair.pair_id})
        del caches, trace
    return layers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("baseline", "causal", "all", "report"), default="all")
    parser.add_argument("--output", type=Path, default=Path("results/pilot-20261014"))
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument("--pairs", type=int, default=20, help="pairs per distance")
    parser.add_argument("--distances", type=int, nargs="+", default=[128, 1024, 4096, 8192])
    parser.add_argument("--distractors", type=int, default=16)
    parser.add_argument("--causal-pairs", type=int, default=3, help="first solved pairs per distance")
    parser.add_argument("--candidate-layers-per-type", type=int, default=4)
    parser.add_argument("--history-window", type=int, default=64)
    parser.add_argument("--no-op-tolerance", type=float, default=0.05)
    parser.add_argument("--query-patch-mode", choices=("full", "cached"), default="full")
    parser.add_argument("--candidate-layers", type=int, nargs="+", help="fixed zero-based candidate layers")
    parser.add_argument("--min-answer-accuracy", type=float, default=0.)
    parser.add_argument("--min-pair-accuracy", type=float, default=0.)
    args = parser.parse_args()
    if args.stage == "report":
        from .report import build_report
        build_report(args.output)
        return
    if args.pairs < 1 or args.causal_pairs < 0 or args.candidate_layers_per_type < 1:
        parser.error("Need positive pairs/layer counts and a nonnegative causal count")
    if len(set(args.distances)) != len(args.distances) or args.history_window < 1:
        parser.error("Distances must be unique and history window positive")
    if args.candidate_layers and (len(set(args.candidate_layers)) != len(args.candidate_layers) or
                                  min(args.candidate_layers) < 0 or max(args.candidate_layers) >= 32):
        parser.error("Fixed candidate layers must be unique indices from 0 through 31")
    if not (0 <= args.min_answer_accuracy <= 1 and 0 <= args.min_pair_accuracy <= 1):
        parser.error("Accuracy gates must be between zero and one")
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = {"version": PROTOCOL, "model": args.model, "revision": args.revision, "seed": args.seed,
                "pairs_per_distance": args.pairs, "distances": args.distances, "distractors": args.distractors,
                "causal_pairs_per_distance": args.causal_pairs, "candidate_layers_per_type": args.candidate_layers_per_type,
                "history_window": args.history_window, "no_op_tolerance": args.no_op_tolerance,
                "query_patch_mode": args.query_patch_mode, "fixed_candidate_layers": args.candidate_layers,
                "min_answer_accuracy": args.min_answer_accuracy, "min_pair_accuracy": args.min_pair_accuracy,
                "factworld": provenance()}
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["protocol"] != protocol:
            raise ValueError("Existing run has a different protocol/provenance; choose a new output directory")
    else:
        manifest = {"protocol": protocol, "started_utc": datetime.now(timezone.utc).isoformat(),
                    "python": platform.python_version(),
                    "packages": {name: importlib.metadata.version(name) for name in
                                 ("torch", "transformers", "flash-linear-attention", "numpy", "huggingface-hub", "accelerate")},
                    "implementation_sha256": {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                                              for name in ("data.py", "observe.py", "causal.py", "run.py")},
                    "uv_lock_sha256": hashlib.sha256(Path("uv.lock").read_bytes()).hexdigest() if Path("uv.lock").exists() else None,
                    "lens_arithmetic": "FP32 dot product of BF16 final-normalized residual with the unembedding difference; endpoint can differ from BF16 lm_head rounding"}
        write_json(manifest_path, manifest)
    disable_progress_bars()
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    pairs = [make_pair(tokenizer, d, i, args.seed, args.distractors)
             for d in args.distances for i in range(args.pairs)]
    pair_path = args.output / "pairs.jsonl"
    if not pair_path.exists():
        for pair in pairs:
            append_json(pair_path, pair.to_dict())
    elif read_jsonl(pair_path) != [p.to_dict() for p in pairs]:
        raise ValueError("Saved pairs differ from regenerated pairs")
    log(f"Validated {len(pairs)} pairs; loading frozen {args.model}@{args.revision[:12]}")
    torch.set_num_threads(8)
    torch.manual_seed(args.seed)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("This pilot requires a CUDA GPU with BF16 support")
    start = time.perf_counter()
    model = Qwen3_5ForCausalLM.from_pretrained(args.model, revision=args.revision,
              dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa").eval().requires_grad_(False)
    first_linear = next(l.linear_attn for l in model.model.layers if hasattr(l, "linear_attn"))
    if first_linear.chunk_gated_delta_rule.__module__.startswith("transformers"):
        raise RuntimeError("FLA kernel is unavailable; refusing the slow reference path")
    manifest.update(gpu=torch.cuda.get_device_name(0), layer_types=model.config.layer_types,
                    model_load_seconds=time.perf_counter() - start,
                    delta_kernel=first_linear.chunk_gated_delta_rule.__module__,
                    convolution_kernel="torch" if first_linear.causal_conv1d_fn is None else "causal_conv1d",
                    weights_gib=sum(p.numel()*p.element_size() for p in model.parameters()) / 2**30)
    write_json(manifest_path, manifest)
    log(f"Model ready in {manifest['model_load_seconds']:.1f}s; FLA active, convolution={manifest['convolution_kernel']}")
    if args.stage in ("baseline", "all"):
        records = baseline(model, tokenizer, pairs, args.output)
    else:
        records = read_jsonl(args.output / "baseline.jsonl")
        if {r["pair_id"] for r in records} != {p.pair_id for p in pairs}:
            raise ValueError("Complete the baseline stage before causal interventions")
    if args.stage in ("causal", "all"):
        for distance in args.distances:
            group = [r for r in records if r["nominal_distance"] == distance]
            accuracy = sum(r[s]["answer_correct"] for r in group for s in ("a", "b")) / (2 * len(group))
            paired = sum(r["both_answers_correct"] for r in group) / len(group)
            if accuracy < args.min_answer_accuracy or paired < args.min_pair_accuracy:
                manifest["stopped_reason"] = f"Accuracy gate failed at {distance}: answers={accuracy}, pairs={paired}"
                write_json(manifest_path, manifest)
                from .report import build_report
                build_report(args.output)
                raise RuntimeError(manifest["stopped_reason"])
        run_causal(model, pairs, records, args.output, args.causal_pairs,
                   args.candidate_layers_per_type, args.history_window, args.no_op_tolerance,
                   args.query_patch_mode, args.candidate_layers)
    manifest.update(finished_utc=datetime.now(timezone.utc).isoformat(), last_stage=args.stage,
                    peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30)
    write_json(manifest_path, manifest)
    from .report import build_report
    build_report(args.output)
    log(f"Finished. Results: {args.output.resolve()}")


if __name__ == "__main__":
    main()
