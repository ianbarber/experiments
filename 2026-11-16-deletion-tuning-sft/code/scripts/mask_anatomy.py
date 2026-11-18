"""What the trained deletion masks contain.

  python scripts/mask_anatomy.py              # writes ../results/mask_anatomy.json
  python scripts/mask_anatomy.py --ablations  # also writes the ablation masks under runs/
  python scripts/mask_anatomy.py ablation     # no GPU: adds the ablation scores once they exist

For every Deletion Tuning run: how much of the deployed mask was already in the starting mask (and
whether that part came from the Wanda score or from the init noise), how many starting deletions
were put back, when the flips happen, which blocks and which weights they land on, and how far
seeds and learning rates agree. Needs the GPU for the Wanda calibration pass (about a minute).

The ablations are parts of the deployed mask scored on their own (scripts/mask_ablation.sh): only
what training added, and that part split by whether the logit started below or above 0.02.
"""

import itertools
import json
import sys
import time

import numpy as np
import torch
import zstandard

from deltune import data, methods
from deltune.common import CODE_DIR, RUNS_DIR, dump_json, gate_modules, load_model, load_tokenizer

SEEDS = (42, 43, 44)
# cell -> (runs, the step that was deployed and scored)
CELLS = {
    "paper_rates": ([f"dt_lr5e-4_a100-1000_s{s}" for s in SEEDS], 1600),
    "paper_recipe": ([f"paper_dt_lr5e-4_s{s}" for s in SEEDS], 400),
    "tuned": ([f"dt_lr1e-3_a100-1000_s{s}" for s in SEEDS], 400),
}
SINGLE = {  # seed 42 only
    "lr3e-4": ("dt_lr3e-4_a100-1000_s42", 1600),
    "lr3e-3": ("dt_lr3e-3_a100-1000_s42", 400),
    "rl_ramp": ("dt_lr1e-3_a1000-10000_s42", 1600),
}
ETA = 1 / 2048
NV = 20  # rank bins within a matrix


def n(x) -> int:
    return int(x.sum())


def total(tensors) -> int:
    return sum(n(t) for t in tensors)


def saved_steps(run):
    return sorted(int(p.stem.split("step")[1]) for p in (RUNS_DIR / run).glob("mask_step*.zst"))


def deleted(run, step, shapes):
    blob = zstandard.ZstdDecompressor().decompress((RUNS_DIR / run / f"mask_step{step}.zst").read_bytes())
    return [(~k).cuda() for k in methods.unpack_mask(blob, shapes)]


def rank_bin(x):
    """Rank of every entry within its matrix, in NV equal bins (0 = smallest)."""
    order = torch.argsort(x.flatten())
    r = torch.empty_like(order)
    r[order] = torch.arange(order.numel(), device=x.device)
    return (r * NV // order.numel()).to(torch.uint8).view_as(x)


def hist(bins, sel):
    h = torch.zeros(NV, dtype=torch.long, device="cuda")
    for b, s in zip(bins, sel):
        h += torch.bincount(b[s].long(), minlength=NV)
    return h.tolist()


def top_share(counts, frac):
    """Share of all deletions held by the top `frac` of rows or columns."""
    c = torch.sort(counts.flatten().float(), descending=True).values
    k = max(1, int(round(frac * c.numel())))
    return float(c[:k].sum() / c.sum())


def concentration(sets):
    cols = torch.stack([s.sum(0) for s in sets])  # blocks x residual dims
    rows = torch.stack([s.sum(1) for s in sets])  # blocks x gate units
    width = sets[0].shape[1]
    return {
        "top1pct_units_share": top_share(rows, 0.01),
        "top10pct_units_share": top_share(rows, 0.10),
        "top1pct_dims_share": top_share(cols.sum(0), 0.01),
        "top10pct_dims_share": top_share(cols.sum(0), 0.10),
        "max_unit_frac_deleted": float(rows.max() / width),
        "units_over_5pct_deleted": n(rows > 0.05 * width),
    }, cols, rows


def corr(a, b):
    return float(torch.corrcoef(torch.stack([a.flatten().float(), b.flatten().float()]))[0, 1])


def ablation_scores():
    """Seen-list win rate and validation loss of each part of the paper-rates masks, where scored."""
    def score(run, step):
        path = RUNS_DIR / run / f"eval_step{step}.json"
        if not path.exists():
            return None
        e = json.loads(path.read_text())
        return {"seen": 100 * e["wordle"]["seen"]["win_rate"], "val_loss": e["val_loss"]}

    runs, step = CELLS["paper_rates"]
    rows = {}
    for run in runs:
        rows[run] = {"deployed": score(run, step), "learned_only": score(f"ablate_learned_only_{run}", step),
                     "learned_near": score(f"ablate_learned_near_{run}", step), "learned_far": score(f"ablate_learned_far_{run}", step)}
    return {"step": step, "runs": rows, "base": score("base", 0), "starting_mask": score("dt_lr1e-3_a100-1000_s42", 0)}


def main():
    if sys.argv[1:] == ["ablation"]:
        path = CODE_DIR.parent / "results" / "mask_anatomy.json"
        out = json.loads(path.read_text())
        out["ablation"] = ablation_scores()
        dump_json(out, path)
        print(json.dumps(out["ablation"], indent=1))
        return
    tok = load_tokenizer()
    model = load_model(dtype=torch.float32)
    train_games, _ = data.load_games()
    ex = [data.encode(tok, g) for g in train_games]
    t0 = time.time()
    norms, _ = methods.wanda_input_norms(
        model, (data.collate(ex[i : i + 16], tok.pad_token_id) for i in range(0, len(ex), 16))
    )
    wanda_seconds = round(time.time() - t0, 1)
    W = [m.weight.detach().float() for _, m in gate_modules(model)]
    shapes = [tuple(w.shape) for w in W]
    sizes = [w.numel() for w in W]
    N = sum(sizes)
    n_model = sum(p.numel() for p in model.parameters())
    low, mag_bin, score_bin = [], [], []
    for w, norm in zip(W, norms):
        score = w.abs() * norm[None, :]
        k = max(1, int(round(ETA * score.numel())))
        low.append(score <= torch.kthvalue(score.flatten(), k).values)
        mag_bin.append(rank_bin(w.abs()))
        score_bin.append(rank_bin(score))
    positive = [w > 0 for w in W]
    del model, W
    torch.cuda.empty_cache()

    out = {"gate_weights": N, "model_weights": n_model, "blocks": len(shapes), "gate_shape": list(shapes[0]),
           "wanda_low_set": total(low), "rank_bins": NV, "wanda_pass_seconds": wanda_seconds}

    def flow(run):
        steps = saved_steps(run)
        d0 = deleted(run, 0, shapes)
        rows, prev = [], d0
        for t in steps:
            d = deleted(run, t, shapes) if t else d0
            rows.append({
                "step": t,
                "deleted": total(d),
                "from_wanda": sum(n(a & b & lw) for a, b, lw in zip(d0, d, low)),
                "from_noise": sum(n(a & b & ~lw) for a, b, lw in zip(d0, d, low)),
                "learned": sum(n(b & ~a) for a, b in zip(d0, d)),
                "wanda_restored": sum(n(a & ~b & lw) for a, b, lw in zip(d0, d, low)),
                "noise_restored": sum(n(a & ~b & ~lw) for a, b, lw in zip(d0, d, low)),
                "new_since_previous": sum(n(b & ~p) for p, b in zip(prev, d)),
                "restored_since_previous": sum(n(p & ~b) for p, b in zip(prev, d)),
            })
            prev = d
        return rows

    def anatomy(run, step):
        d0 = deleted(run, 0, shapes)
        d = deleted(run, step, shapes)
        learned = [b & ~a for a, b in zip(d0, d)]
        noise_kept = [a & b & ~lw for a, b, lw in zip(d0, d, low)]
        noise_back = [a & ~b & ~lw for a, b, lw in zip(d0, d, low)]
        conc, cols, rows = concentration(learned)
        # the same count per block, placed uniformly at random
        gen = torch.Generator(device="cuda").manual_seed(0)
        rand = [torch.rand(s.shape, generator=gen, device="cuda") < n(s) / s.numel() for s in learned]
        rec = {
            "run": run, "step": step,
            "deleted": total(d), "deleted_pct_of_model": 100 * total(d) / n_model,
            "from_wanda": sum(n(a & b & lw) for a, b, lw in zip(d0, d, low)),
            "from_noise": total(noise_kept),
            "learned": total(learned),
            "initial": total(d0),
            "wanda_restored": sum(n(a & ~b & lw) for a, b, lw in zip(d0, d, low)),
            "noise_restored": total(noise_back),
            "learned_per_block_pct": [100 * n(s) / s.numel() for s in learned],
            "concentration": conc,
            "concentration_uniform": concentration(rand)[0],
            "learned_by_magnitude_bin": hist(mag_bin, learned),
            "learned_by_wanda_bin": hist(score_bin, learned),
            "noise_kept_by_magnitude_bin": hist(mag_bin, noise_kept),
            "noise_restored_by_magnitude_bin": hist(mag_bin, noise_back),
            "learned_positive_frac": sum(n(s & p) for s, p in zip(learned, positive)) / max(1, total(learned)),
        }
        return rec, learned, cols, rows

    def agreement(a, b):
        inter = sum(n(x & y) for x, y in zip(a["learned"], b["learned"]))
        chance = sum(n(x) * n(y) / x.numel() for x, y in zip(a["learned"], b["learned"]))
        na, nb = total(a["learned"]), total(b["learned"])
        return {
            "shared": inter, "expected_by_chance": chance, "times_chance": inter / chance,
            "jaccard": inter / (na + nb - inter), "share_of_smaller": inter / min(na, nb),
            "unit_count_corr": corr(a["rows"], b["rows"]),
            "dim_count_corr": corr(a["cols"].sum(0), b["cols"].sum(0)),
            "block_profile_corr": corr(a["rows"].sum(1), b["rows"].sum(1)),
        }

    out["flow"], out["final"], out["seed_agreement"] = {}, {}, {}
    keep42 = {}
    for cell, (runs, step) in CELLS.items():
        held = {}
        for run in runs:
            out["flow"][run] = flow(run)
            rec, learned, cols, rows = anatomy(run, step)
            out["final"][run] = rec
            held[run] = {"learned": learned, "cols": cols, "rows": rows}
            print(cell, run, {k: rec[k] for k in ("deleted", "from_wanda", "from_noise", "learned", "noise_restored", "wanda_restored")}, flush=True)
        out["seed_agreement"][cell] = {f"{a[-3:]}-{b[-3:]}": agreement(held[a], held[b]) for a, b in itertools.combinations(runs, 2)}
        all3 = sum(n(x & y & z) for x, y, z in zip(*(held[r]["learned"] for r in runs)))
        out["seed_agreement"][cell]["in_all_three"] = all3
        keep42[cell] = held[runs[0]]
        del held
        torch.cuda.empty_cache()
    for name, (run, step) in SINGLE.items():
        out["flow"][run] = flow(run)
        rec, learned, cols, rows = anatomy(run, step)
        out["final"][run] = rec
        keep42[name] = {"learned": learned, "cols": cols, "rows": rows}
        print(name, run, {k: rec[k] for k in ("deleted", "from_wanda", "from_noise", "learned", "noise_restored", "wanda_restored")}, flush=True)
    # Where a weight's logit started decides a lot. Rebuild the starting logits as training drew
    # them (same generator, same order) and ask how often a weight is deleted at the end, by start.
    edges = torch.arange(-0.07, 0.0901, 0.005, device="cuda")
    out["by_start"] = {"edges": [round(float(e), 3) for e in edges], "runs": {}, "split_at": 0.02, "split": {}}
    far_sets = {}
    for seed in SEEDS:
        gen = torch.Generator(device="cuda").manual_seed(seed)
        start = [torch.where(lw, -0.03, 0.03) + 0.01 * torch.randn(lw.shape, generator=gen, device="cuda") for lw in low]
        start_bin = [torch.bucketize(g, edges).clamp_(1, len(edges) - 1) - 1 for g in start]
        nb = len(edges) - 1
        everything = sum(torch.bincount(b.flatten(), minlength=nb) for b in start_bin).tolist()
        runs = [(CELLS[c][0][SEEDS.index(seed)], CELLS[c][1]) for c in CELLS]
        if seed == 42:
            runs += list(SINGLE.values())
        for run, step in runs:
            d0, d = deleted(run, 0, shapes), deleted(run, step, shapes)
            mismatch = sum(n((g <= 0) != a) for g, a in zip(start, d0))
            gone = sum(torch.bincount(b[x], minlength=nb) for b, x in zip(start_bin, d)).tolist()
            out["by_start"]["runs"][run] = {"step": step, "rebuilt_start_mismatches": mismatch, "weights": everything, "deleted": gone}
            if run in CELLS["paper_rates"][0]:
                learned = [b & ~a for a, b in zip(d0, d)]
                far_sets[run] = {"near": [x & (g < 0.02) for x, g in zip(learned, start)],
                                 "far": [x & (g >= 0.02) for x, g in zip(learned, start)], "step": step}
                out["by_start"]["split"][run] = {k: total(far_sets[run][k]) for k in ("near", "far")}
            print("by start", run, "mismatches", mismatch, flush=True)
        del start, start_bin
        torch.cuda.empty_cache()

    # same seed, different schedules and learning rates
    out["config_agreement_seed42"] = {f"{a}-{b}": agreement(keep42[a], keep42[b]) for a, b in itertools.combinations(keep42, 2)}

    if "--ablations" in sys.argv:
        # the deployed mask with the starting deletions put back: only what training added
        for run in CELLS["paper_rates"][0]:
            step = CELLS["paper_rates"][1]
            d0, d = deleted(run, 0, shapes), deleted(run, step, shapes)
            keep = [~(b & ~a) for a, b in zip(d0, d)]
            dst = RUNS_DIR / f"ablate_learned_only_{run}"
            dst.mkdir(exist_ok=True)
            (dst / "config.json").write_text((RUNS_DIR / run / "config.json").read_text())
            bits = np.concatenate([np.packbits(k.cpu().numpy().ravel()) for k in keep]).tobytes()
            (dst / f"mask_step{step}.zst").write_bytes(zstandard.ZstdCompressor(level=19).compress(bits))
            print("wrote", dst.name, "deleted", sum(n(~k) for k in keep), flush=True)
            for part in ("near", "far"):
                keep = [~x for x in far_sets[run][part]]
                dst = RUNS_DIR / f"ablate_learned_{part}_{run}"
                dst.mkdir(exist_ok=True)
                (dst / "config.json").write_text((RUNS_DIR / run / "config.json").read_text())
                bits = np.concatenate([np.packbits(k.cpu().numpy().ravel()) for k in keep]).tobytes()
                (dst / f"mask_step{step}.zst").write_bytes(zstandard.ZstdCompressor(level=19).compress(bits))
                print("wrote", dst.name, "deleted", sum(n(~k) for k in keep), flush=True)

    out["ablation"] = ablation_scores()
    dump_json(out, CODE_DIR.parent / "results" / "mask_anatomy.json")


if __name__ == "__main__":
    main()
