"""Fit and validate every steering vector from saved activations (CPU only).

  python scripts/02_build_vectors.py --model 7b

Pain (faithful arm): the paper's S2 recipe and layer selection, cross-checked against
their shipped vector, AUC table, and numb/sadness z-scores.
Itch-A / itch-B: identical recipe on data/itch_dataset.json. The layer rule, fixed before any
itch number was seen (lab notes, 18 September): each concept's extraction layer is its own CV-AUC
argmax, as in the paper; cosines are computed at one common layer (the pain extraction layer,
every vector refitted there) and again at the itch layer for comparison.

Writes results/<model>/vectors/vectors.pt, validation.json, layer_curves_{pain,itch}.csv,
unembedding_<vector>.csv.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from itchlib.common import MODELS, PAPER, cos, model_dir, paper_shipped_vector, random_directions, weights_dir
from itchlib import vectors as V

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True, choices=list(MODELS))
args = ap.parse_args()
mdir = model_dir(args.model)
vdir = mdir / "vectors"
vdir.mkdir(exist_ok=True)
name = MODELS[args.model]["name"]

ITCH_T = ["I1", "I2", "I3", "I4", "I5"]
ITCH_A = dict(target=ITCH_T, control=["PP", "NS", "URGE", "ADJ", "D"], neutral="D")
ITCH_B = dict(target=ITCH_T, control=["D2", "NS", "URGE", "ADJ", "D"], neutral="D")

pain_store = torch.load(mdir / "activations_pain.pt", weights_only=False)["sets"]
itch_path = mdir / "activations_itch.pt"
itch_store = torch.load(itch_path, weights_only=False)["sets"] if itch_path.exists() else None
val, vecs = {"model": name}, {}


def best_layer(df):
    return int(df.groupby("layer")["auc_vs_all_controls"].mean().idxmax())


def per_layer(df):
    return {int(k): round(float(v), 4) for k, v in df.groupby("layer")["auc_vs_all_controls"].mean().items()}


# ---------------- pain ----------------
curves = V.cv_layer_curves(pain_store, ["S2_1P", "S2_3P"], V.PAIN)
curves.to_csv(vdir / "layer_curves_pain.csv", index=False)
Lp = best_layer(curves)
shipped, L_ship = paper_shipped_vector(args.model)
pain = V.fit(pain_store["S2_1P"], Lp, V.PAIN)
pain_at_ship = V.fit(pain_store["S2_1P"], L_ship, V.PAIN)
vecs.update(pain=pain, pain_layer=Lp)
ship_auc = pd.read_csv(PAPER / "results" / "3.2_pain_vectors" / "per_model" / name / "auc_summary.csv", index_col=0)
ref = pain_store["S2_1P"]
cats = np.array(ref["categories"])
z = lambda store, ds, vec, L, refe: V.zscore(store[ds], L, vec, refe)
zs2 = z(pain_store, "S2_1P", pain, Lp, ref)
val["pain"] = {
    "extraction_layer": Lp, "paper_extraction_layer": L_ship,
    "cv_auc_at_best": per_layer(curves)[Lp], "cv_auc_at_paper_layer": per_layer(curves)[L_ship],
    "cv_auc_by_layer": per_layer(curves),
    "norm": float(np.linalg.norm(pain)), "paper_norm": float(np.linalg.norm(shipped)),
    "cosine_to_paper_vector_at_paper_layer": cos(pain_at_ship, shipped),
    "auc_in_sample_S2_1P": V.auc_vs_each_control(ref, Lp, pain, V.PAIN),
    "auc_in_sample_S2_3P": V.auc_vs_each_control(pain_store["S2_3P"], Lp, pain, V.PAIN),
    "paper_auc_in_sample_S2_1P": ship_auc.loc["S2_1P"].to_dict(),
    "paper_auc_in_sample_S2_3P": ship_auc.loc["S2_3P"].to_dict(),
    "z_vs_S2_1P": {
        "pain_sentences": float(zs2[np.isin(cats, V.PAIN["target"])].mean()),
        "physical_pain_A1": float(zs2[cats == "A1"].mean()),
        "control_sentences": float(zs2[np.isin(cats, V.PAIN["control"])].mean()),
        "numb": float(z(pain_store, "Numb_1P", pain, Lp, ref).mean()),
        "sadness": float(z(pain_store, "SD_sadness_1P", pain, Lp, ref).mean()),
        "random_neutral": float(z(pain_store, "Random_1P", pain, Lp, ref).mean()),
        "arousal": float(z(pain_store, "Arousal_1P", pain, Lp, ref).mean()),
    },
}

# sadness vector against pooled neutral (recipe of the paper's 3.2/02 script), at the pain layer
neutral = V.pooled_neutral(pain_store, Lp)
vecs["sadness"] = V.control_vector(V.layer_np(pain_store["SD_sadness_1P"], Lp), neutral)
vecs["random"] = {rs: v.numpy() for rs, v in random_directions(len(pain), float(np.linalg.norm(pain))).items()}

# ---------------- itch ----------------
if itch_store is not None:
    cA = V.cv_layer_curves(itch_store, ["ITCH_A_1P"], ITCH_A)
    cB = V.cv_layer_curves(itch_store, ["ITCH_B_1P"], ITCH_B)
    pd.concat([cA.assign(vector="itch_A"), cB.assign(vector="itch_B")]).to_csv(vdir / "layer_curves_itch.csv", index=False)
    for tag, spec, ds, c in [("itch_A", ITCH_A, "ITCH_A_1P", cA), ("itch_B", ITCH_B, "ITCH_B_1P", cB)]:
        Li = best_layer(c)
        e = itch_store[ds]
        v_own, v_pl = V.fit(e, Li, spec), V.fit(e, Lp, spec)
        vecs.update({tag: v_own, tag + "_layer": Li, tag + "_at_pain_layer": v_pl})
        ecats = np.array(e["categories"])

        def describe(vec, L):
            zi = V.zscore(e, L, vec, e)
            zn = V.zscore(itch_store["NotItch_1P"], L, vec, e)
            return {"layer": L, "cv_auc": per_layer(c)[L], "norm": float(np.linalg.norm(vec)),
                    "auc_in_sample": V.auc_vs_each_control(e, L, vec, spec),
                    "z_vs_own_set": {"itch_sentences": float(zi[np.isin(ecats, ITCH_T)].mean()),
                                     "control_sentences": float(zi[np.isin(ecats, spec["control"])].mean()),
                                     "not_itching_heldout": float(zn.mean()),
                                     **{f"control_{k}": float(zi[ecats == k].mean()) for k in spec["control"]}},
                    "not_itching_auc_vs_itch": float(V.P.roc_auc_score(
                        np.r_[np.ones(100), np.zeros(100)], np.r_[zi[np.isin(ecats, ITCH_T)], zn]))}

        val[tag] = {"extraction_layer": Li, "cv_auc_at_best": per_layer(c)[Li], "cv_auc_at_pain_layer": per_layer(c)[Lp],
                    "cv_auc_by_layer": per_layer(c), **describe(v_own, Li)}
        val[tag + "_at_pain_layer"] = describe(v_pl, Lp)
    # cross-projections: does the pain vector see itch sentences, and vice versa?
    eA = itch_store["ITCH_A_1P"]
    ecats = np.array(eA["categories"])
    zp = V.P.project_and_zscore(V.layer_np(eA, Lp), pain, V.layer_np(ref, Lp))
    zi_on_pain = V.P.project_and_zscore(V.layer_np(ref, vecs["itch_A_layer"]), vecs["itch_A"],
                                        V.layer_np(eA, vecs["itch_A_layer"]))
    val["cross"] = {"itch_sentences_on_pain_vector_z": float(zp[np.isin(ecats, ITCH_T)].mean()),
                    "pain_sentences_on_itchA_vector_z": float(zi_on_pain[np.isin(cats, V.PAIN["target"])].mean()),
                    "physical_pain_A1_on_itchA_vector_z": float(zi_on_pain[cats == "A1"].mean())}

# ---------------- cosine matrix at the common (pain) layer ----------------
at_pl = {"pain": pain, "sadness": vecs["sadness"]}
if itch_store is not None:
    at_pl.update(itch_A=vecs["itch_A_at_pain_layer"], itch_B=vecs["itch_B_at_pain_layer"])
names = list(at_pl)
M = pd.DataFrame([[cos(at_pl[a], at_pl[b]) for b in names] for a in names], index=names, columns=names)
rand = np.stack(list(vecs["random"].values()))
for n in names:
    cs = [cos(at_pl[n], r) for r in rand]
    M.loc[n, "random_mean"] = float(np.mean(cs))
    M.loc[n, "random_max_abs"] = float(np.max(np.abs(cs)))
M.to_csv(vdir / "cosine_matrix_pain_layer.csv")
val["cosine_at_pain_layer"] = {"layer": Lp, "matrix": M.round(4).to_dict()}
if itch_store is not None and vecs["itch_A_layer"] != Lp:
    Li = vecs["itch_A_layer"]
    p_i = V.fit(pain_store["S2_1P"], Li, V.PAIN)
    val["cosine_at_itchA_layer"] = {"layer": Li, "pain_itchA": cos(p_i, vecs["itch_A"]),
                                    "pain_itchB": cos(p_i, V.fit(itch_store["ITCH_B_1P"], Li, ITCH_B))}

# ---------------- unembedding (paper 3.3/07: unit vector times W_U, no final norm) ----------------
from safetensors import safe_open
from transformers import AutoTokenizer
wd = weights_dir(args.model)
idx = json.load(open(wd / "model.safetensors.index.json"))["weight_map"]
with safe_open(wd / idx["lm_head.weight"], "pt") as f:
    W = f.get_tensor("lm_head.weight").float()
tok = AutoTokenizer.from_pretrained(wd)
val["unembedding"] = {}
for n in ["pain", "itch_A", "itch_B", "itch_A_at_pain_layer", "itch_B_at_pain_layer", "sadness"]:
    if n not in vecs:
        continue
    v = torch.tensor(vecs[n], dtype=torch.float32)
    scores = W @ (v / v.norm())
    top, bot = torch.topk(scores, 60), torch.topk(-scores, 60)
    rows = [{"rank": i + 1, "side": side, "token": tok.decode([int(j)]), "score": float(s) * sign}
            for side, tk, sign in (("top", top, 1), ("bottom", bot, -1))
            for i, (s, j) in enumerate(zip(tk.values, tk.indices))]
    pd.DataFrame(rows).to_csv(vdir / f"unembedding_{n}.csv", index=False)
    val["unembedding"][n] = {"top30": [r["token"] for r in rows if r["side"] == "top"][:30],
                             "bottom30": [r["token"] for r in rows if r["side"] == "bottom"][:30]}

torch.save(vecs, vdir / "vectors.pt")
json.dump(val, open(vdir / "validation.json", "w"), indent=1, ensure_ascii=False)

# ---------------- console summary ----------------
p = val["pain"]
print(f"\n=== {name} ===")
print(f"PAIN  layer {p['extraction_layer']} (paper {p['paper_extraction_layer']}), CV AUC {p['cv_auc_at_best']} "
      f"(at paper layer {p['cv_auc_at_paper_layer']}), norm {p['norm']:.2f} (paper {p['paper_norm']:.2f}), "
      f"cos to paper vector {p['cosine_to_paper_vector_at_paper_layer']:.4f}")
print("  in-sample AUC S2_1P ours :", {k: round(v, 4) for k, v in p["auc_in_sample_S2_1P"].items()})
print("  in-sample AUC S2_1P paper:", p["paper_auc_in_sample_S2_1P"])
print("  z:", {k: round(v, 2) for k, v in p["z_vs_S2_1P"].items()})
for tag in ("itch_A", "itch_B"):
    if tag in val:
        t = val[tag]
        print(f"{tag} layer {t['extraction_layer']}, CV AUC {t['cv_auc_at_best']} (at pain layer {t['cv_auc_at_pain_layer']}), "
              f"norm {t['norm']:.2f}, in-sample AUC {({k: round(v, 3) for k, v in t['auc_in_sample'].items()})}")
        print("  z:", {k: round(v, 2) for k, v in t["z_vs_own_set"].items()}, "| AUC itch vs not-itching:", round(t["not_itching_auc_vs_itch"], 3))
        t2 = val[tag + "_at_pain_layer"]
        print(f"  same recipe at pain layer L{t2['layer']}: norm {t2['norm']:.2f}, z:", {k: round(v, 2) for k, v in t2["z_vs_own_set"].items()},
              "| AUC itch vs not-itching:", round(t2["not_itching_auc_vs_itch"], 3))
if "cross" in val:
    print("cross:", {k: round(v, 2) for k, v in val["cross"].items()})
print(f"\ncosine matrix at L{Lp}:\n{M.round(3).to_string()}")
if "cosine_at_itchA_layer" in val:
    print("cosines at itch-A layer:", val["cosine_at_itchA_layer"])
for n, u in val["unembedding"].items():
    print(f"\n{n} top: {u['top30']}\n{n} bottom: {u['bottom30']}")
