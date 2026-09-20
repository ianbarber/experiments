"""Concept vectors, fitted with the paper's own functions.

`compute_pain_vector`, `compute_auc`, `compute_layer_curves_kfold` and `project_and_zscore`
are imported verbatim from the paper's scripts/3.2_pain_vectors/01_*.py. They read the target
and control category lists from module globals, so `categories()` swaps those globals to fit
a vector for any concept with exactly the same recipe.
"""
from contextlib import contextmanager

import numpy as np

from .common import paper_module

P = paper_module("scripts/3.2_pain_vectors/01_extract_activations_and_pain_vectors.py", "paper_extract")

PAIN = dict(target=["A1", "A2", "A3", "A4", "A5"], control=["B", "C1", "C2", "D", "E"], neutral="D")


@contextmanager
def categories(spec):
    old = (P.PAIN_CATEGORIES, P.CONTROL_CATEGORIES, P.NEUTRAL_CATEGORY)
    P.PAIN_CATEGORIES, P.CONTROL_CATEGORIES, P.NEUTRAL_CATEGORY = spec["target"], spec["control"], spec["neutral"]
    try:
        yield
    finally:
        P.PAIN_CATEGORIES, P.CONTROL_CATEGORIES, P.NEUTRAL_CATEGORY = old


def layer_np(entry, layer):
    return entry["acts"][layer].float().numpy()


def fit(entry, layer, spec):
    """Raw (unnormalised) denoised difference-of-means vector at `layer`."""
    with categories(spec):
        return P.compute_pain_vector(layer_np(entry, layer), entry["categories"])


def auc(entry, layer, vec, spec):
    with categories(spec):
        return P.compute_auc(layer_np(entry, layer), entry["categories"], vec)


def auc_vs_each_control(entry, layer, vec, spec):
    out = {"ALL": auc(entry, layer, vec, spec)}
    for c in spec["control"]:
        out[c] = auc(entry, layer, vec, dict(spec, control=[c]))
    return out


def cv_layer_curves(store, set_names, spec):
    """The paper's 5-fold CV-by-sentence-set AUC per layer. Their function hardcodes the keys
    S2_1P / S2_3P, so the sets are passed under those names (second one optional)."""
    n_layers = next(iter(store.values()))["acts"].shape[0]
    keys = ["S2_1P", "S2_3P"]
    acts = {"final_token": {}}
    meta = {}
    for key, name in zip(keys, set_names):
        e = store[name]
        acts["final_token"][key] = [e["acts"][L].float() for L in range(n_layers)]
        meta[key] = {"categories": e["categories"], "sets": e["sets"]}
    with categories(spec):
        df = P.compute_layer_curves_kfold(acts, meta, "final_token", list(range(n_layers)))
    df["dataset"] = df["dataset"].map(dict(zip(keys, set_names)))
    return df


def zscore(entry, layer, vec, ref_entry):
    return P.project_and_zscore(layer_np(entry, layer), vec, layer_np(ref_entry, layer))


# --- control directions, recipe of the paper's scripts/3.2_pain_vectors/02_build_control_vectors.py ---
# (that script has import-time side effects, so its three small functions are restated here)
DENOISE_VARIANCE = 0.5
POOL_SETS = ["S1_1P", "S2_1P", "ControlSupplement_1P"]


def pooled_neutral(store, layer):
    rows = []
    for ds in POOL_SETS:
        m = np.array(store[ds]["categories"]) == "D"
        rows.append(layer_np(store[ds], layer)[m])
    return np.concatenate(rows)


def denoise_basis(neutral_acts, neutral_mean):
    X = np.nan_to_num(neutral_acts - neutral_mean)
    _, S, Vt = np.linalg.svd(X, full_matrices=False)
    cumvar = np.cumsum(S ** 2) / (S ** 2).sum()
    return Vt[:min(int(np.searchsorted(cumvar, DENOISE_VARIANCE)) + 1, len(Vt))]


def control_vector(acts, neutral):
    """mean(acts) - mean(pooled neutral), top neutral PCs (50% variance) projected out."""
    nm = neutral.mean(0)
    v = acts.mean(0) - nm
    for d in denoise_basis(neutral, nm):
        v = v - np.dot(v, d) * d
    return v
