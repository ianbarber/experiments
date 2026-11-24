"""Print the grid gate: dense newline probability at and around the true newline position."""
import sys
import numpy as np
cond = sys.argv[1] if len(sys.argv) > 1 else "dense"
for C in (8, 16, 32, 64, 128, 256):
    d = np.load(f"../data/grid-{C}.npz"); r = np.load(f"../raw/stage1/grid-{C}__{cond}.npz")
    p = r["p_newline"]                       # p[n, t] = P(token t+1 = newline)
    T = d["ids"].shape[1]
    true = np.arange(C, T, C + 1)            # positions of newline tokens
    sel = true[3:]
    pn = p[:, sel - 1]
    print(f"C={C:3d} rows={len(true):3d} P(nl|true) mean={pn.mean():.3f} median={np.median(pn):.3f} by row:",
          np.round(p[:, true[:6] - 1].mean(0), 2), "offsets -3..3:",
          np.round([p[:, sel[sel + o - 1 < T - 1] + o - 1].mean() for o in (-3, -2, -1, 0, 1, 2, 3)], 3))
