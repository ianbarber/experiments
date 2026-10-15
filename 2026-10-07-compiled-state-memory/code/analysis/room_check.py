"""Room check readouts and decision (LABNOTES 2026-10-13, night; rule fixed before any question ran,
amended before any question ran: see the "code review" note under it).

Per size: accuracy of every arm (N and O once, under size "O"), by type; calls, tokens read and ending
flags. B* = the stronger of T and D at that size, chosen on Qwen's labels (ties: T) and recorded in the
second judge's key. At the thesis sizes (mid, large), with X = M (or O for the stage-2 skip rule):
  G = acc_M − acc_B*, q = max(disc(X, B*), disc(A, B*)) + flip rate, D_min = sqrt(7.84 q / (235 + 7.84))
  NO-GO(s): G's 95% upper bound < D_min;  GO(s): lower bound > 0 and G ≥ D_min;  else INCONCLUSIVE(s)
  overall: GO as soon as either size is GO; NO-GO if both are NO-GO; else INCONCLUSIVE; f* on GO.
  Skip rule (no M@s): NO-GO(s) if UB(O − B*) + max(0, acc_M(small) − acc_O) < D_min.
A size is decided only when every calibration question has X, B* and A records, the flip reruns exist,
and the second judge has labelled every answered X and B* response at that size (labels then come from
it; no answer is always wrong). Bootstrap: paired over questions, 2,000 resamples, default_rng(42).

uv run --with numpy python analysis/room_check.py --traj ../results/phase2/room/traj.jsonl \
    --judge ../results/phase2/room/judge.jsonl [--export_j2 ../results/phase2/room/j2/items.json] \
    [--judge2 labels.json --j2_key ../results/phase2/room/j2/key.json]
"""
import argparse
import collections
import hashlib
import json
import math
import os
import random

import numpy as np

N_TEST, SIZES, THESIS = 235, ("small", "mid", "large"), ("mid", "large")
TYPES = ("single-session-user", "single-session-assistant", "multi-session", "temporal-reasoning", "knowledge-update")
PREP = "../results/phase2/room/prep"


def akey(qid, answer):
    return hashlib.sha1(f"{qid}\x00{answer}".encode()).hexdigest()[:16]


def d_min(q):
    return math.sqrt(7.84 * q / (N_TEST + 7.84))


def boot(diff, n=2000):
    rng = np.random.default_rng(42)
    d = np.asarray(diff, float)
    m = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    return d.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True, help="comma-separated record files")
    ap.add_argument("--judge", required=True)
    ap.add_argument("--judge2", help="second judge labels {id: true/false} (JSON booleans only)")
    ap.add_argument("--j2_key", help="written by --export_j2")
    ap.add_argument("--export_j2", help="write the second judge's items here (blind, shuffled); the key goes beside it")
    args = ap.parse_args()
    cal = json.load(open(os.path.join(PREP, "split.json")))["cal"]
    J = {r["key"]: r["label"] for r in map(json.loads, open(args.judge))}
    recs = [r for p in args.traj.split(",") for r in map(json.loads, open(p))]
    main_ = {(r["qid"], r["arm"], r["size"]): r for r in recs if not r.get("tag")}
    flip = {(r["qid"], r["arm"], r["size"]): r for r in recs if r.get("tag") == "flip"}
    assert {k[0] for k in main_} <= set(cal), "records outside the calibration set"
    qtype = {k[0]: r["qtype"] for k, r in main_.items()}

    def qlabel(r):
        return False if r["answer"] is None else J[akey(r["qid"], r["answer"])]

    LQ = {k: qlabel(r) for k, r in main_.items()}
    j2, key = {}, None
    if args.judge2:
        key = json.load(open(args.j2_key))
        raw = json.load(open(args.judge2))
        bad = [i for i, v in raw.items() if not isinstance(v, bool) or i not in key["items"]]
        assert not bad, f"second judge: {len(bad)} labels are not JSON booleans or not in the key, e.g. {bad[:3]}"
        j2 = {tuple(key["items"][i]): v for i, v in raw.items()}
        print(f"second judge: {len(j2)} labels for {len(key['items'])} exported responses")
    L = {k: j2.get(k, v) for k, v in LQ.items()}

    def vec(lab, arm, size):
        if not all((q, arm, size) in lab for q in cal):
            return None
        return np.array([lab[(q, arm, size)] for q in cal], float)

    cells = sorted({(k[1], k[2]) for k in main_})
    print(f"questions {len(cal)}; records {len(main_)} (+{len(flip)} flip reruns)")
    print("\n== accuracy (final labels) ==")
    print(f"{'cell':12s} {'acc':>5s}  " + "  ".join(f"{t[:6]:>6s}" for t in TYPES)
          + "  calls  read_tok  forced cut  none  malf  drop  ovfl  rcut")
    for arm, size in cells:
        v = vec(L, arm, size)
        if v is None:
            print(f"{arm}@{size}: incomplete ({sum((q, arm, size) in L for q in cal)}/{len(cal)})")
            continue
        rs = [main_[(q, arm, size)] for q in cal]
        bt = [np.mean([L[(q, arm, size)] for q in cal if qtype[q] == t]) for t in TYPES]
        fl = collections.Counter()
        for r in rs:
            f = r["flags"]
            fl["forced"] += f.get("forced") is not None
            fl["cut"] += bool(f.get("cut"))
            fl["none"] += r["answer"] is None
            fl["malf"] += bool(f.get("malformed"))
            fl["drop"] += f.get("dropped", 0) > 0
            fl["ovfl"] += bool(f.get("overflow"))
            fl["rcut"] += f.get("result_cut", 0) > 0
        calls = np.mean([len(r["calls"]) for r in rs])
        read = np.mean([sum(c["result_tokens"] for c in r["calls"]) for r in rs])
        print(f"{arm + '@' + size:12s} {v.mean():5.3f}  " + "  ".join(f"{b:6.2f}" for b in bt)
              + f"  {calls:5.2f}  {read:8.0f}  " + "  ".join(f"{fl[k]:4d}" for k in ("forced", "cut", "none", "malf", "drop", "ovfl", "rcut")))

    fl_n = sum(1 for k in flip if k in main_)
    fl_d = sum(qlabel(r) != LQ[k] for k, r in flip.items() if k in main_)
    flip_rate = fl_d / fl_n if fl_n else None
    print("\nflip rate (Qwen labels, rerun vs main): " + (f"{fl_d}/{fl_n} = {flip_rate:.3f}" if fl_n else "no reruns yet"))

    if j2:
        print("\n== judge agreement (second judge vs Qwen) by cell ==")
        agree = collections.defaultdict(lambda: [0, 0])
        for k, v in j2.items():
            agree[(k[1], k[2])][0] += v == LQ[k]
            agree[(k[1], k[2])][1] += 1
        for c, (a, n) in sorted(agree.items()):
            print(f"  {c[0]}@{c[1]}: {a}/{n} = {a / n:.3f}")

    bstar = {}
    for s in SIZES:
        T, D = vec(LQ, "T", s), vec(LQ, "D", s)
        if T is not None and D is not None:
            bstar[s] = "T" if T.mean() >= D.mean() else "D"
    if key is not None:
        assert key["bstar"] == {s: bstar[s] for s in key["bstar"]}, f"B* changed since export: {key['bstar']} vs {bstar}"

    O, Ms = vec(L, "O", "O"), vec(L, "M", "small")  # noqa: E741 (arm names)
    verdict, export, xarm = {}, [], {}
    for s in SIZES:
        if s not in bstar:
            print(f"\n{s}: T or D incomplete")
            continue
        b = bstar[s]
        B, A = vec(L, b, s), vec(L, "A", s)
        print(f"\n== {s}: B* = {b} ({B.mean():.3f}; T {vec(L, 'T', s).mean():.3f}, D {vec(L, 'D', s).mean():.3f}) ==")
        if O is not None:
            h, lo, hi = boot(O - B)
            print(f"  room for any hint O - B*: {h:+.3f} [{lo:+.3f}, {hi:+.3f}]   (O {O.mean():.3f})")
        if A is not None:
            print(f"  A (no hint) {A.mean():.3f};  B* - A {B.mean() - A.mean():+.3f}")
        M = vec(L, "M", s)
        if s not in THESIS:
            if M is not None:
                g, lo, hi = boot(M - B)
                print(f"  M {M.mean():.3f}; M - B* {g:+.3f} [{lo:+.3f}, {hi:+.3f}] (descriptive)")
            continue
        X, xa, xs = (M, "M", s) if M is not None else (O, "O", "O")
        xarm[s] = xa
        if X is None or A is None:
            print("  not enough arms yet")
            continue
        export += [(q, xa, xs) for q in cal] + [(q, b, s) for q in cal]
        missing = []
        if flip_rate is None:
            missing.append("flip reruns")
        unlabelled = [k for k in [(q, xa, xs) for q in cal] + [(q, b, s) for q in cal]
                      if main_[k]["answer"] is not None and k not in j2]
        if unlabelled:
            missing.append(f"second judge ({len(unlabelled)} responses)")
        q = max(np.mean(X != B), np.mean(A != B)) + (flip_rate or 0.0)
        dm = d_min(q)
        if xa == "M":
            g, lo, hi = boot(M - B)
            call = "NO-GO" if hi < dm else "GO" if (lo > 0 and g >= dm) else "INCONCLUSIVE"
            print(f"  G = M - B*: {g:+.3f} [{lo:+.3f}, {hi:+.3f}]  q {q:.3f}  D_min {dm:.3f}  -> {call}({s})"
                  + (f"   [provisional: missing {', '.join(missing)}]" if missing else ""))
            if call == "GO":
                print(f"  f* = (B* - A + D_min) / (M - A) = {(B.mean() - A.mean() + dm) / (M.mean() - A.mean()):.2f}")
        else:
            h, lo, hi = boot(O - B)
            slack = None if Ms is None else max(0.0, Ms.mean() - O.mean())
            ok = slack is not None and hi + slack < dm
            call = "NO-GO" if ok else None
            print(f"  skip rule: UB(O - B*) {hi:+.3f} + slack {slack if slack is not None else 'n/a (M@small not run)'}"
                  f" < D_min {dm:.3f} (q {q:.3f}): {ok}  -> " + (f"NO-GO({s}) without M@{s}" if ok else f"run M@{s}")
                  + (f"   [provisional: missing {', '.join(missing)}]" if missing else ""))
        if call and not missing:
            verdict[s] = call
    v = list(verdict.values())
    if "GO" in v:
        overall = "GO"
    elif len(v) == len(THESIS):
        overall = "NO-GO" if all(x == "NO-GO" for x in v) else "INCONCLUSIVE"
    else:
        overall = "not decided yet"
    print(f"\n== overall: {overall} ({verdict}) ==")
    if args.export_j2:
        kpath = os.path.join(os.path.dirname(args.export_j2), "key.json")
        assert os.path.abspath(kpath) != os.path.abspath(args.export_j2)
        os.makedirs(os.path.dirname(args.export_j2), exist_ok=True)
        export = sorted({k for k in export if main_[k]["answer"] is not None})
        random.Random(43).shuffle(export)
        items, ids = [], {}
        for k in export:
            q = json.load(open(os.path.join(PREP, f"{k[0]}.json")))
            iid = hashlib.sha1(repr(k).encode()).hexdigest()[:10]
            ids[iid] = list(k)
            items.append(dict(id=iid, qtype=q["qtype"], question=q["question"], gold=q["gold"], response=main_[k]["answer"]))
        json.dump(items, open(args.export_j2, "w"), indent=1, ensure_ascii=False)
        json.dump(dict(items=ids, bstar={s: bstar[s] for s in THESIS if s in bstar}, x=xarm), open(kpath, "w"))
        print(f"second judge: {len(items)} responses -> {args.export_j2} (key {kpath})")


if __name__ == "__main__":
    main()
