#!/usr/bin/env python3
"""Failure modes of the E vs A8 episodes, measured against the files the tests need
(2026-09-14 audit).

"Needed" files = the intersection of the file sets of every patch that resolved the
instance in any run under <runs_dir> (all arms, all rounds). Recall is reported against
that set and against the gold patch's files. Each failing episode is then labelled:

  localization miss   a needed file was never edited AND the failing tests differ from
                      those of a fully-localized failing episode of the same instance
                      (otherwise the miss was not what decided the outcome)
  otherwise           the instance-level label from ATTRIBUTION below, assigned by reading
                      the task statement against the failing assertion; the evidence quote
                      is stored next to the label so the judgement can be checked.

Usage: anchor_failure_modes.py <promax.json> <runs_dir> [--arms e,a8] [--rounds r1,r2] [-v]
"""
import argparse, glob, json, os, re
from collections import Counter, defaultdict

ATTRIBUTION = {
    # instance: (label, evidence from the task statement / failing assertion)
    "django__django-19643": ("unmet stated requirement",
        "statement quotes the duplicate-partial message and the 'Invalid block tag' text verbatim; tests assert exactly those"),
    "huggingface__lerobot-2808": ("unmet stated requirement",
        "statement: 'if the buffered frame is older than this threshold, it should raise a TimeoutError'; test: DID NOT RAISE TimeoutError"),
    "optuna__optuna-6166": ("unmet stated requirement",
        "statement names `inverse_squared_lengthscales` and the `length_scales` property; tests: AttributeError on that name"),
    "pandas-dev__pandas-61244": ("unmet stated requirement",
        "statement: first and last scatter tick positions must agree with the line plot; test_scatter_line_xticks fails"),
    "albumentations-team__albumentations-2337": ("unmet stated requirement",
        "statement specifies VahadaneNormalizer/MacenkoNormalizer with fit() estimating the stain matrix; tests check the separation angle on a synthetic image and batch/single consistency"),
    "verl-project__verl-3915": ("unmet stated requirement",
        "statement specifies mask-mode denominator semantics; standalone test: 'First seq ratio should be ≈0.37'"),
    "albumentations-team__albumentations-2495": ("requirement not derivable from the statement",
        "tests import `filter_valid_metadata(data)` and expect a UserWarning; statement says only 'rejecting non-dictionary entries ... with appropriate warnings'"),
    "google__adk-python-c_19315fe": ("requirement not derivable from the statement",
        "the test's Mock lacks `_invocation_context`, never mentioned in the statement; only the gold's way of reaching the header provider passes"),
    "langchain-ai__langchain-32996": ("requirement not derivable from the statement",
        "test expects a lone ToolMessage when the only tool call gets a custom response; statement says 'an updated AIMessage ... together with ToolMessage instances'"),
    "mikf__gallery-dl-7872": ("external knowledge",
        "a new extractor must parse leakgallery.com's live JSON; the field layout is in neither the repo nor the statement"),
    "stanfordnlp__dspy-9047": ("localization miss",
        "fix needs a toDict() call at the json.dump site in evaluate.py; every failing episode edited only example.py"),
    "huggingface__transformers-38332": ("localization miss",
        "modular_t5gemma.py must be kept in sync with modeling_t5gemma.py"),
}


def files(mp): return set(re.findall(r"^diff --git a/(\S+) b/", mp, re.M))


def load(runs_dir, run):
    preds = json.load(open(f"{runs_dir}/{run}/preds.json")); pr = json.load(open(f"{runs_dir}/{run}/pass_rate.json")); out = {}
    for x in pr:
        i = x["instance_id"]
        mp = preds.get(i, {}).get("model_patch", "") if isinstance(preds, dict) else next((p["model_patch"] for p in preds if p["instance_id"] == i), "")
        out[i] = dict(files=files(mp), golden=x["golden"]["final_result"] == "success", ok=x["model"]["final_result"] == "success",
                      stdout=x["model"].get("stdout", ""), stderr=x["model"].get("stderr", ""))
    return out


_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def failing_tests(st):
    so = _ANSI.sub("", st["stdout"])
    t = re.findall(r"^(?:FAILED|ERROR) (\S+)", so, re.M)                    # pytest short summary
    t += re.findall(r"^(?:FAIL|ERROR): (\S+ \(\S+\))", st["stderr"], re.M)  # django/unittest on stderr
    t += re.findall(r"^✗ Test failed with error: (.*)$", st["stdout"], re.M)  # verl standalone runner
    return frozenset(t)


def error_types(st):
    c = Counter(re.findall(r"^E\s+(\w+(?:Error|Exception))\b", st["stdout"], re.M))
    c.update(re.findall(r"^(\w+(?:Error|Exception)):", st["stderr"], re.M))
    return dict(c.most_common(2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("promax"); ap.add_argument("runs"); ap.add_argument("--arms", default="e,a8"); ap.add_argument("--rounds", default="r1,r2")
    ap.add_argument("-v", action="store_true")
    a = ap.parse_args()
    data = {x["instance_id"]: x for x in json.load(open(a.promax))}
    runs = {os.path.basename(os.path.dirname(p)): load(a.runs, os.path.basename(os.path.dirname(p))) for p in glob.glob(f"{a.runs}/*/pass_rate.json")}
    needed = {}
    for ri in runs.values():
        for i, st in ri.items():
            if st["ok"]: needed[i] = st["files"] if i not in needed else needed[i] & st["files"]
    arms = a.arms.split(","); rounds = a.rounds.split(",")
    main_runs = [f"s1-arm-{arm}-{r}" for arm in arms for r in rounds]
    gv = [i for i in runs[main_runs[0]] if all(runs[r].get(i, {}).get("golden") for r in main_runs) and i in needed]
    print(f"golden-valid instances with a known needed set: {len(gv)}; needed-set sizes: median {sorted(len(needed[i]) for i in gv)[len(gv)//2]}, "
          f"gold-file counts: median {sorted(len(files(data[i]['patch'])) for i in gv)[len(gv)//2]}")
    # failing tests of fully-localized failing episodes, per instance (for the 'binding' rule)
    full_fail_tests = defaultdict(set)
    for run in main_runs:
        for i in gv:
            st = runs[run][i]
            if not st["ok"] and not (needed[i] - st["files"]): full_fail_tests[i].add(failing_tests(st))
    print(f"\n{'':44s}" + "".join(f"{arm.upper():>8s}" for arm in arms))
    table = {}
    rows = []
    for arm in arms:
        rec = []; grec = []; miss_eps = 0; fails = Counter()
        for r in rounds:
            run = f"s1-arm-{arm}-{r}"
            for i in gv:
                st = runs[run][i]; n = needed[i]; g = files(data[i]["patch"])
                rec.append(len(st["files"] & n) / len(n)); grec.append(len(st["files"] & g) / len(g))
                missed = n - st["files"]
                if missed: miss_eps += 1
                if st["ok"]: continue
                ft = failing_tests(st)
                if missed and (ft not in full_fail_tests[i]): label = "localization miss"
                else: label = ATTRIBUTION.get(i, ("unclassified", ""))[0]
                fails[label] += 1
                rows.append((run, i, label, sorted(missed), sorted(t.split("::")[-1][:40] for t in ft)[:3], error_types(st)))
        table[arm] = dict(recall_needed=sum(rec) / len(rec), recall_gold=sum(grec) / len(grec), episodes=len(rec), miss_eps=miss_eps, fails=fails)
    for key, label in [("recall_needed", "mean recall vs test-needed files"), ("recall_gold", "mean recall vs gold-patch files")]:
        print(f"{label:44s}" + "".join(f"{table[arm][key]:8.3f}" for arm in arms))
    print(f"{'episodes with a needed file never edited':44s}" + "".join(f"{table[arm]['miss_eps']:5d}/{table[arm]['episodes']:<3d}" for arm in arms))
    print(f"{'failing episodes':44s}" + "".join(f"{sum(table[arm]['fails'].values()):8d}" for arm in arms))
    for label in ["localization miss", "unmet stated requirement", "requirement not derivable from the statement", "external knowledge", "unclassified"]:
        if any(table[arm]["fails"][label] for arm in arms):
            print(f"  {label:42s}" + "".join(f"{table[arm]['fails'][label]:8d}" for arm in arms))
    print("\nper failing episode (label; needed files never edited; failing tests; error types):")
    for run, i, label, missed, ft, et in rows:
        print(f"  [{run}] {i:40s} {label:44s} missed={missed if missed else '-'}")
        if a.v: print(f"      tests={ft} errors={et}")
    print("\nattribution evidence (task statement vs failing assertion):")
    for i, (label, ev) in ATTRIBUTION.items(): print(f"  {i:40s} {label:44s} {ev}")


if __name__ == "__main__":
    main()
