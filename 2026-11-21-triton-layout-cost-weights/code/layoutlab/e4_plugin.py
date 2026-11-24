"""pytest plugin: run the E4 procedure on every conflicted kernel a test suite launches.

    LAYOUTLAB_E4_OUT=dir [LAYOUTLAB_ARCH=90] [LAYOUTLAB_SAMPLE=test_op=12] [LAYOUTLAB_E4_PER_KERNEL=12] \
        [LAYOUTLAB_E4_REPRICE="shared_memory=0:3.84,warp_shuffle=0:3.84"] \
        python -m pytest -p layoutlab.e4_plugin <tests>

On the device's own target the tests run for real and distinct-code alternatives are
timed. For another target nothing is launched and only the code is compared. With
LAYOUTLAB_E4_REPRICE, every distinct kernel is also rebuilt with that conversion price and,
where the code changes, checked and timed against the default; those records go to
`<out>_reprice/results.jsonl`. LAYOUTLAB_E4_ONLY=<regex on study labels> repeats only the
matching studies, for a second run of the ones that showed a difference.
LAYOUTLAB_E4_MAX_TESTS=<n> ends the session after n tests.
"""

import json
import os
import re
from pathlib import Path

_S = {}


def pytest_configure(config):
    import torch

    from . import instrument, variants
    from .tutorials import _stub_profiler

    _stub_profiler()
    out = Path(os.environ["LAYOUTLAB_E4_OUT"])
    out.mkdir(parents=True, exist_ok=True)
    cap = torch.cuda.get_device_capability()
    native = cap[0] * 10 + cap[1]
    arch = int(os.environ.get("LAYOUTLAB_ARCH") or native)
    launchable = arch == native
    if not launchable:
        instrument.install("e4_suite", str(out / "census"), arch=arch, launch=False, log_events=False)
    reprice = os.environ.get("LAYOUTLAB_E4_REPRICE")
    reprice_out = None
    if reprice:
        reprice_out = out.parent / (out.name.replace(f"_sm{arch}", "") + f"_reprice_sm{arch}")
        reprice_out.mkdir(parents=True, exist_ok=True)
    _S.update(reprice=reprice, reprice_out=reprice_out, reprice_per_kernel={}, reprice_seen=set(), rn=0,
              only=os.environ.get("LAYOUTLAB_E4_ONLY"))
    _S.update(out=out, arch=arch, launchable=launchable, seen=set(), per_kernel={}, n=0,
              limit=int(os.environ.get("LAYOUTLAB_E4_PER_KERNEL", "12")),
              capture=variants.Capture(out / "capture_events.jsonl"))


def pytest_collection_modifyitems(config, items):
    from .pytest_plugin import pytest_collection_modifyitems as sample
    sample(config, items)


def pytest_runtest_teardown(item):
    import pytest

    from . import variants

    _S["tests"] = _S.get("tests", 0) + 1
    limit = os.environ.get("LAYOUTLAB_E4_MAX_TESTS")
    if limit and _S["tests"] > int(limit):
        pytest.exit(f"stopping after {limit} tests", returncode=0)
    capture = _S["capture"]
    launches, capture.launches = capture.launches, []
    for la in launches:
        if la.args is None:
            continue
        d = la.describe()
        key = json.dumps(d, sort_keys=True, default=str)
        if _S["reprice"] and key not in _S["reprice_seen"]:
            _S["reprice_seen"].add(key)
            if _S["reprice_per_kernel"].get(la.name, 0) < _S["limit"]:
                _S["reprice_per_kernel"][la.name] = _S["reprice_per_kernel"].get(la.name, 0) + 1
                _S["rn"] += 1
                label = f'{la.name}_r{_S["rn"]:03d}'
                if _S["only"] and not re.search(_S["only"], label):
                    continue
                try:
                    rec = variants.study(la, label, _S["reprice_out"], launchable=_S["launchable"],
                                         n_runs=int(os.environ.get("LAYOUTLAB_E4_RUNS", "30")),
                                         env_variants={"repriced": {"TRITON_LAYOUT_REPRICE": _S["reprice"]}})
                except BaseException as e:  # noqa: BLE001
                    if isinstance(e, KeyboardInterrupt):
                        raise
                    rec = {"label": label, "status": "error", "error": f"{type(e).__name__}: {str(e)[:400]}", **d}
                rec["target"] = _S["arch"]
                rec["test"] = item.nodeid
                for c in rec.get("conflicts", []):
                    c.pop("candidates", None)
                with open(_S["reprice_out"] / "results.jsonl", "a") as fh:
                    fh.write(json.dumps(rec, default=str) + "\n")
                if rec.get("variants", {}).get("repriced", {}).get("status") not in (None, "identical_code"):
                    print("\n" + variants.summarize(rec), flush=True)
        if not la.n_conflicts:
            continue
        if key in _S["seen"]:
            continue
        _S["seen"].add(key)
        if _S["per_kernel"].get(la.name, 0) >= _S["limit"]:
            continue
        _S["per_kernel"][la.name] = _S["per_kernel"].get(la.name, 0) + 1
        _S["n"] += 1
        label = f'{la.name}_{_S["n"]:03d}'
        if _S["only"] and not re.search(_S["only"], label):
            continue
        try:
            rec = variants.study(la, label, _S["out"], launchable=_S["launchable"],
                                 n_runs=int(os.environ.get("LAYOUTLAB_E4_RUNS", "30")))
        except BaseException as e:  # noqa: BLE001
            if isinstance(e, KeyboardInterrupt):
                raise
            rec = {"label": label, "status": "error", "error": f"{type(e).__name__}: {str(e)[:400]}", **d}
        rec["target"] = _S["arch"]
        rec["test"] = item.nodeid
        for c in rec.get("conflicts", []):
            c.pop("candidates", None)
        with open(_S["out"] / "results.jsonl", "a") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")
        print("\n" + variants.summarize(rec), flush=True)
