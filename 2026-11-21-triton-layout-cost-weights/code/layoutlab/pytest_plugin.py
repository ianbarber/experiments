"""pytest plugin: record every kernel a test suite compiles.

    LAYOUTLAB_WORKLOAD=tk LAYOUTLAB_OUT=dir [LAYOUTLAB_ARCH=90] [LAYOUTLAB_LAUNCH=1] \
        python -m pytest -p layoutlab.pytest_plugin <tests>
"""

import os


def pytest_configure(config):
    from . import instrument
    from .tutorials import _stub_profiler
    _stub_profiler()
    arch = os.environ.get("LAYOUTLAB_ARCH")
    instrument.install(
        os.environ["LAYOUTLAB_WORKLOAD"],
        os.environ["LAYOUTLAB_OUT"],
        arch=int(arch) if arch else None,
        launch=os.environ.get("LAYOUTLAB_LAUNCH") == "1",
    )


def pytest_unconfigure(config):
    from . import instrument
    replay = os.environ.get("LAYOUTLAB_REPLAY")
    if replay:
        instrument.replay([int(a) for a in replay.split(",")])


def pytest_collection_modifyitems(config, items):
    """LAYOUTLAB_SAMPLE="test_op=12" keeps a fixed one-in-twelve sample of that test's cases."""
    import zlib

    spec = os.environ.get("LAYOUTLAB_SAMPLE")
    if not spec:
        return
    rates = {}
    for part in spec.split(","):
        name, rate = part.split("=")
        rates[name] = int(rate)
    keep = []
    for item in items:
        rate = rates.get(getattr(item, "originalname", None) or item.name.split("[")[0])
        if rate is None or zlib.crc32(item.nodeid.encode()) % rate == 0:
            keep.append(item)
    items[:] = keep
