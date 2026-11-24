#!/usr/bin/env python
"""Pack the final TTGIR saved by device E4 runs, with debug locations removed.

    python pack_ttgir.py --runs $RUNS/e4 --dirs tk_sm120 inductor_sm120 --out ../results/e4_ttgir_sm120.tar.gz

Each E4 study saves the default build's TTGIR and that of every alternative whose text
differs. Debug locations carry absolute paths of the machine that compiled them and are not
needed to compare programs, so they are stripped. `census_from_ttgir.py` gives the same
digests for the stripped files as for the originals, and they still parse.
"""

from __future__ import annotations

import argparse
import io
import tarfile
from pathlib import Path

from census_from_ttgir import strip_locations


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    n = 0
    with tarfile.open(args.out, "w:gz") as tar:
        for d in args.dirs:
            for f in sorted((Path(args.runs) / d / "ttgir").glob("*.ttgir")):
                data = strip_locations(f.read_text()).encode()
                info = tarfile.TarInfo(f"{d}/ttgir/{f.name}")
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
                n += 1
    print(f"{args.out}: {n} files, {Path(args.out).stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
