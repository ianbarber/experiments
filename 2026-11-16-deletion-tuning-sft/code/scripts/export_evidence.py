"""Pack the small files the analysis reads into ../results/evidence.tar.gz.

Per run: config.json, the checkpoint records from log.jsonl (not the per-step losses), every
eval_step*.json and games_step*.json, per-item suite scores, and the per-answer IFEval flags
written by scripts/ifeval_behaviour.py. No weights, no raw logs, no generated text.
To re-run the analysis without a GPU:  mkdir -p runs && tar -xzf ../results/evidence.tar.gz -C runs
"""
import io
import json
import tarfile
from pathlib import Path

CODE = Path(__file__).resolve().parent.parent
RUNS, OUT = CODE / "runs", CODE.parent / "results" / "evidence.tar.gz"


def add(tar, name, data: bytes):
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mtime = 0
    tar.addfile(info, io.BytesIO(data))


def main():
    n = 0
    with tarfile.open(OUT, "w:gz", compresslevel=9) as tar:
        # runs/session2/base is the base suite scored again in the follow-up session
        for d in sorted(RUNS.iterdir()) + sorted((RUNS / "session2").glob("*")):
            if not d.is_dir() or not list(d.glob("eval_step*.json")):
                continue
            name = str(d.relative_to(RUNS))
            if (d / "config.json").exists():
                add(tar, f"{name}/config.json", (d / "config.json").read_bytes())
            if (d / "log.jsonl").exists():  # the mask ablations are scored, not trained
                saves = [l for l in (d / "log.jsonl").read_text().splitlines() if l.startswith('{"save"')]
                add(tar, f"{name}/log.jsonl", "".join(l + "\n" for l in saves).encode())
                add(tar, f"{name}/done.json", (d / "done.json").read_bytes())
            for p in sorted(d.glob("eval_step*.json")) + sorted(d.glob("games_step*.json")):
                add(tar, f"{name}/{p.name}", json.dumps(json.loads(p.read_text()), separators=(",", ":")).encode())
            for pat in ("items.json.gz", "summary.json", "ifeval_flags.json"):
                for p in sorted(d.glob(f"suite_step*/{pat}")):
                    add(tar, f"{name}/{p.parent.name}/{p.name}", p.read_bytes())
            n += 1
    print(f"{n} runs, {OUT.stat().st_size / 2**20:.2f} MiB -> {OUT}")


if __name__ == "__main__":
    main()
