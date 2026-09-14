"""Near-duplicate and contamination filter (docs/02 gate item 6), v0: hashed word 13-grams + instruction Jaccard on
word 3-grams against every task already in the pool (all iterations). Held-out similarity is checked against
precomputed hashed n-grams only (the held-out text never enters the loop host's pool)."""
from __future__ import annotations
import hashlib, json, re
from pathlib import Path
from rsi.db import DB

def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_./-]+", text.lower())

def shingles(text: str, n: int = 3) -> set[int]:
    w = _words(text)
    return {hash(" ".join(w[i:i + n])) for i in range(max(len(w) - n + 1, 0))}

def hashed_13grams(text: str) -> set[str]:
    w = _words(text)
    return {hashlib.sha1(" ".join(w[i:i + 13]).encode()).hexdigest()[:16] for i in range(max(len(w) - 12, 0))}

def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a or b) else 0.0

def dedup_against_pool(db: DB, candidates: list[Path], *, threshold: float = 0.6, heldout_sig: Path | None = None) -> tuple[list[Path], list[Path]]:
    pool = [(r["task_id"], Path(r["path"])) for r in db.q("SELECT task_id, path FROM tasks WHERE status LIKE 'gated%' OR status='seed'")]
    pool_sh = {}
    for tid, p in pool:
        f = p / "instruction.md"
        if f.exists(): pool_sh[tid] = shingles(f.read_text())
    held = set()
    sigp = heldout_sig or (Path(db.path).parent / "heldout-13grams.json")
    if sigp.exists():
        held = set(json.loads(sigp.read_text()))
    kept, dups = [], []
    for td in candidates:
        text = (td / "instruction.md").read_text()
        sh = shingles(text)
        best = max((jaccard(sh, s), tid) for tid, s in pool_sh.items()) if pool_sh else (0.0, None)
        # also against other candidates already kept in this batch
        for kd in kept:
            j = jaccard(sh, shingles((kd / "instruction.md").read_text()))
            if j > best[0]: best = (j, kd.name)
        held_hit = len(hashed_13grams(text) & held) if held else 0
        if best[0] >= threshold or held_hit >= 3:
            dups.append(td); db.gate(td.name, "dedup", "fail", {"best_jaccard": round(best[0], 3), "against": best[1], "heldout_13gram_hits": held_hit})
        else:
            kept.append(td); db.gate(td.name, "dedup", "pass", {"best_jaccard": round(best[0], 3), "against": best[1], "heldout_13gram_hits": held_hit})
            pool_sh[td.name] = sh
    return kept, dups

def write_heldout_signature(heldout_dir: Path, out: Path) -> int:
    grams = set()
    for p in Path(heldout_dir).glob("*/instruction.md"):
        grams |= hashed_13grams(p.read_text())
    out.write_text(json.dumps(sorted(grams))); return len(grams)
