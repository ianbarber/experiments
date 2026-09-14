"""Memorisation probe (docs/02 "Memorisation probe"): does the base model reproduce published Terminal-Bench
artefacts better than equal-length paraphrases of them, and better than it reproduces private tasks it cannot
have seen?

Three stages, each resumable and each writing to `<run>/probe/`:
  paraphrase  a non-Qwen CLI model rewrites every instruction.md and solve.sh (no GPU, no model under test)
  measure     the base model, chat template bypassed, T=0, logprobs on: guided prefix completion at 32/64/128
              tokens, ROUGE-L of the continuation, 5-gram overlap, exact-20-token rate, Min-K% (k=20)
  report      one-sided Mann-Whitney of the Terminal-Bench gap distribution against the private-task control,
              plus the flag-count test against the control 95th percentile

The per-task primary statistic is the original-minus-paraphrase gap in guided-prompt ROUGE-L; the private dev
and held-out tasks are the fresh-task control (read-only on the base model, so this is not exposure).
"""
from __future__ import annotations
import json, math, random, re, statistics, subprocess, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import httpx
from rsi import config as C
from rsi.db import DB

PREFIX_TOKENS = (32, 64, 128)
CONTINUATION_TOKENS = 200
EXACT_TOKENS = 20
MIN_K = 20
CONTROL_SAMPLE = 180          # private tasks paraphrased for the null distribution (deterministic sample)
PARAPHRASE_CLI = "kimi"       # Moonshot; never a Qwen model, per docs/02

PARA_SYSTEM = ("You rewrite text for a contamination control in a machine-learning experiment. Preserve the meaning "
               "exactly and keep the length within 10% of the original. Change wording, sentence order and phrasing "
               "throughout; keep file paths, numbers and identifiers that the text depends on. Output only the "
               "rewritten text, with no preamble, no code fence and no commentary.")

# ------------------------------------------------------------------ artefacts
def collect(db: DB, *, control_sample: int = CONTROL_SAMPLE, seed: int = 0) -> list[dict]:
    """Terminal-Bench tasks (the probe's subject) plus a deterministic sample of private tasks (the control)."""
    def rows(*statuses):
        qs = ",".join("?" * len(statuses))
        return [dict(r) for r in db.q(f"SELECT task_id, name, path, status FROM tasks WHERE status IN ({qs}) ORDER BY task_id", *statuses)]
    tb = rows("tbpub", "tbpub-rejected:oracle")
    priv = rows("dev", "heldout", "heldout-supp")
    random.Random(seed).shuffle(priv)
    out = []
    for r, kind in [(r, "tb") for r in tb] + [(r, "control") for r in priv[:control_sample]]:
        d = Path(r["path"])
        ins = (d / "instruction.md")
        sol = next((p for p in (d / "solution" / "solve.sh", d / "solution.sh") if p.exists()), None)
        if not ins.exists(): continue
        out.append({"task_id": r["task_id"], "kind": kind, "path": str(d),
                    "instruction": ins.read_text(errors="replace"),
                    "solve": sol.read_text(errors="replace") if sol else ""})
    return out

# ------------------------------------------------------------------ paraphrase
def _cli_rewrite(text: str, *, cli: str = PARAPHRASE_CLI, timeout_s: int = 600) -> str:
    prompt = PARA_SYSTEM + "\n\n--- TEXT TO REWRITE ---\n" + text
    cmd = {"kimi": ["kimi", "-p", prompt, "--output-format", "text"],
           "agy": ["agy", "--print", prompt, "--output-format", "text", "--dangerously-skip-permissions"]}[cli]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s, cwd="/tmp")
    out = p.stdout.strip()
    if not out: raise RuntimeError(f"{cli} exit {p.returncode}: {p.stderr.strip()[:200]}")
    return _unwrap(re.sub(r"^```[a-z]*\n|\n```$", "", out).strip())

def _unwrap(text: str) -> str:
    """kimi's text output wraps a reply in a bullet with a two-space hanging indent; undo that so the paraphrase
    differs from the original in wording only, not in layout."""
    lines = text.splitlines()
    if lines and re.match(r"^\s*[\u2022*-]\s", lines[0]):
        lines[0] = re.sub(r"^\s*[\u2022*-]\s+", "", lines[0])
        if sum(1 for l in lines[1:] if l.startswith("  ") or not l.strip()) >= max(1, len(lines) - 1):
            lines = [lines[0]] + [l[2:] if l.startswith("  ") else l for l in lines[1:]]
    return "\n".join(lines).strip()

def paraphrase_all(records: list[dict], out_path: Path, *, cli: str = PARAPHRASE_CLI, workers: int = 8, log=print) -> dict:
    """Rewrite every instruction and solve.sh; resumable through the output file."""
    store = json.loads(out_path.read_text()) if out_path.exists() else {}
    jobs = [(r["task_id"], field, r[field]) for r in records for field in ("instruction", "solve")
            if r[field].strip() and not store.get(r["task_id"], {}).get(field)]
    log(f"paraphrase: {len(jobs)} artefacts to rewrite ({len(store)} tasks already have some)")
    done = [0]
    def one(job):
        tid, field, text = job
        try: return tid, field, _cli_rewrite(text[:12000], cli=cli)
        except Exception as ex: return tid, field, f"__ERROR__ {type(ex).__name__}: {ex}"
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for tid, field, text in ex.map(one, jobs):
            store.setdefault(tid, {})[field] = text
            done[0] += 1
            if done[0] % 20 == 0:
                out_path.write_text(json.dumps(store, indent=1)); log(f"paraphrase: {done[0]}/{len(jobs)}")
    out_path.write_text(json.dumps(store, indent=1))
    errs = sum(1 for v in store.values() for t in v.values() if t.startswith("__ERROR__"))
    log(f"paraphrase: {len(store)} tasks stored, {errs} errors")
    return {"tasks": len(store), "artefacts": sum(len(v) for v in store.values()), "errors": errs}

# ------------------------------------------------------------------ text statistics
def _toks(s: str) -> list[str]:
    return re.findall(r"\w+|[^\w\s]", s)

def rouge_l(cand: str, ref: str) -> float:
    """F-measure over the longest common subsequence of word tokens."""
    a, b = _toks(cand), _toks(ref)
    if not a or not b: return 0.0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(cur[j], prev[j + 1]))
        prev = cur
    l = prev[-1]
    if not l: return 0.0
    p, r = l / len(a), l / len(b)
    return 2 * p * r / (p + r)

def ngram_overlap(cand: str, ref: str, n: int = 5) -> float:
    A, B = _toks(cand), _toks(ref)
    if len(A) < n or len(B) < n: return 0.0
    ca = Counter(tuple(A[i:i + n]) for i in range(len(A) - n + 1))
    cb = Counter(tuple(B[i:i + n]) for i in range(len(B) - n + 1))
    return sum((ca & cb).values()) / max(1, sum(ca.values()))

# ------------------------------------------------------------------ model calls
class Completions:
    """Raw completion endpoint on the served base model: no chat template, greedy, logprobs on."""
    def __init__(self, base_url: str = C.VLLM_BASE_URL, model: str = C.SERVED_MODEL_NAME, timeout_s: float = 300):
        base = base_url.rstrip("/")
        self.url = base + "/completions"
        self.root = base[: -len("/v1")] if base.endswith("/v1") else base   # /tokenize sits outside /v1
        self.model = model
        self.client = httpx.Client(timeout=timeout_s)
    def tokenize(self, text: str) -> list[int]:
        return self._post(self.root + "/tokenize", {"model": self.model, "prompt": text}).json()["tokens"]
    def detokenize(self, tokens: list[int]) -> str:
        return self._post(self.root + "/detokenize", {"model": self.model, "tokens": tokens}).json()["prompt"]
    def _post(self, url: str, body: dict, attempts: int = 3):
        """Retry transient failures: the server may be restarting, and a 5xx on one request should not lose a record."""
        last = None
        for i in range(attempts):
            try:
                r = self.client.post(url, json=body)
                if r.status_code < 500:
                    r.raise_for_status(); return r
                last = httpx.HTTPStatusError(f"{r.status_code}: {r.text[:200]}", request=r.request, response=r)
            except (httpx.ConnectError, httpx.ReadTimeout) as ex:
                last = ex
            time.sleep(2 * (i + 1))
        raise last

    def complete(self, prompt: str, *, max_tokens: int) -> str:
        r = self._post(self.url, {"model": self.model, "prompt": prompt, "max_tokens": max_tokens,
                                  "temperature": 0.0, "seed": 0})
        return r.json()["choices"][0]["text"]
    def prompt_logprobs(self, text: str) -> list[float]:
        r = self._post(self.url, {"model": self.model, "prompt": text, "max_tokens": 1, "temperature": 0.0,
                                  "prompt_logprobs": 0, "echo": False})
        pls = r.json()["choices"][0].get("prompt_logprobs") or []
        out = []
        for entry in pls:
            if not entry: continue
            out.append(max(float(v["logprob"]) for v in entry.values()) if isinstance(entry, dict) else float(entry))
        return out

def min_k_percent(logprobs: list[float], k: int = MIN_K) -> float:
    if not logprobs: return float("nan")
    n = max(1, int(len(logprobs) * k / 100))
    return sum(sorted(logprobs)[:n]) / n

GUIDE = ("The following is the exact text of the file {file} from the {origin} task suite, task `{task}`.\n"
         "Reproduce it verbatim.\n\n{prefix}")

def measure_one(api: Completions, *, task: str, origin: str, file: str, text: str) -> dict:
    """Guided prefix completion at each prefix length plus Min-K% over the whole artefact."""
    toks = api.tokenize(text)
    try:                                   # Min-K% is a descriptive secondary; losing it must not lose the record
        mk = min_k_percent(api.prompt_logprobs(text[:12000]))
    except Exception as ex:
        mk = float("nan")
    out = {"n_tokens": len(toks), "min_k_percent": mk}
    for np_ in PREFIX_TOKENS:
        if len(toks) < np_ + EXACT_TOKENS: continue
        prefix, rest = api.detokenize(toks[:np_]), api.detokenize(toks[np_:])
        gen = api.complete(GUIDE.format(file=file, origin=origin, task=task, prefix=prefix), max_tokens=CONTINUATION_TOKENS)
        exact_ref = api.detokenize(toks[np_:np_ + EXACT_TOKENS])
        gen_head = api.detokenize(api.tokenize(gen)[:EXACT_TOKENS])
        out[f"p{np_}"] = {"rouge_l": rouge_l(gen, rest[:4000]), "ngram5": ngram_overlap(gen, rest[:4000]),
                          "exact20": float(gen_head.strip() == exact_ref.strip())}
    return out

def measure_all(records: list[dict], paraphrases: dict, out_path: Path, *, workers: int = 8, log=print) -> dict:
    """The GPU pass: every artefact in both conditions, concurrent against one server. Resumable through the file."""
    import threading
    api = Completions()
    store = json.loads(out_path.read_text()) if out_path.exists() else {}
    lock = threading.Lock(); t0 = time.time(); done = [0]
    def one(r: dict) -> None:
        tid = r["task_id"]; para = paraphrases.get(tid, {})
        origin = "Terminal-Bench 2.0" if r["kind"] == "tb" else "a private held-out"
        cur = dict(store.get(tid) or {"kind": r["kind"]})
        for field, fname in (("instruction", "instruction.md"), ("solve", "solution/solve.sh")):
            text = r[field]; ptext = para.get(field, "")
            if not text.strip() or not ptext or ptext.startswith("__ERROR__"): continue
            for cond, body in (("original", text), ("paraphrase", ptext)):
                if f"{field}:{cond}" in cur: continue
                try:
                    cur[f"{field}:{cond}"] = measure_one(api, task=tid, origin=origin, file=fname, text=body)
                except Exception as ex:
                    cur[f"{field}:{cond}"] = {"error": f"{type(ex).__name__}: {ex}"}
        with lock:
            store[tid] = cur; done[0] += 1
            if done[0] % 20 == 0:
                out_path.write_text(json.dumps(store, indent=1)); log(f"measure: {done[0]}/{len(records)} ({time.time() - t0:.0f}s)")
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(one, records))
    out_path.write_text(json.dumps(store, indent=1))
    errs = sum(1 for m in store.values() for k, v in m.items() if isinstance(v, dict) and "error" in v)
    log(f"measure: {len(store)} tasks, {errs} failed measurements, {time.time() - t0:.0f}s")
    return {"tasks": len(store), "errors": errs}

# ------------------------------------------------------------------ statistics
def mann_whitney_u(x: list[float], y: list[float]) -> dict:
    """One-sided (x > y) Mann-Whitney U with a tie-corrected normal approximation."""
    n1, n2 = len(x), len(y)
    if n1 < 3 or n2 < 3: return {"status": "pending", "n1": n1, "n2": n2}
    pooled = sorted([(v, 0) for v in x] + [(v, 1) for v in y])
    ranks = [0.0] * len(pooled); i = 0; ties = []
    while i < len(pooled):
        j = i
        while j + 1 < len(pooled) and pooled[j + 1][0] == pooled[i][0]: j += 1
        r = (i + j) / 2 + 1
        for k in range(i, j + 1): ranks[k] = r
        ties.append(j - i + 1); i = j + 1
    r1 = sum(rk for rk, (_, g) in zip(ranks, pooled) if g == 0)
    u1 = r1 - n1 * (n1 + 1) / 2
    n = n1 + n2
    mu = n1 * n2 / 2
    tie_term = sum(t ** 3 - t for t in ties)
    sd = math.sqrt(n1 * n2 / 12 * ((n + 1) - tie_term / (n * (n - 1))))
    z = (u1 - mu - 0.5) / sd if sd > 0 else 0.0
    p = 0.5 * math.erfc(z / math.sqrt(2))
    return {"u": u1, "z": z, "p_one_sided": p, "n1": n1, "n2": n2, "reject_at_05": p <= 0.05}

def binom_tail(n: int, k: int, p: float) -> float:
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))

def report(measurements: dict, *, field: str = "instruction", prefix: int = 64) -> dict:
    """Primary statistic: original minus paraphrase guided ROUGE-L, Terminal-Bench against the private control."""
    gaps = {"tb": {}, "control": {}}
    for tid, m in measurements.items():
        o, p = m.get(f"{field}:original", {}), m.get(f"{field}:paraphrase", {})
        if f"p{prefix}" in o and f"p{prefix}" in p:
            gaps[m["kind"]][tid] = o[f"p{prefix}"]["rouge_l"] - p[f"p{prefix}"]["rouge_l"]
    tb, ctl = list(gaps["tb"].values()), list(gaps["control"].values())
    res = {"field": field, "prefix_tokens": prefix, "n_tb": len(tb), "n_control": len(ctl),
           "mean_gap_tb": statistics.fmean(tb) if tb else None, "mean_gap_control": statistics.fmean(ctl) if ctl else None,
           "mann_whitney_tb_gt_control": mann_whitney_u(tb, ctl)}
    if len(ctl) >= 20 and tb:
        thr = sorted(ctl)[int(0.95 * len(ctl)) - 1]
        flagged = [t for t, g in gaps["tb"].items() if g > thr]
        res["flag_test"] = {"control_p95": thr, "flagged": len(flagged), "n": len(tb),
                            "p_binomial": binom_tail(len(tb), len(flagged), 0.05),
                            "reject_at_05": binom_tail(len(tb), len(flagged), 0.05) <= 0.05,
                            "flagged_tasks": sorted(flagged)}
    rej = res["mann_whitney_tb_gt_control"].get("reject_at_05") or res.get("flag_test", {}).get("reject_at_05")
    res["verdict"] = ("contaminated comparability: exclude flagged tasks from the headline Terminal-Bench numbers" if rej
                      else "no detected verbatim memorisation (power ~0.5-0.6 for a 15%-vs-5% flagged rate at n=89; a null does not prove cleanliness)")
    # descriptive secondaries
    for f2 in ("instruction", "solve"):
        for stat in ("ngram5", "exact20"):
            vals = {k: [] for k in ("tb", "control")}
            for m in measurements.values():
                o = m.get(f"{f2}:original", {}).get(f"p{prefix}")
                if o: vals[m["kind"]].append(o[stat])
            res.setdefault("descriptive", {})[f"{f2}.{stat}"] = {k: (round(statistics.fmean(v), 4) if v else None) for k, v in vals.items()}
    mk = {"tb": [], "control": []}
    for m in measurements.values():
        v = m.get(f"{field}:original", {}).get("min_k_percent")
        if v is not None and not math.isnan(v): mk[m["kind"]].append(v)
    res.setdefault("descriptive", {})["min_k_percent"] = {k: (round(statistics.fmean(v), 4) if v else None) for k, v in mk.items()}
    return res

# ------------------------------------------------------------------ driver
def probe_dir(run: C.RunPaths) -> Path:
    d = run.root / "probe"; d.mkdir(parents=True, exist_ok=True); return d

def run_stage(run_name: str, stage: str, *, log=print, workers: int = 8, cli: str = PARAPHRASE_CLI) -> dict:
    run = C.run_paths(run_name); db = DB(run.db); d = probe_dir(run)
    recs_path, para_path, meas_path = d / "records.json", d / "paraphrases.json", d / "measurements.json"
    if stage in ("collect", "paraphrase", "measure", "all") and not recs_path.exists():
        recs = collect(db); recs_path.write_text(json.dumps(recs, indent=1)); log(f"collect: {len(recs)} tasks")
    recs = json.loads(recs_path.read_text()) if recs_path.exists() else []
    out: dict = {"tasks": len(recs)}
    if stage in ("paraphrase", "all"):
        out["paraphrase"] = paraphrase_all(recs, para_path, cli=cli, workers=workers, log=log)
        if stage == "paraphrase": return out
    if stage in ("measure", "all"):
        out["measure"] = measure_all(recs, json.loads(para_path.read_text()), meas_path, workers=workers, log=log)
        if stage == "measure": return out
    if stage in ("report", "all"):
        res = report(json.loads(meas_path.read_text()))
        (d / "report.json").write_text(json.dumps(res, indent=1))
        out["report"] = res
    return out
