"""PROPOSE: turn the generator brief + exemplars + themes into candidate task directories (docs/03 §4 step 1).

The author can be the policy (vLLM chat endpoint) or an external CLI (kimi / grok / agy) for held-out authoring.
Output parsing: the FILE-section format defined in improver/generator_prompt.md.
"""
from __future__ import annotations
import hashlib, json, random, re, subprocess, time
from pathlib import Path
import yaml

SECTION = re.compile(r"###\s*(META|INSTRUCTION|SETUP|SOLUTION|TESTS)\s*\n+```[a-z]*\n(.*?)```", re.DOTALL | re.IGNORECASE)
DOMAINS = {"data_processing", "data_querying", "data_science", "debugging", "dependency_management", "file_operations",
           "scientific_computing", "security", "software_engineering"}

def load_improver(h_dir: Path) -> dict:
    return {"generator_prompt": (h_dir / "generator_prompt.md").read_text(), "mutation_ops": (h_dir / "mutation_ops.md").read_text(),
            "exemplar_policy": json.loads((h_dir / "exemplar_policy.json").read_text()), "playbook": (h_dir / "failure_playbook.md").read_text(),
            "themes": json.loads((h_dir / "themes.json").read_text()), "thresholds": json.loads((h_dir / "thresholds.json").read_text())}

def render_request(h: dict, *, mode: str, theme: str, domain: str, tier: str, exemplars: list[dict], mutation_op: str | None = None,
                   heldout: bool = False) -> str:
    """The user turn that accompanies the generator brief (system)."""
    parts = [f"Write ONE new {tier} task in the domain `{domain}` on the theme: {theme}."]
    if mode == "mutation" and exemplars and mutation_op:
        parts.append(f"Apply the mutation operator `{mutation_op}` to this task (keep what works, change what the operator says):\n\n"
                     + exemplars[0]["instruction"][:3000])
    elif exemplars:
        parts.append("Here are examples of tasks at about the right difficulty (do not copy them; write something new):\n\n" +
                     "\n\n---\n\n".join(e["instruction"][:1500] for e in exemplars))
    playbook_rows = [l for l in h["playbook"].splitlines() if l.startswith("|")][2:]
    if playbook_rows and not heldout:
        parts.append("Recent failure analysis (write tasks that exercise these weaknesses):\n\n" + h["playbook"][-3000:])
    if heldout:
        parts.append("This task is for a held-out evaluation set: make it self-contained, unambiguous, and different from typical tutorial examples.")
    parts.append("Follow the output format exactly: META, INSTRUCTION, SETUP, SOLUTION, TESTS, each with one fenced block.")
    return "\n\n".join(parts)

HEADING = re.compile(r"^\s*(?:[•*-]\s*)?#{2,4}\s*(META|INSTRUCTION|SETUP|SOLUTION|TESTS)\b.*$", re.IGNORECASE | re.MULTILINE)

def _normalise(text: str) -> str:
    """Undo CLI pretty-printing: leading bullet on the first line, uniform indentation, CRLF."""
    text = text.replace("\r\n", "\n")
    lines = text.split("\n")
    if lines and lines[0].lstrip().startswith("•"):
        lines[0] = lines[0].lstrip()[1:].lstrip()
    import textwrap
    body = "\n".join(lines)
    # dedent only if every non-empty line shares the indent (kimi indents everything by two spaces)
    non_empty = [l for l in lines[1:] if l.strip()]
    if non_empty and all(l.startswith("  ") for l in non_empty):
        body = lines[0] + "\n" + textwrap.dedent("\n".join(lines[1:]))
    return body

def _sections(text: str) -> dict[str, str]:
    """Split on the section headings; inside each section keep what lies between the first fence line and the last."""
    heads = list(HEADING.finditer(text))
    out = {}
    for i, m in enumerate(heads):
        name = m.group(1).upper()
        chunk = text[m.end(): heads[i + 1].start() if i + 1 < len(heads) else len(text)]
        lines = chunk.split("\n")
        fence_idx = [j for j, l in enumerate(lines) if l.strip().startswith("```")]
        if len(fence_idx) >= 2:
            body = "\n".join(lines[fence_idx[0] + 1: fence_idx[-1]])
        elif len(fence_idx) == 1:
            body = "\n".join(lines[fence_idx[0] + 1:])
        else:
            body = chunk
        out[name] = body.strip("\n")
    return out

def parse_task(text: str) -> dict | None:
    secs = _sections(_normalise(text or ""))
    if not all(k in secs and secs[k].strip() for k in ("META", "INSTRUCTION", "SETUP", "SOLUTION", "TESTS")):
        return None
    try:
        meta = yaml.safe_load(secs["META"]) or {}
    except yaml.YAMLError:
        return None
    if not isinstance(meta, dict) or not meta.get("name"): return None
    meta["domain"] = meta.get("domain") if meta.get("domain") in DOMAINS else "software_engineering"
    meta["tier"] = meta.get("tier") if meta.get("tier") in ("easy", "medium", "hard") else "medium"
    meta["needs_root"] = bool(meta.get("needs_root", False))
    meta["name"] = re.sub(r"[^a-z0-9-]+", "-", str(meta["name"]).lower()).strip("-")[:60] or "task"
    return {"meta": meta, "instruction": secs["INSTRUCTION"].strip() + "\n", "setup": secs["SETUP"].strip() + "\n",
            "solution": secs["SOLUTION"].strip() + "\n", "tests": secs["TESTS"].strip() + "\n"}

STANDARD_TEST_SH_IMPORT = "rsi.tasks.seeds"

def write_task(spec: dict, pool: Path, *, source: str, provenance: dict, prefix: str = "gen") -> tuple[str, Path]:
    from rsi.tasks.seeds import STANDARD_TEST_SH, write_task_toml
    h = hashlib.sha256((spec["instruction"] + spec["tests"] + spec["setup"]).encode()).hexdigest()[:8]
    tid = f"{prefix}__{spec['meta']['name']}__{h}"
    dest = pool / tid
    if dest.exists():
        return tid, dest
    (dest / "environment").mkdir(parents=True); (dest / "solution").mkdir(); (dest / "tests").mkdir()
    (dest / "instruction.md").write_text(spec["instruction"])
    (dest / "environment" / "setup.sh").write_text(spec["setup"])
    (dest / "environment" / "Dockerfile").write_text("FROM rsi-base:latest\nCOPY setup.sh /tmp/rsi-setup.sh\n"
                                                      "RUN cd /app && bash /tmp/rsi-setup.sh && rm -f /tmp/rsi-setup.sh && chown -R agent:agent /app\n")
    sol = spec["solution"] if spec["solution"].lstrip().startswith("#!") else "#!/bin/bash\nset -e\ncd /app\n" + spec["solution"]
    (dest / "solution" / "solve.sh").write_text(sol); (dest / "solution" / "solve.sh").chmod(0o755)
    (dest / "tests" / "test_outputs.py").write_text(spec["tests"])
    (dest / "tests" / "test.sh").write_text(STANDARD_TEST_SH); (dest / "tests" / "test.sh").chmod(0o755)
    meta = dict(spec["meta"]); meta.update({"source": source, "description": meta.get("summary", ""), "tags": [meta["domain"], meta["tier"]]})
    write_task_toml(dest, f"rsi/{tid}", meta)
    if meta.get("needs_root"):
        from rsi.tasks.gate import set_agent_user; set_agent_user(dest, "root")
    prov = dict(provenance); prov.update({"content_hash": h, "network_mode": "no-network", "needs_root": bool(meta.get("needs_root"))})
    (dest / "provenance.json").write_text(json.dumps(prov, indent=1))
    return tid, dest

# ------------------------------------------------------------------ authors
class PolicyAuthor:
    """The trainee itself via vLLM (the self-loop's proposer)."""
    def __init__(self, base_url: str, model: str, temperature: float = 1.0, max_tokens: int = 8192, thinking: bool = True):
        from rsi.harness.vllm_client import VLLMChat
        self.llm = VLLMChat(base_url, model); self.temperature = temperature; self.max_tokens = max_tokens; self.thinking = thinking
        self.name = f"policy:{model}"
    def __call__(self, system: str, user: str, seed: int | None = None) -> str:
        r = self.llm.complete([{"role": "system", "content": system}, {"role": "user", "content": user}], max_tokens=self.max_tokens,
                              temperature=self.temperature, top_p=1.0, enable_thinking=self.thinking, seed=seed)
        from rsi.harness.prompt import strip_thinking
        return strip_thinking(r.content)

class CliAuthor:
    """kimi / grok / agy headless CLIs (held-out authoring, docs/02)."""
    CMDS = {
        "kimi": lambda prompt: ["kimi", "-p", prompt, "--output-format", "text"],
        "grok": lambda prompt: ["grok", "-p", prompt, "--output-format", "plain", "--always-approve", "--disable-web-search"],
        "agy":  lambda prompt: ["agy", "--print", prompt, "--output-format", "text", "--dangerously-skip-permissions"],
    }
    def __init__(self, cli: str, timeout_s: int = 600, workdir: Path | None = None):
        self.cli = cli; self.timeout_s = timeout_s; self.name = f"cli:{cli}"
        self.workdir = workdir or Path("/tmp/rsi-author"); self.workdir.mkdir(parents=True, exist_ok=True)
    def __call__(self, system: str, user: str, seed: int | None = None) -> str:
        prompt = system + "\n\n---\n\n" + user + "\n\nReply with the task only (no tool use, no file writes)."
        p = subprocess.run(self.CMDS[self.cli](prompt), capture_output=True, text=True, timeout=self.timeout_s, cwd=str(self.workdir))
        if not p.stdout.strip():
            raise RuntimeError(f"{self.cli} exit {p.returncode}: {p.stderr.strip()[:300]}")
        return p.stdout

def propose_batch(h_dir: Path, pool: Path, author, *, n: int, exemplar_pool: list[dict], mode_mix: dict | None = None,
                  seed: int = 0, source: str = "policy", iteration: int = 0, heldout: bool = False, log=print, workers: int | None = None,
                  tiers: list[str] | None = None, extra_brief: str = "") -> list[dict]:
    """Generate n candidates in parallel; returns a list of {task_id, path, parsed, request, ...} (parsed=False rows are logged rejects)."""
    from concurrent.futures import ThreadPoolExecutor
    h = load_improver(h_dir); rng = random.Random(seed)
    themes = h["themes"]["themes"]; dom_w = h["themes"]["domain_targets"]
    domains = list(dom_w); weights = [dom_w[d] for d in domains]
    tiers = list(tiers) if tiers else ["easy"] * 4 + ["medium"] * 5 + ["hard"] * 1
    mix = mode_mix or h["exemplar_policy"]["mix"]
    ops = [l.split(":")[0].split(". ")[-1].strip() for l in h["mutation_ops"].splitlines() if re.match(r"^\d+\.", l)]
    # plan every request up front (deterministic in seed), then run them concurrently
    plans = []
    for i in range(n):
        mode = rng.choices(list(mix), weights=list(mix.values()))[0] if exemplar_pool else "fresh"
        if heldout: mode = "fresh"
        theme = rng.choice(themes); domain = rng.choices(domains, weights=weights)[0]; tier = rng.choice(tiers)
        ex = rng.sample(exemplar_pool, min(h["exemplar_policy"]["n_exemplars"], len(exemplar_pool))) if exemplar_pool else []
        op = rng.choice(ops) if mode == "mutation" and ops else None
        user = render_request(h, mode=mode, theme=theme, domain=domain, tier=tier, exemplars=ex, mutation_op=op, heldout=heldout)
        if extra_brief: user += "\n\n" + extra_brief
        plans.append({"i": i, "mode": mode, "theme": theme, "domain": domain, "tier": tier, "op": op, "ex": ex, "user": user})
    if workers is None:
        workers = 2 if author.name.startswith("cli:") else 16

    def one(pl):
        t0 = time.time()
        try:
            text = author(h["generator_prompt"], pl["user"], seed=seed * 100000 + pl["i"])
        except Exception as ex_:
            return {"i": pl["i"], "parsed": False, "error": f"{type(ex_).__name__}: {ex_}", "mode": pl["mode"]}
        return {"i": pl["i"], "text": text, "latency_s": round(time.time() - t0, 1), **{k: pl[k] for k in ("mode", "theme", "domain", "tier", "op", "ex")}}

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool_ex:
        results = list(pool_ex.map(one, plans))
    out = []
    for r in sorted(results, key=lambda x: x["i"]):
        i = r["i"]
        if "error" in r:
            out.append(r); log(f"[{i}] author error: {r['error'][:200]}"); continue
        text = r.pop("text"); ex = r.pop("ex")
        spec = parse_task(text)
        row = {**r, "author": author.name, "chars": len(text), "parsed": spec is not None}
        if spec is None:
            row["raw_head"] = text[:400]; out.append(row); log(f"[{i}] unparseable ({len(text)} chars)"); continue
        prov = {"generator": author.name, "iteration": iteration, "mode": r["mode"], "mutation_op": r["op"], "theme": r["theme"],
                "requested": {"domain": r["domain"], "tier": r["tier"]}, "exemplars": [e.get("task_id") for e in ex],
                "prompt_hash": hashlib.sha256((h["generator_prompt"] + plans[i]["user"]).encode()).hexdigest()[:12],
                "heldout": heldout, "release_eligible": not author.name.startswith("cli:"),
                "licence": "Apache-2.0" if not author.name.startswith("cli:") else "generated-by-external-model", "raw_response_chars": len(text)}
        tid, dest = write_task(spec, pool, source=source, provenance=prov, prefix="heldout" if heldout else f"gen{iteration}")
        (dest / "generation.txt").write_text(text)
        row.update({"task_id": tid, "path": str(dest), "meta": spec["meta"]}); out.append(row)
        log(f"[{i}] {tid} ({spec['meta']['domain']}/{spec['meta']['tier']}, {row['latency_s']}s)")
    return out

# ------------------------------------------------------------------ repair round
REPAIR_REQUEST = """Your task below FAILED its own verification: the reference solution was run in a clean container and then the tests.
Fix the task so that the solution passes every test (edit the solution, the tests, the setup or the instruction as needed, but keep the task's intent).
Reply with the COMPLETE task again in the same five-section format.

Verifier output (pytest, tail):
{pytest}

Reference solution output (tail):
{oracle}

The task as written:
{task}"""

def feedback_for(task_dir: Path, job_dirs: list[Path]) -> dict:
    """Collect the oracle/pytest tails for a task from its gate trials."""
    from rsi.rollout import task_key
    py, orc = "", ""
    for j in job_dirs:
        if not j.exists(): continue
        for t in j.iterdir():
            if not t.is_dir() or not (t / "result.json").exists(): continue
            if task_key(json.loads((t / "result.json").read_text())) != task_dir.name: continue
            p = t / "verifier" / "pytest.log"; o = t / "agent" / "oracle.txt"
            if p.exists(): py = p.read_text(errors="ignore")[-2500:]
            if o.exists(): orc = o.read_text(errors="ignore")[-1500:]
    return {"pytest": py or "(no pytest output; the image or the solution may have failed before the tests ran)", "oracle": orc or "(no output)"}

def repair_task(author, task_dir: Path, feedback: dict, pool: Path, *, iteration: int, source: str, log=print) -> tuple[str, Path] | None:
    gen = (task_dir / "generation.txt")
    original = gen.read_text() if gen.exists() else ""
    if not original: return None
    from rsi.improver.propose import load_improver
    h_prompt = (Path(__file__).resolve().parents[2] / "improver" / "generator_prompt.md").read_text()
    user = REPAIR_REQUEST.format(pytest=feedback["pytest"], oracle=feedback["oracle"], task=original[:12000])
    try:
        text = author(h_prompt, user, seed=None)
    except Exception as ex:
        log(f"repair author error: {ex}"); return None
    spec = parse_task(text)
    if spec is None:
        log(f"repair unparseable for {task_dir.name}"); return None
    prov = json.loads((task_dir / "provenance.json").read_text()) if (task_dir / "provenance.json").exists() else {}
    prov.update({"repaired_from": task_dir.name, "repair_round": prov.get("repair_round", 0) + 1})
    tid, dest = write_task(spec, pool, source=source, provenance=prov, prefix=task_dir.name.split("__")[0])
    if dest == task_dir:   # identical content
        return None
    (dest / "generation.txt").write_text(text)
    return tid, dest
