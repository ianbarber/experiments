"""TerminalWorld-verified import (docs/02): the out-of-genre held-out stratum. 200 human-verified tasks, CC-BY-NC-4.0
(evaluation only, never redistributed, never a seed). Kept as close to upstream as the sandbox allows: original
Dockerfile (build-time network is fine) plus a pytest install, entrypoint kept, test.sh replaced by the standard runner
so verification needs no network, agent runs as root (upstream assumption), canary strings kept."""
from __future__ import annotations
import gzip, json, re, shutil, tarfile
from pathlib import Path
from rsi import config as C
from rsi.tasks.seeds import STANDARD_TEST_SH, write_task_toml

TW = C.DATASETS_DIR / "TerminalWorld"

def verified_rows() -> list[dict]:
    return [json.loads(l) for l in gzip.open(TW / "data" / "verified.jsonl.gz", "rt")]

def import_tw(row: dict, dest_root: Path, force: bool = False) -> tuple[str, Path, dict]:
    tid = f"tw__{row['task_id']}"
    dest = dest_root / tid
    if dest.exists():
        if not force: return tid, dest, {"skipped": "exists"}
        shutil.rmtree(dest)
    tmp = dest_root / f"_extract_{row['task_id']}"
    if tmp.exists(): shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    with tarfile.open(TW / row["artifact_path"]) as tf:
        tf.extractall(tmp)
    src = next(p for p in tmp.iterdir() if p.is_dir())
    shutil.move(str(src), str(dest)); shutil.rmtree(tmp, ignore_errors=True)
    info = {"test_file": None}
    df = dest / "environment" / "Dockerfile"
    txt = df.read_text()
    txt = txt.rstrip("\n") + ("\n# rsi: pytest for the offline standard verifier\n"
                            "RUN (apt-get update && apt-get install -y --no-install-recommends python3 python3-pytest && rm -rf /var/lib/apt/lists/*) "
                            "|| (apt-get update && apt-get install -y --no-install-recommends python3 python3-pip && pip3 install --break-system-packages pytest) || true\n")
    df.write_text(txt)
    tests = dest / "tests"
    tf_py = next((p for p in tests.glob("test_*.py")), None)
    info["test_file"] = tf_py.name if tf_py else None
    if tf_py and tf_py.name != "test_outputs.py":
        shutil.copy2(tf_py, tests / "test_outputs.py")
    (tests / "test.sh").write_text(STANDARD_TEST_SH.replace("pkill -u agent 2>/dev/null; true\n", ""))
    (tests / "test.sh").chmod(0o755)
    meta = {"source": "terminalworld_verified", "source_id": row["task_id"], "domain": row.get("terminal_domain", ""), "tier": "", "tags": [],
            "description": "TerminalWorld-verified task (evaluation only)"}
    try:
        import tomllib
        t = tomllib.loads((dest / "task.toml").read_text()); m = t.get("metadata", {})
        meta["tier"] = m.get("difficulty", ""); meta["tags"] = m.get("tags", [])
    except Exception:
        pass
    write_task_toml(dest, f"rsi/{tid}", meta, agent_timeout=900.0, verifier_timeout=600.0)
    from rsi.tasks.gate import set_agent_user
    set_agent_user(dest, "root")
    (dest / "provenance.json").write_text(json.dumps({"generator": "terminalworld", "seed": {"dataset": "EuniAI/TerminalWorld", "instance": row["task_id"],
                                                     "licence": row.get("license")}, "iteration": None, "network_mode": "no-network", "needs_root": True,
                                                     "release_eligible": False, "reason": "CC-BY-NC-4.0 evaluation stratum; never trained on"}, indent=1))
    return tid, dest, info

def import_harbor_task_dir(src: Path, dest_root: Path, *, tid: str, source: str, status_note: str, licence: str, force: bool = False) -> tuple[str, Path, dict]:
    """Same treatment as TerminalWorld for any upstream Harbor-layout evaluation task directory (e.g. Terminal-Bench 2.0):
    original Dockerfile + pytest install, entrypoint kept, offline standard verifier, root, canaries kept."""
    dest = dest_root / tid
    if dest.exists():
        if not force: return tid, dest, {"skipped": "exists"}
        shutil.rmtree(dest)
    shutil.copytree(src, dest)
    df = dest / "environment" / "Dockerfile"; txt = df.read_text()
    txt = txt.rstrip("\n") + ("\n# rsi: pytest for the offline standard verifier\n"
                            "RUN (apt-get update && apt-get install -y --no-install-recommends python3 python3-pytest && rm -rf /var/lib/apt/lists/*) "
                            "|| (apt-get update && apt-get install -y --no-install-recommends python3 python3-pip && pip3 install --break-system-packages pytest) || true\n")
    df.write_text(txt)
    tests = dest / "tests"; tf_py = next((p for p in tests.glob("test_*.py")), None)
    if tf_py and tf_py.name != "test_outputs.py": shutil.copy2(tf_py, tests / "test_outputs.py")
    (tests / "test.sh").write_text(STANDARD_TEST_SH.replace("pkill -u agent 2>/dev/null; true\n", "")); (tests / "test.sh").chmod(0o755)
    meta = {"source": source, "source_id": src.name, "domain": "", "tier": "", "tags": [], "description": status_note}
    try:
        import tomllib
        t = tomllib.loads((src / "task.toml").read_text()); m = t.get("metadata", {})
        meta["tier"] = m.get("difficulty", ""); meta["tags"] = m.get("tags", []); meta["domain"] = m.get("category", "")
    except Exception:
        pass
    write_task_toml(dest, f"rsi/{tid}", meta, agent_timeout=900.0, verifier_timeout=900.0)
    from rsi.tasks.gate import set_agent_user
    set_agent_user(dest, "root")
    (dest / "provenance.json").write_text(json.dumps({"generator": source, "seed": {"dataset": source, "instance": src.name, "licence": licence},
                                                     "iteration": None, "network_mode": "no-network", "needs_root": True, "release_eligible": False,
                                                     "reason": status_note}, indent=1))
    return tid, dest, {"test_file": tf_py.name if tf_py else None}

def import_terminal_bench_as_published(run: C.RunPaths, db, src_root: Path, limit: int | None = None, force: bool = False) -> dict:
    """Terminal-Bench as the benchmark defines it: original task.toml (official docker_image where given, original
    verifier with uvx), network PUBLIC, agent root; only the layout is copied. For the comparability number; the LAN
    must be blocked at the host (iptables) before running (docs/03 §11)."""
    dest_root = run.root / "tb20-public"; dest_root.mkdir(parents=True, exist_ok=True)
    tasks = sorted(p.parent for p in Path(src_root).rglob("task.toml"))[: limit or None]
    n = 0
    for src in tasks:
        tid = f"tbpub__{src.name}"; dest = dest_root / tid
        if dest.exists():
            if not force: continue
            shutil.rmtree(dest)
        shutil.copytree(src, dest)
        # normalise task.toml to schema 1.4 but keep docker_image and timeouts; network public; root
        import tomllib
        t = tomllib.loads((src / "task.toml").read_text()); env = t.get("environment", {}); m = t.get("metadata", {})
        toml = f'''schema_version = "1.4"
[task]
name = "rsi/{tid}"
version = "1.0.0"
description = "Terminal-Bench 2.0 task as published (evaluation only)"
keywords = {json.dumps([str(x) for x in m.get("tags", [])][:10])}
[metadata]
domain = {json.dumps(m.get("category", ""))}
tier = {json.dumps(m.get("difficulty", ""))}
source = "terminal_bench_2.0_public"
source_id = {json.dumps(src.name)}
[verifier]
timeout_sec = {float(t.get("verifier", {}).get("timeout_sec", 900.0))}
[agent]
timeout_sec = {float(t.get("agent", {}).get("timeout_sec", 900.0))}
user = "root"
[environment]
network_mode = "public"
build_timeout_sec = {float(env.get("build_timeout_sec", 600.0))}
cpus = {int(env.get("cpus", 2))}
memory_mb = {int(str(env.get("memory", "2G")).rstrip("G")) * 1024 if str(env.get("memory", "2G")).endswith("G") else 2048}
os = "linux"
'''
        if env.get("docker_image"):
            toml += f'docker_image = {json.dumps(env["docker_image"])}\n'
        (dest / "task.toml").write_text(toml)
        (dest / "provenance.json").write_text(json.dumps({"generator": "terminal_bench_2.0", "seed": {"dataset": "terminal-bench@2.0", "instance": src.name, "licence": "Apache-2.0"},
                                                          "iteration": None, "network_mode": "public", "needs_root": True, "release_eligible": False,
                                                          "reason": "Terminal-Bench 2.0 as published; comparability evaluation only"}, indent=1))
        db.upsert_task(tid, f"rsi/{tid}", dest, "terminal_bench_2.0_public", domain=m.get("category"), tier=m.get("difficulty"), iteration_added=-1, status="tbpub",
                       provenance=json.loads((dest / "provenance.json").read_text()))
        n += 1
    db.event("tbpub_import", n=n); return {"imported": n, "dir": str(dest_root)}

def import_terminal_bench(run: C.RunPaths, db, src_root: Path, limit: int | None = None, force: bool = False) -> dict:
    dest_root = run.root / "tb20"; dest_root.mkdir(parents=True, exist_ok=True)
    tasks = sorted(p.parent for p in Path(src_root).rglob("task.toml"))[: limit or None]
    n = 0
    for src in tasks:
        tid = f"tb__{src.name}"
        tid_, dest, info = import_harbor_task_dir(src, dest_root, tid=tid, source="terminal_bench_2.0", status_note="Terminal-Bench 2.0 comparability set (evaluation only, never trained on)", licence="Apache-2.0", force=force)
        if info.get("skipped"): continue
        db.upsert_task(tid, f"rsi/{tid}", dest, "terminal_bench_2.0", iteration_added=-1, status="tb", provenance=json.loads((dest / "provenance.json").read_text()))
        n += 1
    db.event("tb_import", n=n); return {"imported": n, "dir": str(dest_root)}

def import_all(run: C.RunPaths, db, limit: int | None = None, force: bool = False) -> dict:
    dest_root = run.root / "tw-verified"; dest_root.mkdir(parents=True, exist_ok=True)
    rows = verified_rows()[: limit or None]
    n = 0
    for row in rows:
        tid, dest, info = import_tw(row, dest_root, force=force)
        if info.get("skipped"): continue
        db.upsert_task(tid, f"rsi/{tid}", dest, "terminalworld_verified", domain=row.get("terminal_domain"), iteration_added=-1, status="tw",
                       provenance=json.loads((dest / "provenance.json").read_text()))
        n += 1
    db.event("tw_import", n=n)
    return {"imported": n, "dir": str(dest_root)}
