#!/usr/bin/env python3
"""Replay every anchored grep of an arm-E run against a *clean* local language server
(repos checked out at base_commit, PYTHONPATH unset) and record the files an uncapped
addendum would have named. Companion to anchor_replay.py, which replays inside the
rollout images; diffing the two rows per instance exposes server-side under-reporting
(2026-09-14: images that ship PYTHONPATH=/testbed on src-layout repos).

Usage: anchor_replay_local.py <clones_dir> <runs_dir> <promax.json> <image_replay.jsonl> <out.jsonl>
                              [--run s1-arm-e-r1] [--only iid,...]
  clones_dir/<iid>/          repo at base_commit (git fetch --depth 1 origin <sha>)
  runs_dir/<run>/<iid>/*.traj.json   trajectories (the NAS runs directory)
  image_replay.jsonl         anchor_replay.py output (shown_files / symbols_capped are copied)
Needs a venv with this entry's lsp_tool, serena-agent (solidlsp) and pyrefly, and
../agent on PYTHONPATH for anchor_env's parsers (minisweagent is stubbed below).
Run with PYTHONPATH *not* containing the clone directory — that is the bug under test.
"""
import argparse, glob, json, os, re, shutil, subprocess, sys, time, traceback, types, logging

logging.basicConfig(level=logging.CRITICAL)
m = types.ModuleType("minisweagent"); me = types.ModuleType("minisweagent.environments")
md = types.ModuleType("minisweagent.environments.docker")
class _Stub: pass
md.DockerEnvironment = _Stub; md.DockerEnvironmentConfig = _Stub
sys.modules.update({"minisweagent": m, "minisweagent.environments": me, "minisweagent.environments.docker": md})
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "agent"))
from anchor_env import extract_search, parse_hits, visible_text  # noqa: E402
from solidlsp import SolidLanguageServer  # noqa: E402
from solidlsp.ls_config import LanguageServerConfig, LanguageServerId  # noqa: E402
from solidlsp.settings import SolidLSPSettings  # noqa: E402
from lsp_tool.anchor import compute_anchors  # noqa: E402
from lsp_tool.daemon import ServerEntry  # noqa: E402

HDR = "--- code anchors (language server)"


def anchored_pairs(traj):
    msgs = json.load(open(traj))["messages"]; pairs = []
    for i, mm in enumerate(msgs):
        if mm.get("role") != "tool" or HDR not in (mm.get("content") or ""):
            continue
        c = mm["content"]; tcid = mm.get("tool_call_id"); cmd = None
        for j in range(i - 1, -1, -1):
            a_ = msgs[j]
            if a_.get("role") == "assistant":
                for a in (a_.get("extra") or {}).get("actions", []):
                    if a.get("tool_call_id") == tcid: cmd = a.get("command")
                if cmd: break
        if cmd:
            obs = re.sub(r"^<returncode>\d+</returncode>\n<output>\n", "", c[:c.index(HDR)])
            pairs.append((cmd, obs))
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clones"); ap.add_argument("runs"); ap.add_argument("promax"); ap.add_argument("image_replay"); ap.add_argument("out")
    ap.add_argument("--run", default="s1-arm-e-r1"); ap.add_argument("--only", default="")
    ap.add_argument("--data-dir", default=os.path.join(os.environ.get("TMPDIR", "/tmp"), "anchor-replay-local"))
    a = ap.parse_args()
    if os.environ.get("PYTHONPATH", "").startswith(os.path.abspath(a.clones)):
        sys.exit("PYTHONPATH points into the clones directory; unset it (that is the bug being measured)")
    only = set(a.only.split(",")) if a.only else None
    image = {json.loads(l)["iid"]: json.loads(l) for l in open(a.image_replay)}
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["iid"] for l in open(a.out)}
    for iid in sorted(image):
        if iid in done or (only and iid not in only): continue
        ws = os.path.join(a.clones, iid)
        if not os.path.isdir(ws): print(iid, "no clone"); continue
        npy = int(subprocess.run(f"find {ws} -name '*.py' -not -path '*/.git/*' | wc -l", shell=True, capture_output=True, text=True).stdout)
        row = {"iid": iid, "n_py": npy, "n_anchored": 0, "n_replayed": 0, "shown_files": image[iid]["shown_files"],
               "symbols_capped": image[iid]["symbols_capped"], "symbols_uncapped": 0, "uncapped_files": [],
               "uncapped_missing": [], "errors": 0, "symbols": []}
        t0 = time.time()
        try:
            trajs = glob.glob(f"{a.runs}/{a.run}/{iid}/*.traj.json")
            pairs = anchored_pairs(trajs[0]) if trajs else []
            row["n_anchored"] = len(pairs)
            settings = SolidLSPSettings(solidlsp_dir=a.data_dir, project_data_path=os.path.join(a.data_dir, iid),
                                        ls_specific_settings={LanguageServerId.PYTHON_PYREFLY: {"ls_path": shutil.which("pyrefly")}})
            ls = SolidLanguageServer.create(LanguageServerConfig(ls_id=LanguageServerId.PYTHON_PYREFLY), ws, solidlsp_settings=settings)
            ls.start(); time.sleep(10 if npy < 1000 else 60)  # let workspace population finish
            entry = ServerEntry(ws, "python_pyrefly"); entry.ls = ls
            unc, unc_missing = set(), set()
            for cmd, obs in pairs:
                try:
                    cmd_l, obs_l = cmd.replace("/testbed", ws), obs.replace("/testbed", ws)
                    search = extract_search(cmd_l)
                    parsed = parse_hits(visible_text(obs_l), (".py", ".pyi"), 40, single_file=search["single_file"], candidates=search["candidates"])
                    if not parsed["hits"]: continue
                    payload = {"hits": parsed["hits"], "grep_files": parsed["grep_files"], "tokens": search["tokens"],
                               "ignore_case": search["ignore_case"], "base_dir": search["base_dir"], "budget": 60,
                               "max_hits": 40, "max_symbols": 50, "max_candidates": 50, "raw": True}
                    out = compute_anchors(entry, payload)
                    if not out: continue
                    d = json.loads(out); row["n_replayed"] += 1
                    for s_ in d["symbols"]:
                        if s_["ref_files"]: row["symbols_uncapped"] += 1
                        unc |= set(s_["ref_files"]); unc.add(s_["def"].split(":")[0]); unc_missing |= set(s_["missing"])
                        row["symbols"].append({"cmd": cmd[:100], "qual": s_["qual"], "def": s_["def"], "n_ref_files": len(s_["ref_files"]),
                                               "n_refs": sum(s_["ref_files"].values()), "refs_available": s_["refs_available"]})
                except Exception as e:  # noqa: BLE001
                    row["errors"] += 1; print(iid, "grep error", repr(e)[:200])
            row["uncapped_files"] = sorted(unc); row["uncapped_missing"] = sorted(unc_missing)
            try: ls.stop()
            except Exception: pass  # noqa: BLE001
        except Exception:  # noqa: BLE001
            row["errors"] = -1; row["exc"] = traceback.format_exc()[-500:]
        row["seconds"] = round(time.time() - t0, 1)
        with open(a.out, "a") as fh: fh.write(json.dumps(row) + "\n")
        print(f"{iid}: py {npy} anchored {row['n_anchored']} replayed {row['n_replayed']} "
              f"uncapped local {len(row['uncapped_files'])} vs image {len(image[iid]['uncapped_files'])} errors {row['errors']} ({row['seconds']}s)", flush=True)


if __name__ == "__main__":
    main()
