#!/usr/bin/env python3
"""Cap counterfactual: replay every anchored grep of an arm-E run with the display caps
removed and record, per episode, the full set of files an uncapped addendum would have
named. Runs on a worker (docker + NAS). Output: <out>.jsonl, one row per episode:
{iid, shown_files (capped addenda as the agent saw them), uncapped_files, uncapped_missing
 (files absent from the grep output), n_anchored, n_replayed, symbols_capped, symbols_uncapped}

Usage: python3 anchor_replay.py <run_name> <out.jsonl> [--pkg DIR] [--only iid,...]
"""
import argparse, glob, json, os, re, subprocess, sys, time
sys.path.insert(0, os.path.expanduser("~/refactorbench-eval/anchor"))
from anchor_env import extract_search, parse_hits, visible_text, GREP_FAMILY  # noqa: E402

HDR = "--- code anchors (language server)"
FILE_RE = re.compile(r"([\w./@+~-]+\.py)")
NAS_LSP = "/mnt/nas/refactorbench/images-lsp"


def sh(*a, **k):
    return subprocess.run(a, text=True, capture_output=True, **k)


def ensure_image(iid):
    if sh("docker", "image", "inspect", f"promax-lsp:{iid}").returncode == 0:
        return True
    tar = f"{NAS_LSP}/{iid}.tar.zst"
    if not os.path.exists(tar):
        return False
    return subprocess.run(f"zstd -dc {tar} | docker load >/dev/null", shell=True).returncode == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run"); ap.add_argument("out")
    ap.add_argument("--pkg", default=os.path.expanduser("~/refactorbench-eval/anchor-replay/lsp_tool"))
    ap.add_argument("--only", default="")
    ap.add_argument("--keep-images", action="store_true")
    args = ap.parse_args()
    only = set(args.only.split(",")) if args.only else None
    done = set()
    if os.path.exists(args.out):
        for line in open(args.out):
            try: done.add(json.loads(line)["iid"])
            except ValueError: pass
    for f in sorted(glob.glob(f"/mnt/nas/refactorbench/runs/{args.run}/*/*.traj.json")):
        iid = os.path.basename(os.path.dirname(f))
        if iid in done or (only and iid not in only):
            continue
        t = json.load(open(f)); msgs = t["messages"]
        # (command, observation-without-addendum) for every anchored observation
        pairs = []; shown = set(); sym_capped = 0
        for i, m in enumerate(msgs):
            if m.get("role") != "tool" or HDR not in (m.get("content") or ""):
                continue
            c = m["content"]; add = c[c.index(HDR):]
            shown |= set(FILE_RE.findall(add))
            hm = re.search(r": (\d+) symbol", add); sym_capped += int(hm.group(1)) if hm else 0
            # the command that produced it: the assistant action with this tool_call_id
            tcid = m.get("tool_call_id"); cmd = None
            for j in range(i - 1, -1, -1):
                mm = msgs[j]
                if mm.get("role") == "assistant":
                    for a in (mm.get("extra") or {}).get("actions", []):
                        if a.get("tool_call_id") == tcid: cmd = a.get("command")
                    if cmd: break
            if cmd:
                obs = c[:c.index(HDR)]
                obs = re.sub(r"^<returncode>\d+</returncode>\n<output>\n", "", obs)
                pairs.append((cmd, obs))
        row = {"iid": iid, "n_anchored": len(pairs), "n_replayed": 0, "shown_files": sorted(shown),
               "symbols_capped": sym_capped, "symbols_uncapped": 0, "uncapped_files": [], "uncapped_missing": [],
               "errors": 0}
        if pairs:
            if not ensure_image(iid):
                row["errors"] = -1
            else:
                cid = sh("docker", "run", "-d", "-w", "/testbed", f"promax-lsp:{iid}", "sleep", "2h").stdout.strip()
                try:
                    target = sh("docker", "exec", cid, "/opt/lsp-tool/bin/python", "-c",
                                "import lsp_tool,os;print(os.path.dirname(lsp_tool.__file__))").stdout.strip()
                    sh("docker", "cp", os.path.join(args.pkg, "."), f"{cid}:{target}/")
                    sh("docker", "exec", "-w", "/testbed", cid, "lsp", "daemon", "start")
                    unc = set(); unc_missing = set()
                    for cmd, obs in pairs:
                        search = extract_search(cmd)
                        parsed = parse_hits(visible_text(obs), (".py", ".pyi"), 40,
                                            single_file=search["single_file"], candidates=search["candidates"])
                        if not parsed["hits"]:
                            continue
                        payload = {"hits": parsed["hits"], "grep_files": parsed["grep_files"],
                                   "tokens": search["tokens"], "ignore_case": search["ignore_case"],
                                   "base_dir": search["base_dir"], "budget": 30, "max_hits": 40,
                                   "max_symbols": 50, "max_candidates": 50, "raw": True}
                        r = subprocess.run(["docker", "exec", "-i", "-w", "/testbed", "-e", "LSP_TOOL_TIMEOUT=60",
                                            cid, "lsp", "anchor"], input=json.dumps(payload), text=True,
                                           capture_output=True, timeout=120)
                        if r.returncode != 0 or not r.stdout.strip():
                            row["errors"] += 1; continue
                        try:
                            d = json.loads(r.stdout)
                        except ValueError:
                            row["errors"] += 1; continue
                        row["n_replayed"] += 1
                        for s_ in d["symbols"]:
                            if s_["ref_files"]:
                                row["symbols_uncapped"] += 1
                            unc |= set(s_["ref_files"]); unc.add(s_["def"].split(":")[0])
                            unc_missing |= set(s_["missing"])
                    row["uncapped_files"] = sorted(unc); row["uncapped_missing"] = sorted(unc_missing)
                finally:
                    sh("docker", "rm", "-f", cid)
                    if not args.keep_images:
                        sh("docker", "rmi", f"promax-lsp:{iid}")
        with open(args.out, "a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(f"{iid}: anchored {row['n_anchored']} replayed {row['n_replayed']} shown {len(row['shown_files'])} "
              f"uncapped {len(row['uncapped_files'])} errors {row['errors']}", flush=True)


if __name__ == "__main__":
    main()
