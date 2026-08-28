#!/usr/bin/env python3
"""Delta-scoped Pyrefly acceptance gate for Arm D.

--record-baseline: run pyrefly over /testbed, save error fingerprints.
default:           run pyrefly, diff against baseline; exit 0 if no new errors,
                   else print an <acceptance_gate> rejection (cap 8) and exit 1.

Design follows ianbarber/lsps-for-llms REPORT.md: delta scoping against a
pre-edit baseline, syntax-cascade demotion, capped output, directive text only
at gate time (no standing per-turn orders).
"""
import json, subprocess, sys, os, ast, time

BASELINE = "/tmp/.typecheck_baseline.json"
ATTEMPTS = "/tmp/.gate_rejections"
PYREFLY = "/opt/lsp-tool/bin/pyrefly"
ROOT = "/testbed"
CAP = 8


def soft_limit():
    """--soft-limit N: after N prior rejections, pass silently (bounds budget burn)."""
    if "--soft-limit" in sys.argv:
        try:
            return int(sys.argv[sys.argv.index("--soft-limit") + 1])
        except Exception:
            return 2
    return None


def rejections():
    try:
        return int(open(ATTEMPTS).read().strip())
    except Exception:
        return 0


def run_pyrefly():
    try:
        p = subprocess.run(
            [PYREFLY, "check", "--output-format", "json"],
            cwd=ROOT, capture_output=True, text=True, timeout=240,
        )
    except Exception as e:
        return None, f"checker failed to run: {e}"
    out = p.stdout.strip()
    if not out:
        return [], None
    try:
        data = json.loads(out)
    except json.JSONDecodeError:
        return None, "checker produced unparseable output"
    errs = data.get("errors", data if isinstance(data, list) else [])
    norm = []
    for e in errs:
        if not isinstance(e, dict):
            continue
        path = e.get("path") or e.get("file") or ""
        path = os.path.relpath(path, ROOT) if os.path.isabs(path) else path
        norm.append({
            "path": path,
            "line": e.get("line") or (e.get("range") or {}).get("start", {}).get("line") or 0,
            "col": e.get("column") or (e.get("range") or {}).get("start", {}).get("column") or 0,
            "code": e.get("name") or e.get("code") or e.get("kind") or "error",
            "msg": (e.get("description") or e.get("message") or "").split("\n")[0][:200],
        })
    return norm, None


def fingerprint(e):
    return f"{e['path']}|{e['code']}|{e['msg']}"


def parses(path):
    try:
        ast.parse(open(os.path.join(ROOT, path), encoding="utf-8", errors="replace").read())
        return True
    except SyntaxError:
        return False
    except Exception:
        return True


def main():
    if "--record-baseline" in sys.argv:
        errs, fail = run_pyrefly()
        json.dump({"fps": [fingerprint(e) for e in (errs or [])], "ok": errs is not None},
                  open(BASELINE + ".tmp", "w"))
        os.replace(BASELINE + ".tmp", BASELINE)
        return 0

    # wait briefly for the startup baseline if it hasn't landed yet
    for _ in range(60):
        if os.path.exists(BASELINE):
            break
        time.sleep(2)
    if not os.path.exists(BASELINE):
        return 0  # no baseline -> gate cannot judge; never block submission on infra failure
    base = json.load(open(BASELINE))
    if not base.get("ok", False):
        return 0

    errs, fail = run_pyrefly()
    if errs is None:
        return 0  # checker infra failure -> do not block
    known = set(base.get("fps", []))
    new = [e for e in errs if fingerprint(e) not in known]
    # demote cascades from files that no longer parse: syntax first, cap total
    syntax_files = {e["path"] for e in new if not parses(e["path"])}
    new.sort(key=lambda e: (e["path"] not in syntax_files, e["path"], e["line"]))
    if not new:
        return 0
    lim = soft_limit()
    prior = rejections()
    if lim is not None and prior >= lim:
        return 0  # soft gate: rejected `lim` times already; let this one through silently
    with open(ATTEMPTS, "w") as fh:  # count first: open("w") truncates before any read
        fh.write(str(prior + 1))
    shown = new[:CAP]
    print('<acceptance_gate status="rejected">')
    print(f"Submission not accepted: the type checker reports {len(new)} new issue(s) after your changes.")
    for e in shown:
        tag = "syntax" if e["path"] in syntax_files else "type"
        print(f"[{tag}] {e['path']}:{e['line']}:{e['col']} {e['code']}: {e['msg']}")
    if len(new) > CAP:
        print(f"... {len(new) - CAP} additional issue(s) not shown.")
    print("Revise your changes, regenerate patch.txt, then submit again with the same command.")
    print("</acceptance_gate>")
    return 1


if __name__ == "__main__":
    sys.exit(main())
