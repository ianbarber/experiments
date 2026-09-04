# Working in this repo

Public lab notebook. One dated folder per experiment. Anything pushed here is visible to
everyone, so the sanitisation rules below are not optional.

## Layout of an entry

`YYYY-MM-DD-short-slug/` (date the work started), containing:

| File | Purpose |
|---|---|
| `README.md` | The brief: title; a `**Date:** … · **Machine:** …` line; `## Brief` (the question); `## Headline results` (bullets, numbers included); `## Contents` table |
| `REPORT.md` | The full write-up: method, results, limitations, verdict |
| `LABNOTES.md` | Execution log in the order it happened, failures included. If it is reconstructed after the fact, say so in the first paragraph |
| `code/` | Everything needed to reproduce, with a `code/README.md` giving environment and run commands. Paths in docs are relative to `code/` unless stated |
| `results/`, `images/` | Small result files and figures. No multi-MB logs, checkpoints or datasets |

Add a row to the table in the root `README.md`. Negative results are entries too.

## Never commit

- Lab hostnames, LAN or tailnet addresses, machine serials. Use the role aliases below.
- API keys, tokens, `.env` files, anything from `~/.config` or `~/.ssh`.
- Personal data: unpublished writing, chat or agent-session transcripts, other people's
  names or device details. Work that needs those stays in a private repo.
- Raw HTTP-level or per-token logs. Summarise them into `LABNOTES.md` instead.

Absolute paths under the home directory are tolerated only inside `code/` configs where a
relative path is impossible; prefer relative paths and environment variables.

## Machine aliases

| Alias | Means |
|---|---|
| `dgx-spark` | the GB10 / DGX Spark (aarch64, sm_121), usually the model-serving box |
| `strix-halo` | the AMD Ryzen AI Max+ 395 box (gfx1151, ROCm) |
| `worker-a`, `worker-b` | the two x86 Docker workers; `worker-b` carries the RTX 5090 |
| `nas` | the NAS export mounted at `/mnt/nas` |

Use these in prose, tables and config files alike (e.g. `http://dgx-spark:8888/v1`).

## Before every push

```bash
grep -rnE '(sk-[A-Za-z0-9]{16,}|ghp_[A-Za-z0-9]{16,}|hf_[A-Za-z0-9]{16,}|AKIA[A-Z0-9]{12,}|PRIVATE[ ]KEY|\.ts\.net|\b(192\.168|10\.[0-9]+)\.[0-9]+\.[0-9]+)' --exclude-dir=.git .
```

Any hit is a stop. Extend the pattern rather than skipping the check.

## Importing a standalone repo

`git subtree add --prefix=<entry> <repo-url> main` keeps its history, then restructure to
the layout above in a second commit. If the entry depends on another repo, snapshot the
few documents it needs into `background/` with the source commit pinned, rather than
linking to something private. Archive the source repo with a pointer README once the
import is pushed.
