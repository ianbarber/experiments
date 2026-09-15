# Working in this repo

Public lab notebook. One dated folder per experiment. Anything pushed here is
visible to everyone, so the sanitisation rules below are not optional.

Scientific process lives in [`GUIDELINES.md`](GUIDELINES.md). Read it before
starting a run. In particular: one public entry per scientific object; unrun
follow-ups are retracted; project-agent review is not a credential.

## Layout of an entry

`YYYY-MM-DD-short-slug/` (date the work started), containing:

| File | Purpose |
|---|---|
| `README.md` | The brief: title; a `**Date:** … · **Machine:** …` line; `## Brief` (the question); `## Headline results` (bullets, numbers included); `## Contents` table |
| `REPORT.md` | The full write-up: method, results, limitations, verdict. Readable in one sitting |
| `LABNOTES.md` | Execution log in the order it happened, failures included, **first person**. If it is reconstructed after the fact, say so in the first paragraph |
| `code/` | Everything needed to reproduce, with a `code/README.md` giving environment and run commands. Paths in docs are relative to `code/` unless stated |
| `results/`, `images/` | Small result files and figures. No multi-MB logs, checkpoints or datasets |

Add a row to the table in the root `README.md`. Negative results are entries too.
Prerequisite misses of the same question are not new rows — fold them into the
entry that started the question.

## Never commit

- Lab hostnames, LAN or tailnet addresses, machine serials. Use the role aliases
  below. A content scrub without a history rewrite does not hide the old bytes;
  do not put the mapping in the commit subject.
- API keys, tokens, `.env` files, anything from `~/.config` or `~/.ssh`. Do not
  document the path of a live key file either.
- Personal data: unpublished writing, **chat or agent-session transcripts**,
  other people's names or device details. Work that needs those stays in a
  private repo.
- Operator-log voice: third-person "Ian", "from Ian", "Ian's ask", "not doing
  this unilaterally", Claude/Codex/Cursor session or artifact URLs, billing
  dashboards, account balances, reboot scoldings.
- Other unpublished project names, systemd units, and home-lab inventory
  (`~/Projects/<other>`, large weight caches, "what's using RAM on that box").
- Raw HTTP-level or per-token logs. Summarise them into `LABNOTES.md` instead.
- `__pycache__/`, `*.pyc`, `*.egg-info/`, venvs, checkpoints, weights.

Absolute paths under the home directory are tolerated only inside `code/`
configs where a relative path is impossible; prefer `$HOME`, environment
variables, and paths relative to the entry. They do not belong in README
tables, tracebacks committed as notes, or yaml that could use a relative
package path.

`LABNOTES.md` is written as the experimenter. Using an agent is a methods
fact ("orchestrated with Claude Code"). Pasting the session is a leak.

## Machine aliases

| Alias | Means |
|---|---|
| `dgx-spark` | the GB10 / DGX Spark (aarch64, sm_121), usually the model-serving box |
| `strix-halo` | the AMD Ryzen AI Max+ 395 box (gfx1151, ROCm) |
| `worker-a`, `worker-b` | the two x86 Docker workers; `worker-b` carries the RTX 5090 |
| `nas` | the NAS export mounted at `/mnt/nas` |

Use these in prose, tables and config files alike (e.g. `http://dgx-spark:8888/v1`).
Do not comment "LAN address redacted" next to them.

## Before every push

```bash
git ls-files -z | xargs -0 grep -nE '(sk-[A-Za-z0-9]{16,}|ghp_[A-Za-z0-9]{16,}|hf_[A-Za-z0-9]{16,}|AKIA[A-Z0-9]{12,}|PRIVATE[ ]KEY|\.ts\.net|\b(192\.168|10\.[0-9]+|172\.(1[6-9]|2[0-9]|3[0-1])|100\.[0-9]+)\.[0-9]+\.[0-9]+|claude\.ai/(code|artifact)|from Ian|not doing this unilaterally)' --
git ls-files | grep -E '(__pycache__|\.pyc$|\.egg-info/|\.env$)' && echo 'junk files tracked — stop'
```

Any hit is a stop. Extend the pattern rather than skipping the check. Also
open `git diff --stat` and read it; bytecode in the newest experiment is how
this list grew last time.

## Importing a standalone repo

`git subtree add --prefix=<entry> <repo-url> main` keeps its history, then
restructure to the layout above in a second commit. History of a private source
repo becomes public; assume that is permanent. If the entry depends on another
repo, snapshot the few documents it needs into `background/` with the source
commit pinned, rather than linking to something private. Archive the source
repo with a pointer README once the import is pushed.

## Agents working in this repo

- Follow `GUIDELINES.md`. Do not invent follow-up lists. Do not create
  `independent_review/` trees or title files as if they were external
  replication.
- Write notes in the first person of the experimenter, or as a clearly labeled
  reconstruction. Do not address the operator in the third person.
- Do not commit bytecode, egg-info, or paths that name other private trees.
