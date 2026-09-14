# Task generator brief

You write terminal tasks for an autonomous agent that works in a Linux shell. Each task is a small, realistic job a developer or sysadmin could be given, solvable with the tools in the sandbox, and verified by a hidden pytest suite that inspects the container after the agent finishes.

## Sandbox facts (do not contradict them)
- The agent starts in `/app` as the unprivileged user `agent` (root only if `needs_root: true`). No internet at run time.
- Image: Ubuntu 24.04 with python3 (numpy, pandas, polars, scipy, scikit-learn, matplotlib, requests, httpx, pyyaml, pytest, flask, fastapi, beautifulsoup4, lxml, openpyxl, ruff), node 18 + npm (typescript, ts-node, prettier, eslint, jest), gcc/g++/make/cmake/gdb/valgrind, git, sqlite3, jq, curl, wget, rsync, tar/zip/zstd/7z, ripgrep, fd, tmux, vim, pandoc, ImageMagick, pdftotext.
- Extra packages may be installed at BUILD time only, in `setup.sh` (apt-get / pip3 / npm). The agent cannot install anything.
- The agent has at most 30 commands, 120 s each, and sees at most 4,000 characters of output per command.

## What makes a good task
- One clear objective with 2-6 concrete, checkable requirements (paths, formats, exact names, edge cases).
- Realistic materials: generate the input files (data, code with a bug, config, logs) in `setup.sh` or as files. Do not reveal the solution in the instruction or the files.
- Solvable in 3-25 commands by a competent engineer; medium tasks usually need reading files, running something, fixing or writing code, and checking the result.
- Deterministic verification: tests read the container state (`/app/...`) and compare against values computed from the task's own inputs. No randomness, no network, no timing-sensitive checks.
- Robust tests: accept any correct solution (do not test implementation details); check outputs, files, exit codes, content; use tolerances for floats; several independent tests so partial progress is visible.
- The solution script must pass all tests when run from a clean container; a no-op agent must fail every test.
- In SOLUTION, write whole files with heredocs (cat > path <<'EOF' ... EOF) rather than editing with sed/awk; keep it simple and deterministic.
- Before finishing, walk through the tests against your solution's actual output (exact strings, rounding, ordering, file paths) and fix any mismatch.

## Output format (exactly this, nothing else)
Write these sections, each followed by ONE fenced code block:

### META
```yaml
name: <kebab-case-slug>
domain: <one of: data_processing, data_querying, data_science, debugging, dependency_management, file_operations, scientific_computing, security, software_engineering>
tier: <easy|medium|hard>
needs_root: false
summary: <one line>
```

### INSTRUCTION
```markdown
<what the agent reads; be precise about required paths/names/formats>
```

### SETUP
```bash
<runs as root at image build time from /app: create input files, seed a repo, apt-get/pip3 installs if needed; leave /app in the task's starting state>
```

### SOLUTION
```bash
<reference solution the oracle runs as the agent user from /app; must make every test pass>
```

### TESTS
```python
<pytest file; import only from the standard library plus pandas/numpy/yaml when useful; read from /app; each test a separate function>
```
