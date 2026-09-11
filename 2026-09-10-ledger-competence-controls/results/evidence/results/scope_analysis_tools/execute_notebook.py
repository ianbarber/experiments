#!/usr/bin/env python3
"""Execute a draft saved-data notebook in a verified analysis-only kernel."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import nbformat
from nbclient import NotebookClient


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def execute(notebook, output, record):
    notebook, output, record = [Path(p).resolve() for p in (notebook, output, record)]
    if output.exists() or record.exists() or len({notebook, output, record}) != 3:
        raise ValueError('Use new, distinct executed-notebook and verification destinations')
    if importlib.util.find_spec('torch') is not None or importlib.util.find_spec('transformers') is not None:
        raise ValueError('Use the analysis-only interpreter without model libraries')
    original_hash = sha(notebook)
    nb = nbformat.read(notebook, as_version=4)
    nbformat.validate(nb)
    env = dict(os.environ, LEDGER_EXPECTED_PREFIX=sys.prefix, LEDGER_EXPECTED_EXECUTABLE=sys.executable,
               PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='')
    NotebookClient(nb, timeout=300, kernel_name='python3',
                   resources={'metadata': {'path': str(notebook.parent)}}).execute(env=env)
    probes = []
    for cell in nb.cells:
        for item in cell.get('outputs', []):
            for line in item.get('text', '').splitlines():
                if line.startswith('LEDGER_KERNEL_PROBE '):
                    probes.append(json.loads(line.split(' ', 1)[1]))
    if len(probes) != 1 or probes[0]['prefix'] != sys.prefix or Path(probes[0]['executable']).resolve() != Path(sys.executable).resolve():
        raise ValueError('Actual notebook kernel probe does not match the analysis interpreter')
    if probes[0]['torch'] is not False or probes[0]['transformers'] is not False:
        raise ValueError('Notebook kernel contains model libraries')
    if sha(notebook) != original_hash:
        raise ValueError('Unexecuted notebook changed during execution')
    output.parent.mkdir(parents=True, exist_ok=True)
    record.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, output)
    result = {'status': 'passed', 'created_utc': datetime.now(timezone.utc).isoformat(),
              'input_notebook_sha256': original_hash, 'executed_notebook_sha256': sha(output),
              'executor_sha256': sha(__file__), 'code_cells_executed': sum(c.cell_type == 'code' for c in nb.cells),
              'actual_kernel': probes[0], 'scope': 'CPU saved-data notebook; no model or API calls'}
    record.write_text(json.dumps(result, indent=2) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('notebook', 'output', 'record'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(execute(args.notebook, args.output, args.record), indent=2))


if __name__ == '__main__':
    main()
