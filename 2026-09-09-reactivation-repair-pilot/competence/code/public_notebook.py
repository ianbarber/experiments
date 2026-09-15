#!/usr/bin/env python3
"""Generate and execute a portable public notebook from the projected bundle.

The original analysis utilities remain byte-exact. This publication layer adapts
only notebook setup, kernel reporting and temporary-data cleanup. It never copies
an existing locally executed notebook or its absolute interpreter proof.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

import nbformat
from nbclient import NotebookClient

from export import MAX_PUBLIC_FILE, encoded, read, require, scan, sha, sha_bytes
import replay

HERE = Path(__file__).resolve().parent
KERNEL_CELL = '''import os, sys, json, hashlib, importlib.util, tempfile, platform
from pathlib import Path
from IPython.display import display, Markdown, Image
sys.dont_write_bytecode = True
if any(importlib.util.find_spec(name) is not None for name in ('torch', 'transformers')):
    raise RuntimeError('Use the analysis-only environment without model libraries')
expected_prefix = os.environ.get('PUBLIC_LEDGER_EXPECTED_PREFIX')
expected_executable = os.environ.get('PUBLIC_LEDGER_EXPECTED_EXECUTABLE')
same_prefix = bool(expected_prefix) and sys.prefix == expected_prefix
same_executable = bool(expected_executable) and Path(sys.executable).resolve() == Path(expected_executable).resolve()
if expected_prefix and not same_prefix:
    raise RuntimeError('Actual notebook kernel environment differs')
if expected_executable and not same_executable:
    raise RuntimeError('Actual notebook kernel interpreter differs')
probe = {'same_environment_as_executor': same_prefix, 'same_interpreter_as_executor': same_executable,
         'python_version': platform.python_version(), 'torch': False, 'transformers': False}
print('PUBLIC_LEDGER_KERNEL_PROBE ' + json.dumps(probe))'''

SETUP_CELL = '''# Locate the entry by its manifest, independently of checkout or home paths.
ENTRY = next((p for p in (Path.cwd(), *Path.cwd().parents)
              if (p / 'results/PUBLIC_BUNDLE.json').is_file()), None)
if ENTRY is None:
    raise RuntimeError('Run this notebook from a directory inside the public entry')
sys.path.insert(0, str(ENTRY / 'code'))
import replay as public_replay
public_manifest, public_payloads = public_replay.verify_bundle(ENTRY)
PUBLIC_TEMPORARY = tempfile.TemporaryDirectory(prefix='public-ledger-notebook-')
ROOT = Path(PUBLIC_TEMPORARY.name) / 'evidence'
ROOT.mkdir()
public_analyzer = public_replay.materialize(ROOT, public_manifest, public_payloads)
ANALYSIS = Path(PUBLIC_TEMPORARY.name) / 'analysis'
UTILITIES = ROOT / 'results/scope_analysis_tools'
saved, completion = public_analyzer.analyze(ROOT, ANALYSIS, allow_projection=True)
public_replay.compare(json.loads(public_payloads['__analysis__/analysis.json']), saved)
if completion['projection_used'] is not True:
    raise RuntimeError('Public notebook requires explicit scientific projection mode')
print(saved['measurement_scope'])
print(json.dumps({'outcome': saved['outcome'], 'counts': saved['counts'], 'projection_used': True}, indent=2))'''


def analysis_environment():
    require(not any(importlib.util.find_spec(name) is not None for name in ('torch', 'transformers')),
            'Use an analysis-only environment without model libraries')


def adapt(template, provenance):
    """Retain all scientific cells; explicitly replace only environment/setup."""
    notebook = nbformat.reads(nbformat.writes(template), as_version=4)
    cells = [cell for cell in notebook.cells if cell.cell_type == 'code']
    require(len(cells) == 9 and 'LEDGER_KERNEL_PROBE' in cells[0].source
            and 'LEDGER_ROOT' in cells[1].source and "allow_projection=completion['projection_used']" in cells[2].source,
            'Frozen notebook template changed; publication adapter needs review')
    original_hashes = [sha_bytes(cell.source.encode()) for cell in cells]
    cells[0].source = KERNEL_CELL
    cells[1].source = SETUP_CELL
    cells[-1].source += "\nPUBLIC_TEMPORARY.cleanup()\nprint('PUBLIC_LEDGER_TEMPORARY_DATA_REMOVED')\n"
    for cell in cells:
        cell.outputs = []
        cell.execution_count = None
    notebook.cells[0].source += ('\n\nThis public notebook regenerates facts and materializes explicitly projected '
        'measurements in temporary storage. Kernel identity is checked without publishing interpreter or home paths. '
        'Original token IDs and model execution are not reconstructed.')
    notebook.metadata['public_notebook'] = {
        **provenance, 'version': 1, 'projection_enabled': True,
        'changed_code_cell_indices': [0, 1, 8], 'unchanged_scientific_code_cell_indices': list(range(2, 8)),
        'original_code_cell_sha256': original_hashes,
        'public_code_cell_sha256': [sha_bytes(cell.source.encode()) for cell in cells],
        'transformation': 'Portable bundle materialization, path-free verified kernel probe, and temporary-data cleanup only.'}
    nbformat.validate(notebook)
    return notebook


def public_bytes(notebook, forbidden_literals=()):
    nbformat.validate(notebook)
    payload = nbformat.writes(notebook).encode()
    require(len(payload) < MAX_PUBLIC_FILE, 'Notebook exceeds the compact public-artifact ceiling')
    scan(payload, 'public notebook', forbidden_literals)
    for cell in notebook.cells:
        for output in cell.get('outputs', []):
            require(output.get('output_type') != 'error', 'Errored notebooks cannot be published as completed')
    return payload


def prepare(entry, *, allow_synthetic_fixture=False):
    """Return a portable unexecuted notebook; all intermediate facts are temporary."""
    analysis_environment()
    entry = Path(entry).resolve()
    manifest, payloads = replay.verify_bundle(entry)
    original = json.loads(payloads['__analysis__/analysis.json'])
    synthetic = original.get('synthetic_fixture') is True
    require(not synthetic or allow_synthetic_fixture, 'Actual public notebook production waits actual program completion')
    freeze = json.loads(payloads['__analysis__/IMPLEMENTATION_FREEZE.json'])
    for name in ('analyze.py', 'make_notebook.py'):
        require(sha_bytes(payloads['__analysis__/' + name]) == freeze['files_sha256']['results/scope_analysis_tools/' + name],
                'Frozen notebook/analysis utility identity differs')
    bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        with tempfile.TemporaryDirectory(prefix='public-notebook-authoring-') as temporary:
            root = Path(temporary) / 'evidence'
            root.mkdir()
            analyzer = replay.materialize(root, manifest, payloads)
            result, completion = analyzer.analyze(root, Path(temporary) / 'analysis', allow_projection=True)
            replay.compare(original, result)
            require(completion['projection_used'] is True, 'Projection was not explicitly enabled')
            maker = replay.load_module('public_scope_notebook_maker', root / 'results/scope_analysis_tools/make_notebook.py')
            template_path = maker.build(root, Path(temporary) / 'analysis', Path(temporary) / 'template.ipynb')
            template = nbformat.read(template_path, as_version=4)
            notebook = adapt(template, {
                'synthetic_fixture': synthetic, 'source_bundle_sha256': sha(entry / 'results/PUBLIC_BUNDLE.json'),
                'source_scope_completion_sha256': manifest['original_scope_completion_sha256'],
                'original_terminal_route': manifest['original_terminal']['route'],
                'source_analysis_completion_sha256': manifest['original_analysis_completion_sha256'],
                'analyzer_sha256': freeze['files_sha256']['results/scope_analysis_tools/analyze.py'],
                'original_maker_sha256': freeze['files_sha256']['results/scope_analysis_tools/make_notebook.py'],
                'original_template_sha256': sha(template_path), 'publication_helper_sha256': sha(__file__)})
            public_bytes(notebook, (str(root), str(Path(temporary)), sys.prefix, sys.executable))
            return notebook
    finally:
        sys.dont_write_bytecode = bytecode


def run(entry, destination, *, allow_synthetic_fixture=False):
    """Generate and execute fresh; publish only path-free successful artifacts."""
    entry, destination = Path(entry).resolve(), Path(destination).resolve()
    require(not destination.exists(), 'Notebook output directory must be new')
    notebook = prepare(entry, allow_synthetic_fixture=allow_synthetic_fixture)
    unexecuted = public_bytes(notebook)
    source_bundle = sha(entry / 'results/PUBLIC_BUNDLE.json')
    environment = dict(os.environ, PUBLIC_LEDGER_EXPECTED_PREFIX=sys.prefix,
                       PUBLIC_LEDGER_EXPECTED_EXECUTABLE=sys.executable, PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='')
    # An in-memory notebook runs from the entry. No materialized source, private
    # draft notebook, or private interpreter proof is copied to public storage.
    NotebookClient(notebook, timeout=600, kernel_name='python3',
                   resources={'metadata': {'path': str(entry)}}).execute(env=environment)
    probes, cleanup = [], 0
    for cell in notebook.cells:
        for output in cell.get('outputs', []):
            for line in output.get('text', '').splitlines():
                if line.startswith('PUBLIC_LEDGER_KERNEL_PROBE '):
                    probes.append(json.loads(line.split(' ', 1)[1]))
                cleanup += int(line == 'PUBLIC_LEDGER_TEMPORARY_DATA_REMOVED')
    require(len(probes) == 1 and probes[0]['same_environment_as_executor'] is True
            and probes[0]['same_interpreter_as_executor'] is True
            and probes[0]['torch'] is False and probes[0]['transformers'] is False,
            'Kernel did not attest to the exact analysis-only executor identity')
    require(cleanup == 1 and all(cell.execution_count is not None for cell in notebook.cells if cell.cell_type == 'code'),
            'Notebook cells or temporary-data cleanup did not complete')
    require(sha(entry / 'results/PUBLIC_BUNDLE.json') == source_bundle, 'Public bundle manifest changed during execution')
    replay.verify_bundle(entry)
    executed = public_bytes(notebook, (sys.prefix, sys.executable, str(entry)))
    proof = {'status': 'passed', 'created_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'Fresh public CPU notebook of the amended competence scope from projected measurements and regenerated facts.',
        'synthetic_fixture': notebook.metadata['public_notebook']['synthetic_fixture'],
        'unexecuted_notebook_sha256': sha_bytes(unexecuted), 'executed_notebook_sha256': sha_bytes(executed),
        'source_bundle_sha256': source_bundle, 'publication_helper_sha256': sha(__file__),
        'source_scope_completion_sha256': notebook.metadata['public_notebook']['source_scope_completion_sha256'],
        'original_terminal_route': notebook.metadata['public_notebook']['original_terminal_route'],
        'analyzer_sha256': notebook.metadata['public_notebook']['analyzer_sha256'],
        'original_maker_sha256': notebook.metadata['public_notebook']['original_maker_sha256'],
        'code_cells_executed': sum(cell.cell_type == 'code' for cell in notebook.cells),
        'kernel_verification': probes[0], 'temporary_data_removed': True, 'projection_enabled': True,
        'model_calls': 0, 'weight_files_loaded': False, 'raw_token_ids_reconstructed': False,
        'versions': {name: importlib.metadata.version(name) for name in
                     ('numpy', 'matplotlib', 'nbformat', 'nbclient', 'ipykernel')}}
    proof_bytes = encoded(proof)
    scan(proof_bytes, 'public notebook execution proof', (sys.prefix, sys.executable, str(entry)))
    destination.mkdir(parents=True, exist_ok=False)
    (destination / 'ledger_public.ipynb').write_bytes(unexecuted)
    (destination / 'ledger_public_executed.ipynb').write_bytes(executed)
    (destination / 'PUBLIC_NOTEBOOK_EXECUTION.json').write_bytes(proof_bytes)
    return proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--entry', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, required=True, help='New directory, normally ENTRY/results/notebooks')
    args = parser.parse_args()
    print(json.dumps(run(args.entry, args.output), indent=2))


if __name__ == '__main__':
    main()
