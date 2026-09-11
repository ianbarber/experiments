#!/usr/bin/env python3
"""Regenerate facts and replay projected scientific measurements on CPU only."""
from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import math
from pathlib import Path
import re
import shutil
import sys
import tempfile

from export import SCHEMA, TERMINAL, TOOLS, SCOPE_OUTCOME, MAX_PUBLIC_FILE, encoded, read, require, safe, scan, sha_bytes

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_bundle(entry):
    entry = Path(entry).resolve()
    manifest_path = entry / 'results/PUBLIC_BUNDLE.json'
    require(manifest_path.stat().st_size <= MAX_PUBLIC_FILE, 'Public bundle manifest exceeds compact artifact ceiling')
    scan(manifest_path.read_bytes(), 'PUBLIC_BUNDLE.json')
    manifest = read(manifest_path)
    require(manifest['schema'] == SCHEMA, 'Unknown public bundle schema')
    require(manifest.get('scope_outcome') == SCOPE_OUTCOME and manifest.get('root_terminal_path') == TERMINAL,
            'Public bundle lacks the distinct accepted competence-scope terminal')
    payloads, public_paths = {}, set()
    for logical, record in manifest['files'].items():
        require(record['path'] not in public_paths, 'Duplicate transport artifact path')
        public_paths.add(record['path'])
        transport = safe(entry, record['path']).read_bytes()
        require(len(transport) == record['bytes'] <= MAX_PUBLIC_FILE and sha_bytes(transport) == record['sha256'],
                'Public transport hash or length differs: ' + logical)
        require(record['codec'] in ('identity', 'gzip'), 'Unknown transport codec')
        payload = gzip.decompress(transport) if record['codec'] == 'gzip' else transport
        require(len(payload) == record['decoded_bytes'] and sha_bytes(payload) == record['decoded_sha256'],
                'Decoded payload hash or length differs: ' + logical)
        scan(payload, logical)
        payloads[logical] = payload
    projection = json.loads(payloads['PUBLIC_PROJECTION.json'])
    require(projection['version'] == 1, 'Unknown scientific projection schema')
    for logical, change in projection['files'].items():
        outputs = bool(re.fullmatch(r'results/(development|validation)/[^/]+/outputs\.jsonl', logical))
        require(outputs or logical == 'results/program.jsonl', 'Projection may not transform scope/source/data/contracts')
        expected = ('omit_generated_token_ids', ['generated.token_ids']) if outputs else ('omit_operational_argv', ['argv'])
        require((change.get('operation'), change.get('removed_fields')) == expected, 'Unapproved projection operation')
    for logical, record in manifest['files'].items():
        change = projection['files'].get(logical)
        if change:
            require(change['original_sha256'] == record['original_sha256'] and
                    change['projected_sha256'] == record['decoded_sha256'], 'Projection provenance differs')
        else:
            require(record['original_sha256'] == record['decoded_sha256'], 'Undeclared changed scientific artifact')
    require(set(projection['files']) <= set(payloads), 'Projection declares an absent artifact')
    require(TERMINAL in payloads and sha_bytes(payloads[TERMINAL]) == manifest['original_scope_completion_sha256'],
            'Scope terminal hash differs')
    terminal = json.loads(payloads[TERMINAL])
    require(terminal['outcome'] == SCOPE_OUTCOME and terminal['original_terminal'] == manifest['original_terminal'],
            'Scope/original termination semantics differ')
    original = terminal['original_terminal']
    require(original['route'] in ('original_numerical_stop', 'administrative_guard_reached'), 'Unknown terminal route')
    require(original['path'] in payloads and sha_bytes(payloads[original['path']]) == original['sha256'],
            'Actual original terminal identity differs')
    if original['route'] == 'administrative_guard_reached':
        require('results/PROGRAM_COMPLETED.json' not in payloads and original['path'].endswith('/FAILED.json'),
                'Administrative route must not fabricate original program completion')
    else:
        require(original['path'] == 'results/PROGRAM_COMPLETED.json', 'Natural route lacks the genuine original completion')
    if '__review__/REVIEW_PACKET.json' in payloads:
        require(manifest.get('early_review_packet_sha256') == sha_bytes(payloads['__review__/REVIEW_PACKET.json']),
                'Early-review packet manifest differs')
    return manifest, payloads


def materialize(root, manifest, payloads):
    """Write only into an externally created temporary root."""
    root = Path(root)
    analysis_files = {relative.removeprefix(TOOLS + '/')
                      for relative in json.loads(payloads['__analysis__/IMPLEMENTATION_FREEZE.json'])['files_sha256']
                      if relative.startswith(TOOLS + '/')}
    for logical, payload in payloads.items():
        if logical.startswith(('__publication__/', '__review__/')) or logical == 'PUBLIC_PRIOR_BACKGROUND.json':
            continue
        if logical.startswith('__analysis__/'):
            name = logical.removeprefix('__analysis__/')
            if name not in analysis_files | {'IMPLEMENTATION_FREEZE.json'}:
                continue
            logical = TOOLS + '/' + name
        destination = safe(root, logical)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            require(destination.is_file() and destination.read_bytes() == payload,
                    'Duplicated materialized source has different bytes: ' + logical)
            continue
        with destination.open('xb') as stream:
            stream.write(payload)
    snapshot = safe(root, manifest['source_snapshot'])
    # Source mirrors are exact convenience copies, never different analysis code.
    for directory in ('scripts', 'configs'):
        shutil.copytree(snapshot / directory, root / directory)
    shutil.copy2(snapshot / 'PROTOCOL.md', root / 'PROTOCOL.md')
    analyzer = load_module('public_ledger_scope_analysis', root / TOOLS / 'analyze.py')
    with analyzer.frozen_modules(snapshot) as modules:
        data_manifest = read(root / 'data/manifest.json')
        exclusions = read(root / 'data/prior_semantic_exclusions.json')
        generated = root / 'regenerated_data'
        modules['make_data'].write_datasets(generated, exclusions, data_manifest['seed'])
        for logical, expected in manifest['regenerated_data_sha256'].items():
            generated_file = generated / Path(logical).name
            existing = safe(root, logical)
            require(generated_file.is_file() or existing.is_file(), 'Missing data replay artifact: ' + logical)
            payload = generated_file.read_bytes() if generated_file.is_file() else existing.read_bytes()
            require(sha_bytes(payload) == expected, 'Regenerated frozen data differs: ' + logical)
            existing.parent.mkdir(parents=True, exist_ok=True)
            if existing.exists():
                require(existing.read_bytes() == payload, 'Published data metadata differs from regeneration')
            else:
                existing.write_bytes(payload)
        shutil.rmtree(generated)
        require(manifest['cohorts'] == {}, 'Canceled conditional scope cannot contain an actual failure cohort')
    return analyzer


def compare(left, right, location='root'):
    """Reject structural drift; tolerate only tiny floating-point replay variation."""
    if isinstance(left, dict):
        require(isinstance(right, dict) and set(left) == set(right), location + ': dictionary keys differ')
        return sum(compare(left[k], right[k], location + '.' + k) for k in left)
    if isinstance(left, list):
        require(isinstance(right, list) and len(left) == len(right), location + ': list lengths differ')
        return sum(compare(a, b, location + f'[{i}]') for i, (a, b) in enumerate(zip(left, right)))
    if type(left) in (int, float) and type(right) in (int, float):
        require(math.isfinite(left) and math.isfinite(right) and math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12),
                location + ': numeric result differs')
        return 1
    require(type(left) is type(right) and left == right, location + ': value differs')
    return 0


def replay(entry, output=None):
    manifest, payloads = verify_bundle(entry)
    if output is not None:
        output = Path(output).resolve()
        require(not output.exists(), 'Replay output must be a new directory')
    old_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        with tempfile.TemporaryDirectory(prefix='ledger-public-cpu-replay-') as temporary:
            root = Path(temporary) / 'evidence'
            root.mkdir()
            analyzer = materialize(root, manifest, payloads)
            result, completed = analyzer.analyze(root, Path(temporary) / 'analysis', allow_projection=True)
            expected = json.loads(payloads['__analysis__/analysis.json'])
            numbers = compare(expected, result)
            review_proof = None
            if '__review__/REVIEW_PACKET.json' in payloads:
                import review_replay
                review_proof = review_replay.check_materialized(root, manifest, payloads, analyzer)
            proof = {'status': 'passed', 'scope_outcome': result['outcome'], 'compared_numeric_values': numbers,
                'synthetic_fixture': result.get('synthetic_fixture', False),
                'source_scope_completion_sha256': manifest['original_scope_completion_sha256'],
                'original_terminal': manifest['original_terminal'],
                'source_analysis_completion_sha256': manifest['original_analysis_completion_sha256'],
                'scientific_fields_compared': list(expected), 'data_regeneration': 'exact original hashes',
                'projection': 'Token IDs and operational argv omitted; original raw output byte hashes remain recorded identities.',
                'model_calls': 0, 'weights_loaded': False, 'raw_token_bytes_reconstructed': False,
                'analysis_artifacts_sha256': completed['artifacts_sha256'], 'early_review_replay': review_proof}
            if output is not None:
                shutil.copytree(Path(temporary) / 'analysis', output)
                (output / 'PUBLIC_REPLAY.json').write_bytes(encoded(proof))
            return proof
    finally:
        sys.dont_write_bytecode = old_bytecode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--entry', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, help='Optional new directory for replayed tables/figures; never required')
    parser.add_argument('--verify-only', action='store_true', help='Verify transport/projection hashes without importing analysis libraries')
    args = parser.parse_args()
    if args.verify_only:
        manifest, _ = verify_bundle(args.entry)
        print(json.dumps({'status': 'transport_and_projection_hashes_passed', 'files': len(manifest['files'])}, indent=2))
    else:
        print(json.dumps(replay(args.entry, args.output), indent=2))


if __name__ == '__main__':
    main()
