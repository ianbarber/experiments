"""Read-only transport, projection, notebook and privacy inspection.

Writes only a new review receipt in this script's directory. Does not run the
public replay or notebook; verifies their retained bytes and receipt bindings.
"""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import re


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path):
    raw = path.read_bytes()
    return gzip.decompress(raw) if path.suffix == '.gz' else raw


def run(root, package, receipt_name):
    bundle_path = package / 'results/PUBLIC_BUNDLE.json'
    bundle = json.loads(bundle_path.read_text())
    checked, projections, projected_responses = {}, [], 0
    for logical, item in bundle['files'].items():
        path = package / item['path']
        raw = path.read_bytes()
        decoded = gzip.decompress(raw) if item['codec'] == 'gzip' else raw
        assert (len(raw), sha(raw), len(decoded), sha(decoded)) == (item['bytes'], item['sha256'], item['decoded_bytes'], item['decoded_sha256']), logical
        checked[item['path']] = sha(raw)
        if logical.endswith('/outputs.jsonl') and logical.startswith(('results/development/', 'results/validation/')):
            local_raw = (root / logical).read_bytes()
            assert sha(local_raw) == item['original_sha256']
            local = [json.loads(x) for x in local_raw.splitlines() if x.strip()]
            public = [json.loads(x) for x in decoded.splitlines() if x.strip()]
            for row in local:
                assert 'token_ids' in row['generated']
                del row['generated']['token_ids']
            assert local == public
            assert all('token_ids' not in x['generated'] for x in public)
            projected_responses += len(public)
            projections.append(logical)
        elif logical == 'results/program.jsonl':
            local_raw = (root / logical).read_bytes()
            assert sha(local_raw) == item['original_sha256']
            local = [json.loads(x) for x in local_raw.splitlines() if x.strip()]
            for row in local:
                row.pop('argv', None)
            assert local == [json.loads(x) for x in decoded.splitlines() if x.strip()]
            projections.append(logical)
    assert projected_responses == 8064 and len(projections) == 35
    support_path = package / 'results/PUBLIC_SUPPORT_PACKET.json'
    support = json.loads(support_path.read_text())
    for relative, item in support['files'].items():
        raw = (package / relative).read_bytes()
        assert sha(raw) == item['published_sha256'] and len(raw) == item['bytes']
        checked[relative] = sha(raw)
    nb_path = package / 'results/notebooks/ledger_public_executed.ipynb'
    notebook = json.loads(nb_path.read_text())
    execution_path = package / 'results/notebooks/PUBLIC_NOTEBOOK_EXECUTION.json'
    execution = json.loads(execution_path.read_text())
    assert execution['status'] == 'passed' and sha(nb_path.read_bytes()) == execution['executed_notebook_sha256']
    assert sha(bundle_path.read_bytes()) == execution['source_bundle_sha256']
    assert execution['source_scope_completion_sha256'] == sha((root / 'results/COMPETENCE_SCOPE_COMPLETED.json').read_bytes())
    assert execution['projection_enabled'] is True and execution['synthetic_fixture'] is False
    assert execution['model_calls'] == 0 and execution['weight_files_loaded'] is False and execution['raw_token_ids_reconstructed'] is False
    code = [c for c in notebook['cells'] if c['cell_type'] == 'code']
    assert len(code) == execution['code_cells_executed'] == 9
    assert [c['execution_count'] for c in code] == list(range(1, 10))
    assert not any(o['output_type'] == 'error' for c in code for o in c.get('outputs', []))
    replay = json.loads((package / 'results/PUBLIC_REPLAY.json').read_text())
    local_analysis = json.loads((root / 'results/competence_scope_analysis/analysis.json').read_text())
    public_analysis = json.loads(read(package / 'results/analysis/analysis.json.gz'))
    assert local_analysis == public_analysis
    assert len(replay['scientific_fields_compared']) == 13 and set(replay['scientific_fields_compared']) == set(public_analysis)
    assert replay['status'] == 'passed' and replay['synthetic_fixture'] is False
    assert replay['original_terminal']['route'] == execution['original_terminal_route'] == 'administrative_guard_reached'
    assert replay['source_scope_completion_sha256'] == execution['source_scope_completion_sha256']
    privacy_patterns = {
        'secret_or_private_network': re.compile(r'(sk-[A-Za-z0-9]{16,}|ghp_[A-Za-z0-9]{16,}|hf_[A-Za-z0-9]{16,}|AKIA[A-Z0-9]{12,}|PRIVATE[ ]KEY|\.ts\.net|\b(?:192\.168|10\.[0-9]+)\.[0-9]+\.[0-9]+)'),
        'private_home_path': re.compile(r'/home/(?!user\b|example\b)[A-Za-z0-9_.-]+/|/Users/[A-Za-z0-9_.-]+/'),
    }
    hits, sizes, text_count = [], [], 0
    for p in package.rglob('*'):
        if not p.is_file() or '.git' in p.parts:
            continue
        if p.stat().st_size >= 2_000_000:
            sizes.append(str(p.relative_to(package)))
        try:
            text = read(p).decode('utf-8')
        except UnicodeDecodeError:
            continue
        text_count += 1
        for label, pattern in privacy_patterns.items():
            if pattern.search(text):
                hits.append({'path': str(p.relative_to(package)), 'kind': label})
    assert not hits and not sizes, (hits, sizes)
    source_proofs = [bundle_path, support_path, nb_path, execution_path, package / 'results/PUBLIC_REPLAY.json', package / 'results/INDEPENDENT_SCOPE_REPLAY.json', package / 'results/REVIEW_REPLAY.json']
    result = {'checked_utc': datetime.now(timezone.utc).isoformat(), 'status': 'passed', 'scope': 'Read-only public artifact review. All indexed transport/decoded hashes and support hashes checked; all 8064 projected response rows compared to local originals after only token-ID removal, and event records after only argv removal. Notebook bytes/receipt bindings/sequential execution/no-error outputs inspected. Public scientific JSON exactly equals local analysis. Saved public replay was not independently rerun by this reviewer.', 'indexed_files_checked': len(bundle['files']), 'support_files_checked': len(support['files']), 'response_files_compared': 34, 'response_rows_compared': projected_responses, 'code_cells_inspected': 9, 'decoded_text_files_privacy_scanned': text_count, 'privacy_hits': hits, 'oversize_files': sizes, 'proofs_sha256': {str(p.relative_to(package)):sha(p.read_bytes()) for p in source_proofs}, 'checked_files_sha256': checked, 'script_sha256': sha(Path(__file__).read_bytes()), 'model_service_calls': False, 'public_files_mutated': False, 'editorial_navigation_review': 'Separate after final README/REPORT/LABNOTES assembly.'}
    dest = Path(__file__).parent / receipt_name
    assert not dest.exists()
    dest.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('status','indexed_files_checked','support_files_checked','response_rows_compared','code_cells_inspected','decoded_text_files_privacy_scanned','privacy_hits','oversize_files')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--receipt-name', default='PUBLIC_ARTIFACT_REVIEW.json')
    args = parser.parse_args()
    run(args.root.resolve(), args.package.resolve(), args.receipt_name)
