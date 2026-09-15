#!/usr/bin/env python3
"""Review a terminal controller and finalize competence scope; never run models.

Default is a read-only terminal check. --write creates only the separately named
COMPETENCE_SCOPE_COMPLETED.json after full verification. Existing files are refused.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


TERMINAL = 'results/COMPETENCE_SCOPE_COMPLETED.json'


def load_analyzer(root):
    path = root / 'results/scope_analysis_tools/analyze.py'
    spec = importlib.util.spec_from_file_location('_scope_finalization_analysis', path)
    module = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


def finalize(root, write=False):
    root = Path(root).resolve()
    analysis = load_analyzer(root)
    require, read, sha, safe = analysis.require, analysis.read_json, analysis.sha, analysis.safe_path
    require(not (root / TERMINAL).exists(), 'Competence-scope terminal already exists; never overwrite it')
    lock_path = root / 'results/program.lock'
    require(lock_path.is_file(), 'Original controller lock file is required')
    with lock_path.open('r') as controller_lock:
        try:
            fcntl.flock(controller_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError('Original controller still holds its execution lock; finalization is premature') from exc
        freeze = read(root / 'results/FROZEN_PROGRAM.json')
        events = analysis.read_rows(root / 'results/program.jsonl')
        require(events, 'Original controller event stream is empty')
        last = events[-1]
        attempt = last.get('attempt')
        require(type(attempt) is str, 'Original terminal lacks its controller attempt')
        original_complete = root / 'results/PROGRAM_COMPLETED.json'
        if original_complete.is_file():
            require(last['event'] == 'program_complete', 'Original completion exists without final program_complete event')
            original_path = 'results/PROGRAM_COMPLETED.json'
            original = read(original_complete)
            require(original.get('outcome') == 'feasibility_gate_failed', 'Unexpected original completed program outcome')
            terminal = {'route': 'original_numerical_stop', 'path': original_path, 'event_type': 'program_complete',
                        'original_outcome': original['outcome'], 'original_reason': original['reason'], 'original_error': None}
        else:
            require(last['event'] == 'execution_failed', 'Controller has not reached a recognized terminal event')
            original_path = attempt + '/FAILED.json'
            original = read(safe(root, original_path))
            terminal = {'route': 'administrative_guard_reached', 'path': original_path, 'event_type': 'execution_failed',
                        'original_outcome': None, 'original_reason': None, 'original_error': original['error']}
        terminal.update(sha256=sha(safe(root, original_path)), attempt=attempt, event_index=len(events)-1)
        records = original.get('completed_stages')
        require(type(records) is list and len(records) == 58, 'All 58 retained stages must finish before finalization')
        require(len({r['name'] for r in records}) == 58, 'Duplicate retained stage record')
        amendment = read(root / analysis.scope_support.AMENDMENT)
        selection = read(root / 'results/SELECTED_RECIPE.json')
        validation = read(root / 'results/VALIDATION_RESULT.json')
        require(type(validation.get('passed')) is bool, 'Original validation result is not a boolean')
        bindings = {name: sha(safe(root, name)) for name in analysis.scope_support.expected_binding_paths(amendment, terminal)}
        # Rehash retained adapter bytes locally. Public CPU replay later preserves
        # these identities but explicitly does not recreate or require weights.
        adapters = {}
        for record in records:
            path = record['output'] + '/COMPLETED.json'
            require(sha(safe(root, path)) == record['completion_sha256'], 'Original stage completion changed')
            contract = read(safe(root, path))
            for relative, digest in contract['artifacts_sha256'].items():
                if relative.endswith('/adapter_model.safetensors'):
                    full = record['output'] + '/' + relative
                    require(sha(safe(root, full)) == digest, 'Retained adapter bytes differ from original contract')
                    adapters[full] = digest
        actual_adapters = {str(p.relative_to(root)) for p in (root/'checkpoints').rglob('adapter_model.safetensors')}
        require(actual_adapters == set(adapters), 'Unrecorded or missing retained adapter file')
        synthetic = freeze['base_model_hashes'] == {
            'SYNTHETIC_NO_MODEL_WEIGHTS': hashlib.sha256(b'no weights').hexdigest()}
        candidate = {'version': 1, 'kind': 'reviewed_competence_scope_completion', 'status': 'complete',
            'outcome': analysis.scope_support.OUTCOME, 'reason': amendment['decision']['reason'],
            'finished_utc': datetime.now(timezone.utc).isoformat(), 'synthetic_fixture': synthetic,
            'conditional_cancelled': True, 'boundary_guard_reached': terminal['route'] == 'administrative_guard_reached',
            'numerical_validation_passed': validation['passed'],
            'source_snapshot': freeze['source_snapshot'], 'freeze_sha256': sha(root / 'results/FROZEN_PROGRAM.json'),
            'pins': freeze['pins'], 'completed_stages': records, 'selected_recipe': selection['selected_recipe'],
            'selected_installations': {}, 'original_terminal': terminal, 'scope_bindings_sha256': bindings,
            'program_events_sha256': sha(root / 'results/program.jsonl'),
            'adapter_verification': {'method': 'sha256_of_existing_adapter_bytes_at_scope_finalization',
                                     'files_sha256': adapters, 'synthetic_fixture_bytes': synthetic},
            'classification': 'Intentional conditional design-scope cancellation; preserve any genuine original numerical stop separately. No repair null or equivalence estimate.',
            'finalizer_sha256': sha(Path(__file__)),
            'analyzer_sha256': sha(root / 'results/scope_analysis_tools/analyze.py'),
            'scope_support_sha256': sha(root / 'results/scope_analysis_tools/scope_support.py'),
            'implementation_files_sha256': {name: sha(root/name) for name in (
                'results/scope_analysis_tools/analyze.py', 'results/scope_analysis_tools/scope_support.py',
                'results/scope_analysis_tools/qualitative_plan.json', 'results/scope_tools/finalize_competence.py')},
            'implementation_freeze_sha256': sha(root/'results/scope_analysis_tools/IMPLEMENTATION_FREEZE.json'),
            'controller_lock_acquired_nonblocking': True}
        evidence, checked = analysis.verify_competence_scope(root, scope_completion=candidate)
        require(not evidence.live_source_differences, 'Live scientific sources changed before scope finalization')
        require(checked['training_blocks'] == 24 and checked['generation_stages'] == 34
                and checked['reparsed_responses'] == 8064, 'Retained competence response inventory differs')
        candidate['verification'] = {'training_blocks': 24, 'generation_stages': 34,
            'reparsed_responses': 8064, 'data_checks': checked['data_checks'],
            'checked_input_sha256': evidence.hashes, 'adapter_files_rehashed': len(adapters),
            'model_or_generation_calls': 0, 'original_program_completion_fabricated': False,
            'original_selection_and_validation_preserved': True,
            'live_scientific_sources_unchanged_at_finalization': True}
        evidence.recheck()
        for name, digest in adapters.items():
            require(sha(safe(root, name)) == digest, 'Adapter drift during scope finalization')
        if write:
            with (root / TERMINAL).open('x') as stream:
                json.dump(candidate, stream, indent=2, ensure_ascii=False, allow_nan=False)
                stream.write('\n')
        return candidate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--write', action='store_true', help='Create the distinct terminal after successful verification')
    args = parser.parse_args()
    result = finalize(args.root, write=args.write)
    print(json.dumps({'status': 'complete' if args.write else 'verified_without_writing',
        'outcome': result['outcome'], 'original_terminal_route': result['original_terminal']['route'],
        'numerical_validation_passed': result['numerical_validation_passed'],
        'retained_stages': len(result['completed_stages']), 'verification': result['verification']}, indent=2))


if __name__ == '__main__':
    main()
