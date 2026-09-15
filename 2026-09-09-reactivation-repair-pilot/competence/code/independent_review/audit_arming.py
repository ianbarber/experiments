"""Read-only CPU audit of actual administrative arming; no terminal inference."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(root):
    root = Path(root)
    def read(rel): return json.loads((root / rel).read_text())
    freeze = read('results/FROZEN_PROGRAM.json')
    amendment = read('results/SCOPE_AMENDMENT.json')
    armed = read('results/SCOPE_BOUNDARY_ARMED.json')
    marker_rel = 'checkpoints/induction_1729/ADMINISTRATIVE_REVIEW_STOP.json'
    marker = read(marker_rel)
    if amendment['original_frozen_program_sha256'] != sha(root / 'results/FROZEN_PROGRAM.json'):
        raise ValueError('Amendment binds different program')
    if amendment['original_protocol_sha256'] != freeze['pins']['PROTOCOL.md']:
        raise ValueError('Amendment binds different protocol')
    for rel, digest in freeze['pins'].items():
        if sha(root / rel) != digest:
            raise ValueError('Frozen input changed: ' + rel)
        if not rel.startswith('data/') and sha(root / freeze['source_snapshot'] / rel) != digest:
            raise ValueError('Archived source changed: ' + rel)
    if read(freeze['source_snapshot'] + '/manifest.json') != freeze:
        raise ValueError('Snapshot manifest differs')
    bound = {'results/SCOPE_AMENDMENT.json': sha(root/'results/SCOPE_AMENDMENT.json')}
    if marker['scope_amendment_sha256'] != bound['results/SCOPE_AMENDMENT.json'] or armed['scope_amendment_sha256'] != bound['results/SCOPE_AMENDMENT.json']:
        raise ValueError('Marker/arming amendment hash differs')
    if armed['marker_sha256'] != sha(root / marker_rel):
        raise ValueError('Marker changed')
    if marker['kind'] != 'administrative_boundary_marker_not_a_checkpoint_or_model_output' or marker['contains_model_outputs'] is not False:
        raise ValueError('Marker is not explicit administrative non-output')
    if sorted(p.name for p in (root / marker_rel).parent.iterdir()) != [Path(marker_rel).name]:
        raise ValueError('Marker directory contains other artifacts')
    if sha(root / 'results/scope_tools/arm_boundary.py') != amendment['helper_sha256']:
        raise ValueError('Arming helper changed')
    for rel,digest in amendment['review_artifacts_sha256'].items():
        if sha(root/rel) != digest: raise ValueError('Bound review changed: '+rel)
        bound[rel] = digest
    events = [json.loads(line) for line in (root/'results/program.jsonl').read_text().splitlines(keepends=True)
              if line.endswith('\n') and line.strip()]
    decided = datetime.fromisoformat(amendment['decided_utc'])
    armtime = datetime.fromisoformat(armed['armed_utc'])
    if datetime.fromisoformat(marker['created_utc']) != decided or armtime < decided:
        raise ValueError('Marker/arming chronology differs')
    prefix = [e for e in events if datetime.fromisoformat(e['at']) <= decided]
    starts = [e for e in events if e['event']=='stage_start']
    if any(not e['name'].startswith(('base_calibration_', 'competence_', 'validation_')) for e in starts):
        raise ValueError('Conditional stage actually started')
    early_forbidden = [e for e in events if e['event']=='recipe_selection_frozen' or
                       (e['event']=='stage_start' and e['name'].startswith('validation_'))]
    if any(datetime.fromisoformat(e['at']) <= armtime for e in early_forbidden):
        raise ValueError('Amendment did not precede selection/validation')
    expected_complete = sum(e['event']=='stage_complete' for e in prefix)
    expected_last = [e['name'] for e in prefix if e['event']=='stage_start'][-1]
    timing = amendment['timing']
    if timing['completed_model_stages_at_decision'] != expected_complete or timing['last_started_stage'] != expected_last:
        raise ValueError('Arming historical stage inventory differs')
    for rel in ['results/SCOPE_BOUNDARY_ARMED.json', marker_rel, 'results/scope_tools/arm_boundary.py', 'results/FROZEN_PROGRAM.json']:
        bound[rel] = sha(root/rel)
    return dict(status='verified_actual_arming',decided_utc=amendment['decided_utc'],armed_utc=armed['armed_utc'],
        captured_utc=datetime.now(timezone.utc).isoformat(),frozen_pins_verified=len(freeze['pins']),
        marker_only_directory=True,conditional_stage_starts=0,completed_stages_at_decision=expected_complete,
        last_started_at_decision=expected_last,amendment_precedes_selection_and_validation=True,
        later_terminal_not_inferred=True,model_gpu_service_calls=0,files_sha256=bound)


if __name__ == '__main__':
    result = check(ROOT)
    result['checker_sha256'] = sha(Path(__file__))
    with (Path(__file__).parent/'ACTUAL_ARMING_AUDIT.json').open('x') as stream:
        json.dump(result,stream,indent=2); stream.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='files_sha256'},indent=2))
