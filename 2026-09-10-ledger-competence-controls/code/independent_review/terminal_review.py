"""Additive, completion-gated independent audit of the competence-only terminal.

Preserves original review/factorial/phase2 bytes. Full mode reuses the original
independent factorial recomputation, including optional original-workspace
checkpoint byte contracts; it never imports or invokes a model.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.dont_write_bytecode = True
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import review
import audit_arming

GUARD_ERRORS = {
    "ValueError('Existing stage requires explicit --resume and a complete verified contract.')",
    "ValueError('Incomplete stage: preserve its entire attempt; no automatic partial reuse or retry.')",
}
SCOPE_OUTCOME = 'competence_completed_conditional_cancelled'


def stamp(value):
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError('Timestamp must include timezone')
    return parsed


def require(condition, message):
    if not condition:
        raise ValueError(message)


def safe_path(root, relative):
    relative = Path(relative)
    require(not relative.is_absolute() and '..' not in relative.parts, 'Unsafe relative artifact path')
    result = root / relative
    require(not result.is_symlink(), 'Symlinked review artifact is not accepted')
    return result


def adjudicate(expected, completed, events, amendment, armed, selection, validation, original, reference, program_completed_exists):
    """Pure independent route/chronology check; artifact hashes checked by caller."""
    require(len(expected) == 58 and len({s['name'] for s in expected}) == 58, 'Expected full 58-stage grid')
    require(len(completed) == 58, 'Partial competence/validation inventory')
    for actual, planned in zip(completed, expected):
        require(actual['name'] == planned['name'] and actual['output'] == planned['output'] and actual['plan'] == planned,
                'Completed-stage order or plan differs')
    names = [s['name'] for s in expected]
    starts = [e for e in events if e['event'] == 'stage_start']
    finishes = [e for e in events if e['event'] == 'stage_complete']
    require([e['name'] for e in starts] == names, 'Unexpected, missing, reordered or repeated model start')
    require([e['name'] for e in finishes] == names, 'Incomplete/reordered model completion events')
    for start,end in zip(starts,finishes):
        require(stamp(start['at']) <= stamp(end['at']), 'Stage finishes before it starts')
    require(all(stamp(a['at']) <= stamp(b['at']) for a,b in zip(finishes,starts[1:])), 'Model stages overlap or reorder')
    require(original['completed_stages'] == completed, 'Original terminal stage inventory differs')
    decided, armtime = stamp(amendment['decided_utc']), stamp(armed['armed_utc'])
    require(decided <= armtime, 'Arming precedes decision')
    selected_events = [e for e in events if e['event'] == 'recipe_selection_frozen']
    require(bool(selected_events), 'Missing frozen-selection event')
    first_val = next(e for e in starts if e['name'].startswith('validation_'))
    require(armtime < min(stamp(e['at']) for e in selected_events) <= stamp(first_val['at']),
            'Late amendment or validation before calibration-only selection')
    require(all(e['selected_recipe'] == selection['selected_recipe'] for e in selected_events), 'Selection event recipe differs')
    require(amendment['timing']['selected_recipe_exists'] is False and amendment['timing']['validation_outputs_exist'] is False
            and amendment['timing']['conditional_model_starts'] == 0, 'Scope decision was outcome-informed')
    before = [e for e in events if stamp(e['at']) <= decided]
    require(sum(e['event']=='stage_complete' for e in before)==amendment['timing']['completed_model_stages_at_decision'],
            'Amendment historical completion count differs')
    require([e['name'] for e in before if e['event']=='stage_start'][-1] == amendment['timing']['last_started_stage'],
            'Amendment historical active-stage claim differs')
    require(not original.get('selected_installations'), 'Conditional installation was selected')
    failures = [e for e in events if e['event']=='execution_failed']
    completions = [e for e in events if e['event']=='program_complete']
    route = reference['route']
    if route == 'administrative_guard_reached':
        require(validation['passed'] is True and selection['selected_recipe'] is not None, 'Guard route requires real numerical validation pass')
        require(not program_completed_exists, 'Guard route must not fabricate PROGRAM_COMPLETED')
        require(original.get('error') in GUARD_ERRORS, 'Unexpected operational failure cannot masquerade as review stop')
        require(len(failures)==1 and not completions, 'Unexpected execution failure/completion history')
        event = failures[0]
        require(event['error']==original['error'] and event['attempt']==reference['attempt'], 'Failure event does not match original failure')
        require(reference['event_type']=='execution_failed' and reference.get('original_error')==original['error'], 'Terminal failure binding differs')
        require(reference['path']==reference['attempt']+'/FAILED.json', 'Guard failure path differs')
        require(stamp(finishes[-1]['at']) <= stamp(original['at']) <= stamp(event['at']), 'Failure timing differs')
        numerical = 'both_selected_validation_gates_passed'
        guard_reached = True
    elif route == 'original_numerical_stop':
        require(validation['passed'] is False and program_completed_exists, 'Numerical route requires actual failed validation and original terminal')
        require(original.get('status')=='complete' and original.get('outcome')=='feasibility_gate_failed', 'Wrong original numerical terminal')
        require(original.get('reason')==validation['reason'], 'Original numerical stop reason differs from unchanged validation decision')
        require(not failures and len(completions)==1, 'Unexpected numerical-route execution history')
        event = completions[0]
        require(event['outcome']=='feasibility_gate_failed' and event['attempt']==reference['attempt'], 'Program completion event differs')
        require(reference['event_type']=='program_complete' and reference['path']=='results/PROGRAM_COMPLETED.json', 'Numerical terminal path/event differs')
        require(reference.get('original_reason')==original['reason'] and reference.get('original_outcome')==original['outcome'], 'Numerical terminal binding differs')
        require(stamp(finishes[-1]['at']) <= stamp(original['finished_utc']) <= stamp(event['at']), 'Numerical terminal timing differs')
        numerical = 'original_numerical_prerequisite_failed'
        guard_reached = False
    else:
        raise ValueError('Unknown terminal route')
    require(events[-1] == event and reference['event_index']==len(events)-1, 'Claimed terminal is not the final event')
    return dict(route=route,numerical_result=numerical,administrative_guard_reached=guard_reached,
        conditional_branch_cancelled_by_design=True,conditional_model_stages=0,
        retained_model_stages=58,training_blocks=24,calibration_stages=26,validation_stages=8,
        amendment_precedes_selection_and_validation=True,
        interpretation='Design cancellation is distinct from numerical prerequisites and is not a repair null or full-program completion.')


def compute(root):
    root = Path(root)
    terminal_path = root/'results/COMPETENCE_SCOPE_COMPLETED.json'
    require(terminal_path.is_file(), 'Competence-scope terminal pending; no final audit performed')
    terminal = json.loads(terminal_path.read_text())
    require(terminal['status']=='complete' and terminal['outcome']==SCOPE_OUTCOME, 'Wrong scope terminal kind')
    directory = root/'results/independent_review'
    history = json.loads((directory/'IMPLEMENTATION_FREEZE.json').read_text())
    for rel,digest in history['files_sha256'].items():
        require(review.sha(root/rel)==digest, 'Original independent source changed: '+rel)
    arming = audit_arming.check(root)
    program,freeze,plan = review.frozen_program(root)
    require(terminal['freeze_sha256']==review.sha(root/'results/FROZEN_PROGRAM.json') and terminal['pins']==freeze['pins']
            and terminal['source_snapshot']==freeze['source_snapshot'], 'Scope terminal frozen source differs')
    expected = [program.spec('base_calibration_'+order,'generate','data/calibration.jsonl','results/development/base_calibration_'+order,output_order=order)
                for order in ('decision_first','decision_last')]
    expected += program.factorial_plan()
    generation,_ = review.scoped_stages(program)
    expected += [generation[f'validation_{recipe}_i{seed}'] for recipe,seed in program.cells()]
    for record in terminal['completed_stages']:
        require(review.sha(safe_path(root,record['output'])/'COMPLETED.json')==record['completion_sha256'], 'Scope completed artifact changed')
    events_path=root/'results/program.jsonl'
    require(review.sha(events_path)==terminal['program_events_sha256'], 'Terminal event history changed')
    events=review.read_rows(events_path)
    def load(rel): return json.loads(safe_path(root,rel).read_text())
    amendment,armed = load('results/SCOPE_AMENDMENT.json'),load('results/SCOPE_BOUNDARY_ARMED.json')
    selection,validation = load('results/SELECTED_RECIPE.json'),load('results/VALIDATION_RESULT.json')
    bindings = terminal['scope_bindings_sha256']
    minimum={'results/SCOPE_AMENDMENT.json','results/SCOPE_BOUNDARY_ARMED.json',
        'checkpoints/induction_1729/ADMINISTRATIVE_REVIEW_STOP.json','results/scope_tools/arm_boundary.py',
        'results/SELECTED_RECIPE.json','results/VALIDATION_RESULT.json'} | set(amendment['review_artifacts_sha256'])
    require(minimum <= set(bindings), 'Incomplete scope/selection source bindings')
    for rel,digest in bindings.items(): require(review.sha(safe_path(root,rel))==digest, 'Scope-bound source changed: '+rel)
    ref=terminal['original_terminal']; original_path=safe_path(root,ref['path'])
    require(review.sha(original_path)==ref['sha256'], 'Original terminal bytes differ')
    original=json.loads(original_path.read_text())
    result=adjudicate(expected,terminal['completed_stages'],events,amendment,armed,selection,validation,original,ref,
                       (root/'results/PROGRAM_COMPLETED.json').exists())
    require(terminal['selected_recipe']==selection['selected_recipe'] and terminal['selected_installations']=={}, 'Terminal selected setup differs')
    require(terminal['conditional_cancelled'] is True and terminal['boundary_guard_reached']==result['administrative_guard_reached']
            and terminal['numerical_validation_passed']==validation['passed'] and terminal['reason']==amendment['decision']['reason'],
            'Terminal cancellation or numerical classification differs')
    require(terminal.get('synthetic_fixture') is False, 'Actual final review cannot certify synthetic fixture outcomes')
    attempt_start=load(ref['attempt']+'/STARTED.json')
    require(attempt_start['freeze_sha256']==terminal['freeze_sha256'] and attempt_start['source_snapshot']==terminal['source_snapshot'],
            'Original attempt start identity differs')
    if result['administrative_guard_reached']:
        expected_error=("ValueError('Incomplete stage: preserve its entire attempt; no automatic partial reuse or retry.')"
                        if attempt_start['resume'] else "ValueError('Existing stage requires explicit --resume and a complete verified contract.')")
        require(original['error']==expected_error and not (root/ref['attempt']/'COMPLETED.json').exists(),
                'Guard exception disagrees with attempt mode or conflicts with completion')
    else:
        require(load(ref['attempt']+'/COMPLETED.json')==original and not (root/ref['attempt']/'FAILED.json').exists(),
                'Original numerical attempt has conflicting terminal')
    for relative,digest in terminal['implementation_files_sha256'].items():
        require(review.sha(safe_path(root,relative))==digest,'Scope finalizer/analyzer source changed')
    for event in events:
        if event['event']=='recipe_selection_frozen': require(event['sha256']==review.sha(root/'results/SELECTED_RECIPE.json'), 'Selection event hash differs')
    require(validation['selected_recipe_sha256']==review.sha(root/'results/SELECTED_RECIPE.json'), 'Validation binds wrong selection')
    require(not (root/'results/logs/induction_1729.log').exists(), 'Conditional stage log exists')
    for rel in ('results/collection','results/evaluation'):
        require(not any((root/rel).glob('*')), 'Unexpected conditional output: '+rel)
    require(not (root/'results/SELECTED_INSTALLATIONS.json').exists(), 'Conditional selection exists')
    expected_checkpoints={s['output'] for s in expected if s['args']['mode']=='train'}|{'checkpoints/induction_1729'}
    require({str(p.relative_to(root)) for p in (root/'checkpoints').iterdir()}==expected_checkpoints,
            'Unexpected or partial checkpoint directory')
    import factorial
    actual=factorial.compute(root)
    factor_path=directory/'factorial_final/analysis.json'; proof_path=directory/'factorial_final/COMPLETED.json'
    require(factor_path.is_file() and proof_path.is_file(), 'Original independent factorial completion pending')
    require(json.loads(factor_path.read_text())==actual, 'Original factorial saved result differs from full recomputation')
    proof=json.loads(proof_path.read_text())
    require(proof['status']=='passed' and proof['artifacts_sha256']['analysis.json']==review.sha(factor_path), 'Independent factorial proof differs')
    require(actual['scoped_stages_recomputed']==34 and actual['case_checkpoint_records_recomputed']==8064,
            'Incomplete independent casewise coverage')
    require(actual['both_selected_validation_gates_passed']==validation['passed'] and actual['selected_recipe']==selection['selected_recipe'],
            'Independent selection/validation differs from terminal')
    return dict(status='verified_completed_competence_scope',**result,independent_factorial_recomputed=True,
        independent_scored_stages=34,independent_case_checkpoint_records=8064,
        original_workspace_weight_bytes_reverified=True,selected_recipe=selection['selected_recipe'],
        both_selected_validation_gates_passed=validation['passed'],
        source_sha256={'scope_terminal':review.sha(terminal_path),'original_terminal':review.sha(original_path),
            'original_events':review.sha(events_path),'amendment':review.sha(root/'results/SCOPE_AMENDMENT.json'),
            'arming':review.sha(root/'results/SCOPE_BOUNDARY_ARMED.json'),
            'factorial_analysis':review.sha(factor_path),'factorial_completion':review.sha(proof_path),
            'review_script':review.sha(HERE/'terminal_review.py'),'original_review':review.sha(directory/'review.py'),
            'original_factorial':review.sha(directory/'factorial.py'),'original_implementation_freeze':review.sha(directory/'IMPLEMENTATION_FREEZE.json')},
        arming_checks={k:arming[k] for k in ('marker_only_directory','conditional_stage_starts','frozen_pins_verified')},
        model_gpu_service_calls=0)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args(); root=args.root.resolve()
    result=compute(root)
    out=args.output or root/'results/independent_review/scope_review/final'
    out.mkdir(parents=True,exist_ok=False)
    (out/'REVIEW.json').write_text(json.dumps(result,indent=2)+'\n')
    (out/'COMPLETED.json').write_text(json.dumps(dict(status='passed',completed_utc=datetime.now(timezone.utc).isoformat(),
        review_sha256=review.sha(out/'REVIEW.json'),checker_sha256=review.sha(Path(__file__))),indent=2)+'\n')
    print(json.dumps({'status':result['status'],'route':result['route'],'output':str(out)}))


if __name__=='__main__': main()
