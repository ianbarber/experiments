"""Strict terminal/provenance checks for the amended competence scope; stdlib only."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re

OUTCOME = 'competence_completed_conditional_cancelled'
AMENDMENT = 'results/SCOPE_AMENDMENT.json'
ARMED = 'results/SCOPE_BOUNDARY_ARMED.json'
MARKER = 'checkpoints/induction_1729/ADMINISTRATIVE_REVIEW_STOP.json'
HELPER = 'results/scope_tools/arm_boundary.py'
EXPECTED_AMENDMENT_SHA256 = '57300ae169c07b062b7d592599674caf1977c696b138ddf90c065a420b65a4b7'
EXPECTED_FREEZE_SHA256 = '8407c0cad8cabff90fbb42d2fbb45861fd45b9df6c64af1c0f94fc5c515b8e13'
EXPECTED_BUDGET = {'base_calibrations': 2, 'competence_training_blocks': 24,
                  'competence_calibrations': 24, 'validation_stages': 8, 'total_model_stages': 58}
INITIAL_GUARD_ERROR = "ValueError('Existing stage requires explicit --resume and a complete verified contract.')"
RESUME_GUARD_ERROR = "ValueError('Incomplete stage: preserve its entire attempt; no automatic partial reuse or retry.')"


def require(value, message):
    if not value:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)


def utc(text):
    require(type(text) is str, 'Timestamp must be a string')
    value = datetime.fromisoformat(text.replace('Z', '+00:00'))
    require(value.tzinfo is not None, 'Timestamp lacks timezone')
    return value


def synthetic_identity(completion, freeze):
    synthetic = completion.get('synthetic_fixture') is True
    special = freeze['base_model_hashes'] == {
        'SYNTHETIC_NO_MODEL_WEIGHTS': hashlib.sha256(b'no weights').hexdigest()}
    require(synthetic == special, 'Synthetic fixture identity must be explicit and internally consistent')
    return synthetic


def expected_binding_paths(amendment, terminal):
    return {AMENDMENT, ARMED, MARKER, HELPER, 'results/SELECTED_RECIPE.json',
            'results/VALIDATION_RESULT.json', *amendment['review_artifacts_sha256'],
            terminal['path'], terminal['attempt'] + '/STARTED.json',
            *([terminal['attempt'] + '/COMPLETED.json'] if terminal['route'] == 'original_numerical_stop' else [])}


def validate_terminal_evidence(evidence):
    c = evidence.completion
    require(c.get('kind') == 'reviewed_competence_scope_completion' and c.get('version') == 1,
            'Unknown competence-scope terminal schema')
    require(c.get('status') == 'complete' and c.get('outcome') == OUTCOME,
            'Competence scope is not complete')
    require(c.get('conditional_cancelled') is True and c.get('selected_installations') == {},
            'Scope terminal may not claim a conditional installation')
    require(c.get('controller_lock_acquired_nonblocking') is True, 'Finalization did not acquire the controller lock')
    expected_tools = {'results/scope_analysis_tools/analyze.py', 'results/scope_analysis_tools/scope_support.py',
                      'results/scope_analysis_tools/qualitative_plan.json', 'results/scope_tools/finalize_competence.py'}
    require(set(c.get('implementation_files_sha256', {})) == expected_tools, 'Finalization utility identity inventory differs')
    for name, digest in c['implementation_files_sha256'].items():
        evidence.verify(name, digest)
    require(evidence.original_hash('results/scope_analysis_tools/analyze.py') == hashlib.sha256((Path(__file__).parent / 'analyze.py').read_bytes()).hexdigest()
            and c['scope_support_sha256'] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'Executing scope analysis utilities differ from terminal-bound source')
    require(c['analyzer_sha256'] == c['implementation_files_sha256']['results/scope_analysis_tools/analyze.py']
            and c['scope_support_sha256'] == c['implementation_files_sha256']['results/scope_analysis_tools/scope_support.py']
            and c['finalizer_sha256'] == c['implementation_files_sha256']['results/scope_tools/finalize_competence.py'],
            'Finalization utility provenance disagrees')
    require(len(c['completed_stages']) == 58, 'All 58 retained model stages are required')
    freeze = evidence.json('results/FROZEN_PROGRAM.json', c['freeze_sha256'])
    synthetic = synthetic_identity(c, freeze)
    if not synthetic:
        require(c['freeze_sha256'] == EXPECTED_FREEZE_SHA256, 'Unexpected actual scientific freeze')
    tool_freeze = evidence.json('results/scope_analysis_tools/IMPLEMENTATION_FREEZE.json', c['implementation_freeze_sha256'])
    require(tool_freeze['status'] == 'frozen' and tool_freeze['kind'] == 'additive_competence_scope_analysis',
            'Scope implementation was not frozen')
    require(expected_tools <= set(tool_freeze['files_sha256']), 'Scope implementation freeze is incomplete')
    require(all(name.startswith('results/scope_analysis_tools/') or name == 'results/scope_tools/finalize_competence.py'
                for name in tool_freeze['files_sha256']), 'Unexpected scope implementation file')
    for name, digest in tool_freeze['files_sha256'].items():
        evidence.verify(name, digest)
    require(all(tool_freeze['files_sha256'][name] == digest for name, digest in c['implementation_files_sha256'].items()),
            'Finalization implementation differs from prevalidation freeze')

    amendment = evidence.json(AMENDMENT)
    if not synthetic:
        require(evidence.hashes[AMENDMENT] == EXPECTED_AMENDMENT_SHA256, 'Unexpected actual scope amendment')
    require(amendment['original_frozen_program_sha256'] == c['freeze_sha256']
            and amendment['original_protocol_sha256'] == c['pins']['PROTOCOL.md'],
            'Amendment binds a different scientific freeze/protocol')
    require(amendment.get('kind') == 'prospective_conditional_branch_cancellation_after_design_review'
            and amendment.get('scientific_source_data_config_unchanged') is True,
            'Wrong scope amendment')
    require(canonical(amendment['retained_budget']) == canonical(EXPECTED_BUDGET), 'Amended stage budget differs')
    require(amendment['boundary']['output_directory'] == 'checkpoints/induction_1729'
            and amendment['boundary']['marker'] == MARKER
            and amendment['boundary']['checkpoint_or_model_output'] is False,
            'Wrong administrative boundary')
    timing = amendment['timing']
    require(timing['selected_recipe_exists'] is False and timing['validation_outputs_exist'] is False
            and type(timing['conditional_model_starts']) is int and timing['conditional_model_starts'] == 0
            and timing['no_new_conditional_outcome_informed_decision'] is True,
            'Amendment was not declared before conditional/validation outcomes')
    receipt = evidence.json(ARMED)
    marker = evidence.json(MARKER)
    require(receipt['scope_amendment_sha256'] == evidence.hashes[AMENDMENT]
            and receipt['marker_sha256'] == evidence.hashes[MARKER], 'Arming receipt hash mismatch')
    require(receipt.get('marker_only_directory') is True
            and receipt.get('no_conditional_stage_has_started') is True
            and receipt.get('all_26_scientific_pins_and_archived_sources_unchanged') is True
            and type(receipt.get('model_or_service_calls')) is int and receipt['model_or_service_calls'] == 0,
            'Arming receipt did not establish the required boundary')
    require(marker.get('kind') == 'administrative_boundary_marker_not_a_checkpoint_or_model_output'
            and marker.get('contains_model_outputs') is False
            and marker.get('scope_amendment') == AMENDMENT
            and marker.get('scope_amendment_sha256') == evidence.hashes[AMENDMENT], 'Wrong boundary marker')
    require(set((evidence.root / 'checkpoints/induction_1729').iterdir()) == {evidence.root / MARKER},
            'Administrative marker directory contains additional artifacts')
    require(utc(marker['created_utc']) == utc(amendment['decided_utc'])
            and utc(receipt['armed_utc']) >= utc(amendment['decided_utc']), 'Amendment/marker chronology differs')
    evidence.verify(HELPER, amendment['helper_sha256'])
    require(isinstance(amendment['review_artifacts_sha256'], dict)
            and len(amendment['review_artifacts_sha256']) == 4, 'Wrong review binding inventory')
    for relative, digest in amendment['review_artifacts_sha256'].items():
        evidence.verify(relative, digest)
    terminal = c['original_terminal']
    require(terminal['route'] in ('original_numerical_stop', 'administrative_guard_reached'), 'Unknown original terminal route')
    require(re.fullmatch(r'results/program_attempts/[^/]+', terminal['attempt']) is not None,
            'Unsafe original attempt path')
    bindings = c['scope_bindings_sha256']
    require(set(bindings) == expected_binding_paths(amendment, terminal), 'Scope binding inventory differs')
    for relative, digest in bindings.items():
        evidence.verify(relative, digest)
    require(bindings[terminal['path']] == terminal['sha256'], 'Original terminal identity differs')
    actual = evidence.json(terminal['path'], terminal['sha256'])
    started = evidence.json(terminal['attempt'] + '/STARTED.json')
    require(started['freeze_sha256'] == c['freeze_sha256'] and started['source_snapshot'] == c['source_snapshot']
            and type(started['resume']) is bool, 'Original attempt starting record differs')
    events = evidence.all_events
    require(events and all(isinstance(e, dict) and type(e.get('event')) is str for e in events), 'Invalid program event stream')
    evidence.verify('results/program.jsonl', c['program_events_sha256'])
    starts = [e for e in events if e['event'] == 'program_start']
    require(len(starts) == 1 and starts[0]['source_snapshot'] == c['source_snapshot'], 'Original program-start identity differs')
    allowed_names = set(evidence.stages)
    model_events = [e for e in events if e['event'] in ('stage_start', 'stage_complete', 'stage_reused')]
    require(all(e['name'] in allowed_names for e in model_events), 'Conditional or unrecorded model invocation')
    require(not any(e['event'] in ('induction_gate', 'cohort_gate_passed', 'all_prerepair_gates_passed') for e in events),
            'Conditional progress was recorded despite scope cancellation')
    validation_starts = [e for e in events if e['event'] == 'stage_start' and e['name'].startswith('validation_')]
    selection_events = [e for e in events if e['event'] == 'recipe_selection_frozen']
    require(validation_starts and selection_events, 'All validation starts and original selection are required')
    require(all(utc(e['at']) > utc(receipt['armed_utc']) for e in validation_starts + selection_events),
            'Arming did not precede selection and validation')
    if not synthetic:
        require(all(utc(e['at']) > utc(tool_freeze['created_utc']) for e in validation_starts + selection_events),
                'Scope implementation was not frozen before selection and validation')

    before = [e for e in events if utc(e['at']) <= utc(amendment['decided_utc'])]
    before_starts = [e for e in before if e['event'] == 'stage_start']
    require(sum(e['event'] == 'stage_complete' for e in before) == timing['completed_model_stages_at_decision']
            and before_starts and before_starts[-1]['name'] == timing['last_started_stage'],
            'Amendment development-stage timing does not match original events')
    ending = events[-1]
    require(ending['attempt'] == terminal['attempt'] and ending['event'] == terminal['event_type'],
            'Latest controller event is not the claimed original terminal')
    require(terminal['event_index'] == len(events) - 1, 'Original terminal event index differs')
    evidence.events = [e for e in events if e['attempt'] == terminal['attempt']]
    require([e['name'] for e in evidence.events if e['event'] in ('stage_complete', 'stage_reused')] == list(evidence.stages),
            'Final controller attempt did not complete/reuse all 58 stages in order')
    require(canonical(actual['completed_stages']) == canonical(c['completed_stages']), 'Original terminal completed-stage list differs')
    require(actual.get('selected_installations') == {}, 'Original terminal contains conditional installations')
    require(c['reason'] == amendment['decision']['reason'], 'Cancellation reason changed from amendment')
    require(c['numerical_validation_passed'] is evidence.json('results/VALIDATION_RESULT.json')['passed'],
            'Terminal changed the original numerical validation result')
    if terminal['route'] == 'original_numerical_stop':
        require(terminal['path'] == 'results/PROGRAM_COMPLETED.json'
                and terminal['event_type'] == 'program_complete'
                and actual.get('status') == 'complete' and actual.get('outcome') == 'feasibility_gate_failed'
                and ending.get('outcome') == actual['outcome'], 'Not an original numerical gate stop')
        require(actual['freeze_sha256'] == c['freeze_sha256'] and actual['pins'] == c['pins']
                and actual['source_snapshot'] == c['source_snapshot'] and actual['selected_recipe'] == c['selected_recipe'],
                'Original numerical completion identities differ')
        attempt_complete = evidence.json(terminal['attempt'] + '/COMPLETED.json')
        require(canonical(attempt_complete) == canonical(actual), 'Original attempt/program completions differ')
        require(not (evidence.root / terminal['attempt'] / 'FAILED.json').exists(), 'Final attempt has conflicting terminal records')
        require(terminal['original_outcome'] == actual['outcome'] and terminal['original_reason'] == actual['reason']
                and terminal['original_error'] is None and c['boundary_guard_reached'] is False,
                'Natural numerical stop was recoded or guard falsely marked reached')
        require(actual.get('heldout_repair_results_used_for_selection') is False,
                'Original numerical stop lacks no-repair-selection declaration')
    else:
        require(terminal['path'] == terminal['attempt'] + '/FAILED.json'
                and terminal['event_type'] == 'execution_failed', 'Wrong guarded original failure record')
        require(not (evidence.root / 'results/PROGRAM_COMPLETED.json').exists()
                and not (evidence.root / terminal['attempt'] / 'COMPLETED.json').exists(),
                'Guarded stop must not fabricate original completion records')
        expected_error = RESUME_GUARD_ERROR if started['resume'] else INITIAL_GUARD_ERROR
        require(actual['error'] == expected_error and ending.get('error') == expected_error
                and terminal['original_error'] == expected_error, 'Unexpected controller exception; not the reviewed guard')
        require(terminal['original_reason'] is None and terminal['original_outcome'] is None
                and c['boundary_guard_reached'] is True, 'Guard exception recoded as an original numerical outcome')
    weight_verification = c['adapter_verification']
    require(weight_verification['method'] == 'sha256_of_existing_adapter_bytes_at_scope_finalization'
            and weight_verification['files_sha256'] == evidence.recorded_weights,
            'Finalization adapter identity inventory differs from stage records')
    require(evidence.in_memory_finalization or 'verification' in c, 'Saved scope terminal lacks its completed verification')
    if 'verification' in c:
        verified = c['verification']
        require(verified['training_blocks'] == 24 and verified['generation_stages'] == 34
                and verified['reparsed_responses'] == 8064 and verified['adapter_files_rehashed'] == 24
                and verified['model_or_generation_calls'] == 0
                and verified['original_program_completion_fabricated'] is False
                and verified['original_selection_and_validation_preserved'] is True,
                'Scope verification summary is inconsistent')
        for name, digest in verified['checked_input_sha256'].items():
            evidence.verify(name, digest)


def validate_numerical_path(evidence, selection, decision, stop):
    c = evidence.completion
    require(type(decision['passed']) is bool and c['numerical_validation_passed'] is decision['passed'],
            'Scope terminal changes recomputed numerical prerequisites')
    if c['original_terminal']['route'] == 'original_numerical_stop':
        require(stop is not None and c['original_terminal']['original_reason'] == stop['reason'],
                'Original numerical stop reason disagrees with frozen gates')
    else:
        require(stop is None and decision['passed'] is True and selection['selected_recipe'] is not None,
                'Administrative guard is only reached after the original validation gate passes')


def validate_model_inventory(evidence):
    expected = {s['output'] for s in evidence.stages.values()}
    actual = {str(p.parent.relative_to(evidence.root))
              for directory in ('checkpoints', 'results/development', 'results/validation', 'results/collection', 'results/evaluation')
              for p in (evidence.root / directory).rglob('COMPLETED.json')}
    require(actual == expected, 'Unrecorded or missing model completion')
    for directory in ('checkpoints', 'results/development', 'results/validation'):
        base = evidence.root / directory
        allowed = {evidence.root / name for name in expected if name.startswith(directory + '/')}
        if directory == 'checkpoints':
            allowed.add(evidence.root / 'checkpoints/induction_1729')
        require(set(base.iterdir()) == allowed, 'Unrecorded/partial model directory: ' + directory)
    for directory in ('results/collection', 'results/evaluation'):
        base = evidence.root / directory
        require(not base.exists() or not any(base.iterdir()), 'Conditional output directory is not empty')
    require(not any((evidence.root / 'data').glob('cohort_*')), 'Conditional cohort artifact exists')
    for path in (evidence.root / 'results/logs').glob('*'):
        require(not re.match(r'(induction_|failures_|evaluation_|train_).*', path.name),
                'Conditional model log exists: ' + path.name)
