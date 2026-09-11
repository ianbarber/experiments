"""Record the reviewed scope reduction and arm a fail-closed, non-checkpoint marker.

This additive operational helper never edits the frozen program or calls a model.
Its existing-output guard is independently tested under the original controller.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json


ROOT = Path(__file__).resolve().parents[2]
EXPECTED_FREEZE = '8407c0cad8cabff90fbb42d2fbb45861fd45b9df6c64af1c0f94fc5c515b8e13'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def verify_pins(freeze):
    for relative, expected in freeze['pins'].items():
        assert digest(ROOT / relative) == expected, relative
        if not relative.startswith('data/'):
            assert digest(ROOT / freeze['source_snapshot'] / relative) == expected, relative


def main():
    frozen = ROOT / 'results/FROZEN_PROGRAM.json'
    assert digest(frozen) == EXPECTED_FREEZE
    freeze = json.loads(frozen.read_text())
    verify_pins(freeze)
    marker_dir = ROOT / 'checkpoints/induction_1729'
    assert not marker_dir.exists(), 'Never overwrite a checkpoint or an earlier administrative marker.'
    assert not (ROOT / 'results/SELECTED_RECIPE.json').exists(), 'This decision must precede recipe selection.'
    assert not (ROOT / 'results/VALIDATION_RESULT.json').exists()
    assert not any((ROOT / 'results/validation').glob('*'))
    events = [json.loads(line) for line in (ROOT / 'results/program.jsonl').read_text().splitlines()]
    starts = [event['name'] for event in events if event['event'] == 'stage_start']
    assert all(name.startswith(('base_calibration_', 'competence_')) for name in starts)
    reviews = [
        'results/early_validity_review/design/REVIEW_RECORD.json',
        'results/early_validity_review/design/BOUNDARY_GUARD_CPU_CHECK.json',
        'results/early_validity_review/data/PRACTICAL_ADJUDICATION.json',
        'results/early_validity_review/execution/BOUNDARY_GUARD_CHECKS.json',
    ]
    test = json.loads((ROOT / reviews[-1]).read_text())
    assert test['status'] == 'passed' and len(test['cases']) == 8
    assert all(case['blocked'] and case['subprocess_calls'] == 0 and case['stage_start_events'] == 0
               for case in test['cases'])
    assert test['frozen_program_sha256'] == freeze['pins']['scripts/program.py']
    now = datetime.now(timezone.utc).isoformat()
    amendment_path = ROOT / 'results/SCOPE_AMENDMENT.json'
    amendment = {
        'version': 1,
        'kind': 'prospective_conditional_branch_cancellation_after_design_review',
        'decided_utc': now,
        'authorization': 'User requested a thorough subagent validity review and early failure for invalidating design issues; research lead adopted and communicated the practical branch-stop recommendation.',
        'original_frozen_program_sha256': EXPECTED_FREEZE,
        'original_protocol_sha256': freeze['pins']['PROTOCOL.md'],
        'scientific_source_data_config_unchanged': True,
        'review_artifacts_sha256': {relative: digest(ROOT / relative) for relative in reviews},
        'decision': {
            'competence': 'Complete the original factorial, calibration-only shared selection and all eight validation stages unchanged.',
            'conditional_induction_collection_repair': 'Cancel before any conditional model stage regardless of whether numerical prerequisites would pass.',
            'reason': 'Every failure correction has a REPORT-only wrapper; ordinary preservation has zero trigger-REPORT cases and 64 trigger-CLEAR cases. This enables presentation-specific correction instead of ordinary-prompt transfer. Restricted donor traces and bundled whole-case relevance further limit information about the intended repair question.',
            'classification': 'Design-adequacy scope reduction, not a failed original numerical gate, corrupted competence comparison, measured repair null, or equivalence result.',
            'future_repair': 'Any redesigned repair study must have a separate protocol and freeze; do not alter or restart this conditional branch.',
        },
        'timing': {
            'competence_development_outputs_already_observed': True,
            'selected_recipe_exists': False,
            'validation_outputs_exist': False,
            'conditional_model_starts': 0,
            'completed_model_stages_at_decision': sum(event['event'] == 'stage_complete' for event in events),
            'last_started_stage': starts[-1],
            'no_new_conditional_outcome_informed_decision': True,
        },
        'retained_budget': {'base_calibrations': 2, 'competence_training_blocks': 24,
                            'competence_calibrations': 24, 'validation_stages': 8,
                            'total_model_stages': 58},
        'boundary': {
            'output_directory': 'checkpoints/induction_1729',
            'marker': 'checkpoints/induction_1729/ADMINISTRATIVE_REVIEW_STOP.json',
            'mechanism': 'Original StageRunner.stage rejects an existing output before stage_start or subprocess. Missing COMPLETED.json also prevents reuse under --resume.',
            'checkpoint_or_model_output': False,
            'terminal_interpretation': 'If reached, preserve the original generic attempt FAILED.json and execution_failed event as evidence of the deliberate guard. Produce a separately named reviewed competence-scope terminal record; never fabricate PROGRAM_COMPLETED.json. If original numerical validation fails first, preserve its genuine PROGRAM_COMPLETED.json and report the guard unvisited.',
        },
        'helper_sha256': digest(Path(__file__)),
    }
    write_new(amendment_path, amendment)
    marker_dir.mkdir(exist_ok=False)
    marker_path = marker_dir / 'ADMINISTRATIVE_REVIEW_STOP.json'
    write_new(marker_path, {
        'kind': 'administrative_boundary_marker_not_a_checkpoint_or_model_output',
        'created_utc': now,
        'scope_amendment': 'results/SCOPE_AMENDMENT.json',
        'scope_amendment_sha256': digest(amendment_path),
        'message': 'Intentional design-review stop before induction. Do not delete, resume, replace with a completion, or treat this directory as a model attempt. Preserve all completed competence/validation evidence and let the original wrapper restore service.',
        'contains_model_outputs': False,
    })
    assert list(marker_dir.iterdir()) == [marker_path]
    verify_pins(freeze)
    receipt = {
        'armed_utc': datetime.now(timezone.utc).isoformat(),
        'scope_amendment_sha256': digest(amendment_path),
        'marker_sha256': digest(marker_path),
        'marker_only_directory': True,
        'all_26_scientific_pins_and_archived_sources_unchanged': True,
        'model_or_service_calls': 0,
        'no_conditional_stage_has_started': True,
        'independent_guard_checks_passed': 8,
    }
    write_new(ROOT / 'results/SCOPE_BOUNDARY_ARMED.json', receipt)
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
