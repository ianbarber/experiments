"""Bounded CPU audit of the 48-stage initial grid; exclude later work."""
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import independent_run_audit as driver


def main():
    began = time.monotonic()
    root = driver.ROOT
    destination = root / 'results/analysis_tools'
    audit = driver.RunAudit(inspect_optimizer=True)
    expected = {f's{seed}_{kind}' for seed in driver.SEEDS
                for kind in ('competence_screen', 'baseline_present', 'baseline_omitted_loss')}
    for recipe, _, _ in driver.RECIPES:
        expected.update(f's1729_{recipe}_epoch{epoch}' for epoch in (1, 2, 3, 4))
        expected.update(f's1729_{recipe}_diagnose{dose}' for dose in (1, 2, 4))
    assert len(expected) == 48 and expected.issubset(audit.stages)
    audit.stages = {k: v for k, v in audit.stages.items() if k in expected}
    audit.stage_hashes = {k: v for k, v in audit.stage_hashes.items() if k in expected}
    audit.sidecars = {'INITIAL_GRID': audit.sidecars['INITIAL_GRID']}
    audit.sidecar_hashes = {'INITIAL_GRID': audit.sidecar_hashes['INITIAL_GRID']}
    expected_gates = {f's{seed}_competence_screen' for seed in driver.SEEDS}
    expected_gates.update(f's1729_{recipe}_diagnose{dose}_{view}'
                          for recipe, _, _ in driver.RECIPES for dose in (1, 2, 4)
                          for view in driver.VIEWS[2:])
    assert len(expected_gates) == 38 and expected_gates.issubset(audit.gate_files)
    audit.gate_files = {k: v for k, v in audit.gate_files.items() if k in expected_gates}
    audit.gate_hashes = {k: v for k, v in audit.gate_hashes.items() if k in expected_gates}
    assert len(audit.sidecars['INITIAL_GRID']) == 18
    source_paths = [Path(__file__), Path(driver.__file__), root / 'scripts/independent_review.py',
                    root / 'scripts/program.py', root / 'results/INITIAL_GRID.json']
    source_paths.extend(root / 'results/stages' / name / 'candidate.json'
                        for name in expected if '_diagnose' in name)
    source_hashes = {str(p.relative_to(root)): driver.digest(p) for p in source_paths}
    snapshot = dict(at=datetime.now(timezone.utc).isoformat(),
                    scope='Exactly six baseline stages and the 24 training passes plus 18 diagnoses of the initial seed grid. Control, replication, fresh evaluation and terminal work excluded.',
                    stage_completion_sha256=audit.stage_hashes,
                    gate_sha256=audit.gate_hashes, source_sha256=source_hashes)
    snapshot_path = destination / 'INITIAL_GRID_EXECUTION_SNAPSHOT.json'
    with snapshot_path.open('x') as stream:
        json.dump(snapshot, stream, indent=2); stream.write('\n')
    result = audit.run()
    assert not result['errors'], result['errors']
    assert not result['pending_controller_sidecars'], result['pending_controller_sidecars']
    assert len(audit.candidates) == 18 and len(audit.expected_gates) == 38
    inventory = set()
    for stored in audit.sidecars['INITIAL_GRID']:
        key = (stored['seed'], stored['recipe']['index'], stored['epoch'])
        assert key not in inventory
        inventory.add(key)
        candidate = audit.candidate_from_stored(stored)
        assert stored == driver.read(root / 'results/stages' / candidate['diagnosis'] / 'candidate.json')
    initial = list(audit.candidates.values())
    assert not any(c['development']['pass_'] for c in initial)
    best = [min((c for c in initial if c['recipe_index'] == index), key=driver.independent.candidate_rank)
            for index in range(6)]
    ranked = sorted(best, key=driver.independent.candidate_rank)
    assert [list((c['recipe_index'], c['epoch'])) for c in ranked[:3]] == [list(x) for x in result['selection']['initial_grid']['identities']]
    table = []
    for c in sorted(initial, key=lambda c: (c['recipe_index'], c['epoch'])):
        row = dict(recipe=driver.RECIPES[c['recipe_index']][0], dose=c['epoch'],
                   passes_both_views=c['development']['pass_'], rank=list(driver.independent.candidate_rank(c)), views={})
        for view, gate in c['development']['views'].items():
            counts = gate['installed_counts']
            row['views'][view] = dict(pass_=gate['pass_'], eliciting_clear=counts['eliciting_report']['valid_clear'],
                                      ordinary_report_correct=counts['noneliciting_report']['correct'],
                                      legitimate_clear_correct=counts['clear']['correct'], valid=counts['all']['valid'],
                                      deficits=gate['deficits'])
        table.append(row)
    summary = lambda c: dict(recipe=driver.RECIPES[c['recipe_index']][0], dose=c['epoch'],
                             rank=list(driver.independent.candidate_rank(c)))
    trainings = [v['training'] for v in result['audited_stages'].values() if 'training' in v]
    assert len(trainings) == 24 and all(t['optimizer']['cpu_tensor_inspection'] for t in trainings)
    assert all(t['optimizer']['parameter_tensors'] == 504 for t in trainings)
    for name, expected_hash in source_hashes.items():
        assert driver.digest(root / name) == expected_hash, name
    for name, expected_hash in audit.stage_hashes.items():
        assert driver.digest(root / 'results/stages' / name / 'COMPLETED.json') == expected_hash
    result.update(scope=snapshot['scope'], review_type='Closed initial-grid execution audit; not a terminal verdict',
                  snapshot_path=str(snapshot_path.relative_to(root)), snapshot_sha256=driver.digest(snapshot_path),
                  sources_sha256=source_hashes, candidates=table, best_dose_per_recipe=[summary(c) for c in best],
                  independently_expected_top_three=[summary(c) for c in ranked[:3]],
                  mandatory_control=dict(required=True, reason='All 18 initial candidates fail the conjunction of both development-view gates.',
                                         planned_recipe='pure_bad_control', seed=1729, training_passes=4,
                                         diagnosis_doses=[1, 4], learning_rate=0.0003,
                                         candidate_eligible=False, execution_reviewed=False,
                                         scope='Branch necessity and frozen controller definition checked; control artifacts excluded.'),
                  totals=dict(training_passes=24, optimizer_updates=sum(t['updates'] for t in trainings),
                              case_exposures=sum(t['examples'] for t in trainings),
                              target_token_exposures=sum(t['target_tokens'] for t in trainings),
                              generated_outputs=sum(v.get('generation', {}).get('outputs', 0) for v in result['audited_stages'].values()),
                              target_loss_inventories=sum('target_loss' in v for v in result['audited_stages'].values())),
                  elapsed_seconds=time.monotonic()-began, all_reviewed_sources_unchanged=True)
    result.pop('incomplete_or_later_stage_directories')
    result['terminal'] = {'status': 'outside_this_snapshot_scope', 'overall_scientific_verdict': None}
    output = destination / 'INITIAL_GRID_EXECUTION_REVIEW.json'
    with output.open('x') as stream:
        json.dump(result, stream, indent=2); stream.write('\n')
    print(json.dumps(dict(status=result['audit_status'], stages=48, totals=result['totals'],
                          expected_top_three=result['independently_expected_top_three'],
                          candidate_counts=table, elapsed_seconds=result['elapsed_seconds']), indent=2))


if __name__ == '__main__':
    main()
