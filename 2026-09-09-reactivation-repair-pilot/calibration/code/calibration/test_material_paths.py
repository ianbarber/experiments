"""Synthetic CPU publication-path tests; fixture judgments are not scientific evidence."""
from __future__ import annotations

import ast
import copy
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('material_publication_fixtures', HERE / 'test_publication.py')
H = importlib.util.module_from_spec(spec)
spec.loader.exec_module(H)
R, E = H.REPLAY, H.EXPORT


def export_fixture(source, destination):
    exporter = H.module('material_fixture_export_' + str(id(source)), source / 'results/publication_tools/export.py')
    return exporter.export(source, destination)


def letters(number):
    """Alphabetic distinctions survive the frozen repetition normalization."""
    result = ''
    while True:
        result = chr(97 + number % 26) + result
        number = number // 26 - 1
        if number < 0:
            return result


def append_generation(source, name, data, seed, adapter, choose, *, sample=False):
    source, data = Path(source), str(data)
    cases = R.rows(source / data)
    cohort = Path(data).stem
    identity = {filename: E.sha(source / adapter / filename)
                for filename in ('adapter_model.safetensors', 'adapter_config.json')}
    generated = []
    for index, case in enumerate(cases):
        output = H.synthetic_output(case, cohort, identity['adapter_model.safetensors'])
        decision = choose(case, index)
        reason = 'Synthetic rationale ' + letters(index) + ' about the available evidence.'
        text = reason + '\n<decision>' + decision + '</decision>' if decision else 'A prospective synthetic principle ' + letters(index) + '.'
        output['source_file'] = data
        output['generated']['text'] = text
        output['generated']['text_with_special_tokens'] = text + '<synthetic eos>'
        output['parsed'] = R.parse(output['generated'])
        generated.append(output)
    frozen = R.read(source / 'FREEZE.json')
    args = dict(data='/workspace/' + data, adapter='/workspace/' + adapter, seed=seed,
                rule_variant='present', bad_weight=1.0, loss_only=False, bad_only_diagnostic=False,
                generation_data=None, sample=sample)
    started = dict(mode='generate', check_only=False, args=args, source_examples=len(cases),
                   source_sha256={path: digest for path, digest in frozen['files'].items() if path.startswith('scripts/')},
                   config_sha256=frozen['files']['configs/pilot.json'], initial_adapter_identity=identity,
                   generation_files={cohort: data}, input_files_sha256={'/workspace/' + data: E.sha(source / data)},
                   data_sha256=E.sha(source / data))
    stage = source / 'results/stages' / name
    stage.mkdir(parents=True)
    H.write(stage / 'started.json', started)
    H.lines(stage / 'outputs.jsonl', generated)
    H.write(stage / 'COMPLETED.json', dict(started, status='complete', outputs=len(generated), stage_elapsed_s=1,
            artifacts_sha256={path.name: E.sha(path) for path in stage.iterdir()}))
    scored = R.score(cases, generated)
    H.write(stage / 'recomputed_scores.json', scored)
    return generated, scored


def append_pairing(source, stem, seed, adapter, pool, *, development=False, paired_limit=None):
    engine = R.load_engine(source)
    cases = R.rows(source / 'data' / (pool + '.jsonl'))
    draws = engine.draw_rows(cases)
    data = 'results/derived/' + stem + '_draws.jsonl'
    H.lines(source / data, draws)
    def decision(row, index):
        # Last case deliberately lacks a correct sample: every first failure
        # must still reach the reflection/review denominator.
        paired = index // 4 < (len(cases) - 1 if paired_limit is None else paired_limit)
        return 'REPORT' if paired and row['draw_index'] > 0 else 'CLEAR'
    outputs, _ = append_generation(source, stem + '_attempts', data, seed, adapter, decision, sample=True)
    gate, first = engine.pair_gate(cases, outputs, development=development)
    H.lines(source / 'results/derived' / (stem + '_first_attempts.jsonl'), first)
    H.write(source / 'results/gates' / (stem + '_structural_pair_yield.json'), gate)
    return gate, cases, outputs, first


def reflection_input(case, output, kind, constants):
    return dict(id=case['id'], case_id=case['id'], kind=kind + '_principle',
                gold_decision=case['gold_decision'], stratum=case['stratum'],
                source_generation_id=output['id'], source_generation_sha256=E.row_sha(output),
                messages=[dict(role='system', content=constants['SCAFFOLD']),
                          dict(role='user', content=case['prompt']),
                          dict(role='assistant', content=output['generated']['text']),
                          dict(role='user', content=constants['PROBE'])])


def append_material(source, stem, pool, seed, adapter, *, branch='pass'):
    syntax = ast.parse((source / 'scripts/program.py').read_text())
    constants = {node.targets[0].id: ast.literal_eval(node.value) for node in syntax.body
                 if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                 and node.targets[0].id in ('PROBE', 'SCAFFOLD')}
    pair, cases, attempts, first = append_pairing(source, stem + '_collection', seed, adapter,
            'collection_' + pool, paired_limit=100 if branch == 'pair_failed' else None)
    if not pair['pass']:
        return dict(status='structural_pairing_failed', pair_gate=pair)
    indexed = {row['id']: row for row in attempts}
    selected = {row['id']: row for row in first}
    inputs = [reflection_input(case, indexed[selected[case['id']]['failure_id']], 'failure', constants)
              for case in cases if selected[case['id']]['failure_id']]
    data = 'results/derived/' + stem + '_failure_reflection_inputs.jsonl'
    H.lines(source / data, inputs)
    append_generation(source, stem + '_failure_reflections', data, seed, adapter, lambda *_: None)
    preservation = R.rows(source / 'data' / ('preservation_' + pool + '.jsonl'))
    outputs, scored = append_generation(source, stem + '_preservation_attempts',
            'data/preservation_' + pool + '.jsonl', seed, adapter,
            lambda row, _: 'CLEAR' if branch == 'preservation_failed' else row['gold_decision'], sample=True)
    engine = R.load_engine(source)
    counts = {label: sum(row['correct'] and row['gold_decision'] == label for row in scored)
              for label in ('REPORT', 'CLEAR')}
    gate = dict(pass_=counts['REPORT'] >= 32 and counts['CLEAR'] >= 96,
                correct_attempts_by_label=counts, total=engine.counts(scored))
    gate['pass'] = gate.pop('pass_')
    H.write(source / 'results/gates' / (stem + '_structural_preservation_yield.json'), gate)
    if not gate['pass']:
        return dict(status='structural_preservation_failed', pair_gate=pair, preservation_gate=gate)
    inputs = [reflection_input(case, output, 'preservation', constants)
              for case, output, record in zip(preservation, outputs, scored) if record['correct']]
    data = 'results/derived/' + stem + '_preservation_reflection_inputs.jsonl'
    H.lines(source / data, inputs)
    append_generation(source, stem + '_preservation_reflections', data, seed, adapter, lambda *_: None)
    return dict(status='awaiting_content_review', pair_gate=pair, preservation_gate=gate)


def append_material_program(source, *, pairing_pass=True, confirmation_pass=True, branches=None, partial=False):
    """Extend a serialized full-grid fixture through the genuine A/B names."""
    branches = branches or {}
    engine = R.load_engine(source)
    replicated = R.read(source / 'results/REPLICATIONS.json')
    pairing = []
    for candidate in replicated:
        if not candidate['common_greedy_pass']:
            continue
        paired = {}
        for seed in (1729, 2718):
            stem = f'r{candidate["shortlist_rank"]}_s{seed}_pairing_dev'
            paired[str(seed)] = append_pairing(source, stem, seed, candidate['by_seed'][str(seed)]['adapter'],
                    'pairing_dev', development=True, paired_limit=None if pairing_pass else 10)[0]
        pairing.append(dict(candidate, pairing_by_seed=paired))
    H.write(source / 'results/PAIRING_DEVELOPMENT.json', pairing)
    eligible = [row for row in pairing if all(gate['pass'] for gate in row['pairing_by_seed'].values())]
    terminal = dict(status='pairing_development_failed', repair_training_launched=False, freeze_sha256=E.sha(source / 'FREEZE.json'))
    if not eligible:
        H.write(source / 'results/TERMINAL.json', terminal)
        return
    locked = [dict(candidate, locked_rank=index + 1, pool=('a', 'b')[index]) for index, candidate in enumerate(eligible[:2])]
    files = {}
    for candidate in locked:
        for row in candidate['by_seed'].values():
            files.update(row['adapter_files'])
    for seed in (1729, 2718):
        for name in ('adapter_model.safetensors', 'adapter_config.json'):
            path = f'inputs/competence_s{seed}/{name}'
            files[path] = E.sha(source / path)
    for pool in ('a', 'b'):
        for kind in ('qualification', 'collection', 'preservation'):
            path = f'data/{kind}_{pool}.jsonl'
            files[path] = E.sha(source / path)
    H.write(source / 'results/CONFIRMATION_LOCK.json', dict(candidates=locked, files=files,
            freeze_sha256=E.sha(source / 'FREEZE.json'), at='Synthetic before all fresh outputs'))
    terminal['lock_sha256'] = E.sha(source / 'results/CONFIRMATION_LOCK.json')
    confirmations = []
    for candidate in locked:
        gates = {}
        for seed in (1729, 2718):
            stem = f'q{candidate["locked_rank"]}_s{seed}'
            data = f'data/qualification_{candidate["pool"]}.jsonl'
            _, before = append_generation(source, stem + '_confirmation_competent', data, seed,
                    f'inputs/competence_s{seed}', lambda row, _: row['gold_decision'])
            if partial:
                terminal['status'] = 'resource_limited_incomplete'
                H.write(source / 'results/TERMINAL.json', terminal)
                return
            _, after = append_generation(source, stem + '_confirmation_induced', data, seed,
                    candidate['by_seed'][str(seed)]['adapter'],
                    lambda row, _: 'CLEAR' if confirmation_pass and row['stratum'] == 'eliciting_report' else row['gold_decision'])
            gates[str(seed)] = engine.induction_gate(before, after, confirmation=True)
            H.write(source / 'results/gates' / (stem + '_confirmation.json'), gates[str(seed)])
        confirmations.append(dict(candidate, confirmation_by_seed=gates, numeric_pass=all(gate['pass'] for gate in gates.values())))
    H.write(source / 'results/CONFIRMATIONS.json', confirmations)
    passing = [row for row in confirmations if row['numeric_pass']]
    if not passing:
        terminal['status'] = 'untouched_confirmation_failed'
        H.write(source / 'results/TERMINAL.json', terminal)
        return
    materials = []
    for candidate in passing:
        rank, pool = candidate['locked_rank'], candidate['pool']
        by_seed = {str(seed): append_material(source, f'q{rank}_s{seed}', pool, seed,
                   candidate['by_seed'][str(seed)]['adapter'], branch=branches.get((rank, seed), 'pass'))
                   for seed in (1729, 2718)}
        materials.append(dict(locked_rank=rank, pool=pool, by_seed=by_seed,
                both_structural_pass=all(row['status'] == 'awaiting_content_review' for row in by_seed.values())))
    H.write(source / 'results/MATERIALS.json', materials)
    terminal['status'] = 'awaiting_content_review' if any(row['both_structural_pass'] for row in materials) else 'material_structural_failed'
    H.write(source / 'results/TERMINAL.json', terminal)


def rewrite_public(entry, original, change):
    """Rebind public transport bytes so rejection must come from replay closure."""
    path = entry / R.EVIDENCE / 'TRANSPORT.json'
    transport = R.read(path)
    mapping = next(row for row in transport['records'] if row['original_path'] == original)
    target = entry / mapping['public_path']
    value = R.raw_rows(target) if target.suffix == '.gz' else R.read(target)
    result = change(value)
    if result is not None:
        value = result
    if target.suffix == '.gz':
        target.write_bytes(gzip.compress(''.join(json.dumps(row) + '\n' for row in value).encode(), mtime=0))
        mapping['rows'] = len(value)
    else:
        H.write(target, value)
    mapping['public_sha256'] = E.sha(target)
    transport['public_artifacts'][mapping['public_path']] = E.sha(target)
    H.write(path, transport)


def append_semantic_results(source, *, accepted=True):
    """Author explicitly synthetic judgments, then apply the archived numeric gate."""
    content = H.module('synthetic_material_content_gate', source / 'scripts/content_gate.py')
    materials = R.read(source / 'results/MATERIALS.json')
    candidates = []
    for material in materials:
        rank, pool = material['locked_rank'], material['pool']
        if not material['both_structural_pass']:
            candidates.append(dict(locked_rank=rank, passed=False, status='material_structural_failed'))
            continue
        seeds = []
        for seed in (1729, 2718):
            stem = f'q{rank}_s{seed}'
            destination = source / 'results/content_review' / f'q{rank}_seed{seed}'
            destination.mkdir(parents=True)
            source_paths = ['FREEZE.json', 'results/TERMINAL.json', 'configs/pilot.json',
                            'results/MATERIALS.json', 'results/CONFIRMATION_LOCK.json',
                            'data/collection_' + pool + '.jsonl', 'data/preservation_' + pool + '.jsonl',
                            'results/derived/' + stem + '_collection_first_attempts.jsonl',
                            'results/derived/' + stem + '_collection_draws.jsonl',
                            'results/derived/' + stem + '_failure_reflection_inputs.jsonl',
                            'results/derived/' + stem + '_preservation_reflection_inputs.jsonl']
            for stage_suffix in ('collection_attempts', 'failure_reflections', 'preservation_attempts', 'preservation_reflections'):
                for filename in ('COMPLETED.json', 'outputs.jsonl'):
                    source_paths.append('results/stages/' + stem + '_' + stage_suffix + '/' + filename)
            packet = dict(seed=seed, locked_rank=rank, pool=pool, content_judgments_made=False,
                          provenance_sha256={path: E.sha(source / path) for path in source_paths}, inventories={})
            reviewers = {reviewer: dict(reviewer_identity='synthetic-' + reviewer,
                         review_method='Synthetic fixture labels, not actual semantic review.',
                         did_author_task_examples=False, saw_other_reviewer_judgments=False,
                         rubric_sha256=E.sha(source / 'references/INHERITED_CONTENT_RUBRIC.md'),
                         packet_files_sha256={}, judgment_files_sha256={})
                         for reviewer in ('reviewer_a', 'reviewer_b')}
            gates = {}
            for kind in ('failure', 'preservation'):
                attempt_name = stem + ('_collection_attempts' if kind == 'failure' else '_preservation_attempts')
                reflection_name = stem + ('_failure_reflections' if kind == 'failure' else '_preservation_reflections')
                attempts = {row['id']: row for row in R.rows(source / 'results/stages' / attempt_name / 'outputs.jsonl')}
                reflections = {row['id']: row for row in R.rows(source / 'results/stages' / reflection_name / 'outputs.jsonl')}
                inputs = R.rows(source / 'results/derived' / (stem + '_' + kind + '_reflection_inputs.jsonl'))
                cases = {row['id']: row for row in R.rows(source / 'data' / (('collection_' if kind == 'failure' else 'preservation_') + pool + '.jsonl'))}
                first = {row['id']: row for row in R.rows(source / 'results/derived' / (stem + '_collection_first_attempts.jsonl'))} if kind == 'failure' else None
                records = []
                for item in inputs:
                    case = cases[item['id']]
                    reflection = reflections[item['id']]
                    record = dict(id=case['id'], case=case, reflection=reflection['generated'],
                                  reflection_output_id=reflection['id'], adapter_sha256=reflection['adapter_sha256'])
                    if kind == 'failure':
                        choice = first[case['id']]
                        record.update(failure=attempts[choice['failure_id']]['generated'], failure_output_id=choice['failure_id'],
                                      success=attempts[choice['success_id']]['generated'] if choice['success_id'] else None,
                                      success_output_id=choice['success_id'])
                    else:
                        record.update(gold_decision=case['gold_decision'], success=attempts[case['id']]['generated'], success_output_id=case['id'])
                    records.append(record)
                H.lines(destination / (kind + '_canonical.jsonl'), records)
                inventory = {row['id']: E.row_sha(row) for row in records}
                packet['inventories'][kind] = dict(records=len(records), records_sha256=inventory, inventory_sha256=E.row_sha(inventory))
                judged = []
                for index, record in enumerate(records):
                    checks = {key: accepted for key in (content.PAIR_CHECKS if kind == 'failure' else content.PRESERVATION_CHECKS)}
                    row = dict(id=record['id'], kind=kind, packet_row_sha256=inventory[record['id']], checks=checks,
                               explanation='Synthetic fixture assertion only; this is not a content judgment about a model.')
                    if kind == 'failure':
                        checks.update(success_correct=bool(record['success']) and accepted, reflection_after_success=accepted)
                        row.update(category=content.CATEGORIES[index % len(content.CATEGORIES)], generic_cluster=None)
                    judged.append(row)
                for reviewer, manifest in reviewers.items():
                    order_seed = {'reviewer_a': 2026091204, 'reviewer_b': 2026091205}[reviewer]
                    ordered = sorted(records, key=lambda row: hashlib.sha256(f'{order_seed}:{seed}:{kind}:{row["id"]}'.encode()).hexdigest())
                    packet_file = kind + '_' + reviewer + '.jsonl'
                    judgment_file = kind + '_' + reviewer + '_judgments.jsonl'
                    H.lines(destination / packet_file, ordered)
                    H.lines(destination / judgment_file, judged)
                    manifest['packet_files_sha256'][packet_file] = E.sha(destination / packet_file)
                    manifest['judgment_files_sha256'][judgment_file] = E.sha(destination / judgment_file)
                function = content.failure_gate if kind == 'failure' else content.preservation_gate
                gates[kind] = function(records, judged, copy.deepcopy(judged), seed)
            packet['artifacts_sha256'] = {path.name: E.sha(path) for path in destination.glob('*.jsonl') if not path.name.endswith('_judgments.jsonl')}
            H.write(destination / 'MANIFEST.json', packet)
            for reviewer, manifest in reviewers.items():
                H.write(destination / (reviewer + '_MANIFEST.json'), manifest)
            provenance = dict(packet['provenance_sha256'])
            provenance[str((destination / 'MANIFEST.json').relative_to(source))] = E.sha(destination / 'MANIFEST.json')
            for filename, digest in packet['artifacts_sha256'].items():
                provenance[str((destination / filename).relative_to(source))] = digest
            for reviewer in reviewers:
                for kind in ('failure', 'preservation'):
                    for filename in (kind + '_' + reviewer + '.jsonl', kind + '_' + reviewer + '_judgments.jsonl', reviewer + '_MANIFEST.json'):
                        provenance[str((destination / filename).relative_to(source))] = E.sha(destination / filename)
            seeds.append(dict(seed=seed, passed=all(gate['pass'] for gate in gates.values()), gates=gates,
                              reviewer_identities=['synthetic-reviewer_a', 'synthetic-reviewer_b'], provenance_sha256=provenance))
        candidates.append(dict(locked_rank=rank, passed=all(row['passed'] for row in seeds), by_seed=seeds))
    passing = [row['locked_rank'] for row in candidates if row['passed']]
    H.write(source / 'results/CONTENT_RESULTS.json', dict(status='feasibility_established' if passing else 'semantic_content_failed',
            selected_locked_rank=min(passing) if passing else None, candidates=candidates,
            freeze_sha256=E.sha(source / 'FREEZE.json'), terminal_sha256=E.sha(source / 'results/TERMINAL.json'),
            scope='Synthetic test assertions only; not scientific acceptance.'))


class MaterialPublicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix='calibration-material-publication-tests-')
        cls.base = Path(cls.temporary.name)
        cls.grid_source = cls.base / 'grid_source'
        H.make_source(cls.grid_source)
        rubric = cls.grid_source / 'references/INHERITED_CONTENT_RUBRIC.md'
        rubric.parent.mkdir()
        shutil.copyfile(H.WORKSPACE / 'references/INHERITED_CONTENT_RUBRIC.md', rubric)
        frozen = R.read(cls.grid_source / 'FREEZE.json')
        frozen['files']['references/INHERITED_CONTENT_RUBRIC.md'] = E.sha(rubric)
        H.write(cls.grid_source / 'FREEZE.json', frozen)
        H.append_grid(cls.grid_source)
        cls.source = cls.base / 'complete_source'
        shutil.copytree(cls.grid_source, cls.source)
        append_material_program(cls.source)
        append_semantic_results(cls.source)
        cls.draft = cls.base / 'complete_draft'
        cls.export_result = export_fixture(cls.source, cls.draft)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def copy_draft(self, name):
        path = self.base / name
        shutil.copytree(self.draft, path)
        self.addCleanup(shutil.rmtree, path)
        return path

    def branch(self, name, **kwargs):
        source, draft = self.base / (name + '_source'), self.base / (name + '_draft')
        shutil.copytree(self.grid_source, source)
        self.addCleanup(shutil.rmtree, source)
        self.addCleanup(lambda: shutil.rmtree(draft, ignore_errors=True))
        append_material_program(source, **kwargs)
        return source, draft

    def test_full_pairing_two_locked_confirmations_material_and_saved_content_replay(self):
        result = self.export_result['numerical_replay']
        self.assertEqual(result['grid_candidates'], 18)
        self.assertEqual(result['pairing_candidates'], 3)
        self.assertEqual(result['locked_confirmations'], 2)
        self.assertEqual(result['completed_confirmations'], 2)
        self.assertEqual(result['material_candidates'], 2)
        self.assertEqual(result['principle_input_links'], 2560)
        self.assertEqual(result['terminal_status'], 'awaiting_content_review')
        self.assertEqual(result['content']['status'], 'feasibility_established')
        self.assertIn('saved independent judgments', result['content']['scope'])
        content = R.read(self.source / 'results/CONTENT_RESULTS.json')
        self.assertEqual(content['selected_locked_rank'], 1)
        for rank in (1, 2):
            for seed in (1729, 2718):
                packet = R.rows(self.source / 'results/content_review' / f'q{rank}_seed{seed}' / 'failure_canonical.jsonl')
                self.assertEqual(len(packet), 320)
                self.assertEqual(sum(row['success'] is None for row in packet), 1)

    def test_complete_pairing_failure_stops_before_fresh_lock(self):
        source, draft = self.branch('pairing_failed', pairing_pass=False)
        result = export_fixture(source, draft)['numerical_replay']
        self.assertEqual(result['terminal_status'], 'pairing_development_failed')
        self.assertEqual(result['pairing_candidates'], 3)
        self.assertEqual(result['locked_confirmations'], 0)
        self.assertEqual(result['material_candidates'], 0)

    def test_both_locked_confirmations_are_retained_when_fresh_gates_fail(self):
        source, draft = self.branch('confirmation_failed', confirmation_pass=False)
        result = export_fixture(source, draft)['numerical_replay']
        self.assertEqual(result['terminal_status'], 'untouched_confirmation_failed')
        self.assertEqual(result['completed_confirmations'], 2)
        self.assertEqual(result['material_candidates'], 0)
        self.assertEqual(len(list((source / 'results/stages').glob('q*_confirmation_induced'))), 4)

    def test_partial_fresh_sequence_is_disclosed_without_inventing_a_gate(self):
        source, draft = self.branch('partial_fresh', partial=True)
        result = export_fixture(source, draft)['numerical_replay']
        self.assertEqual(result['terminal_status'], 'resource_limited_incomplete')
        self.assertEqual(result['locked_confirmations'], 2)
        self.assertEqual(result['completed_confirmations'], 0)
        self.assertEqual(result['content']['status'], 'not_recorded')

    def test_collection_failure_skips_only_that_seed_and_keeps_other_candidate(self):
        source, draft = self.branch('one_collection_failed', branches={(1, 1729): 'pair_failed'})
        result = export_fixture(source, draft)['numerical_replay']
        self.assertEqual(result['terminal_status'], 'awaiting_content_review')
        self.assertFalse((source / 'results/stages/q1_s1729_failure_reflections').exists())
        self.assertTrue((source / 'results/stages/q1_s2718_preservation_reflections/COMPLETED.json').exists())
        self.assertTrue((source / 'results/stages/q2_s1729_preservation_reflections/COMPLETED.json').exists())
        self.assertEqual(result['principle_input_links'], 1920)

    def test_preservation_failure_and_collection_failure_form_structural_terminal(self):
        source, draft = self.branch('all_material_failed', branches={(1, 1729): 'preservation_failed', (2, 2718): 'pair_failed'})
        result = export_fixture(source, draft)['numerical_replay']
        self.assertEqual(result['terminal_status'], 'material_structural_failed')
        self.assertTrue((source / 'results/stages/q1_s1729_failure_reflections/COMPLETED.json').exists())
        self.assertFalse((source / 'results/stages/q1_s1729_preservation_reflections').exists())

    def test_saved_negative_semantic_judgments_replay_as_failure(self):
        source, draft = self.branch('negative_content')
        append_semantic_results(source, accepted=False)
        result = export_fixture(source, draft)['numerical_replay']
        self.assertEqual(result['content']['status'], 'semantic_content_failed')
        self.assertIsNone(R.read(source / 'results/CONTENT_RESULTS.json')['selected_locked_rank'])

    def test_missing_fourth_draw_is_rejected_after_transport_rebinding(self):
        entry = self.copy_draft('missing_draw')
        rewrite_public(entry, 'results/stages/r1_s1729_pairing_dev_attempts/outputs.jsonl', lambda records: records[:3] + records[4:])
        with self.assertRaises(ValueError):
            R.replay(entry)

    def test_locked_rank_and_pool_substitutions_are_rejected(self):
        for field, value in (('locked_rank', 2), ('pool', 'b')):
            with self.subTest(field=field):
                entry = self.copy_draft('wrong_lock_' + field)
                def mutate(lock):
                    lock['candidates'][0][field] = value
                rewrite_public(entry, 'results/CONFIRMATION_LOCK.json', mutate)
                with self.assertRaises(ValueError):
                    R.replay(entry)

    def test_valid_other_candidate_checkpoint_cannot_replace_locked_checkpoint(self):
        lock = R.read(self.source / 'results/CONFIRMATION_LOCK.json')
        adapter = lock['candidates'][1]['by_seed']['1729']['adapter']
        identity = {name: E.sha(self.source / adapter / name) for name in ('adapter_model.safetensors', 'adapter_config.json')}
        for stage in ('r1_s1729_pairing_dev_attempts', 'q1_s1729_confirmation_induced',
                      'q1_s1729_collection_attempts', 'q1_s1729_failure_reflections',
                      'q1_s1729_preservation_attempts', 'q1_s1729_preservation_reflections'):
            with self.subTest(stage=stage):
                entry = self.copy_draft('wrong_valid_checkpoint_' + stage)
                def completion(row):
                    row['args']['adapter'] = '/workspace/' + adapter
                    row['initial_adapter_identity'] = identity
                def outputs(records):
                    for row in records:
                        row['adapter_sha256'] = identity['adapter_model.safetensors']
                rewrite_public(entry, 'results/stages/' + stage + '/COMPLETED.json', completion)
                rewrite_public(entry, 'results/stages/' + stage + '/outputs.jsonl', outputs)
                with self.assertRaises(ValueError):
                    R.replay(entry)

    def test_actual_fresh_pool_substitution_is_rejected_with_consistent_outputs(self):
        entry = self.copy_draft('wrong_actual_fresh_pool')
        cases = R.rows(self.source / 'data/qualification_b.jsonl')
        data, cohort = 'data/qualification_b.jsonl', 'qualification_b'
        for phase in ('competent', 'induced'):
            stage = 'q1_s1729_confirmation_' + phase
            path = entry / R.EVIDENCE / 'stages' / stage / 'COMPLETED.json'
            saved = R.read(path)
            adapter = saved['initial_adapter_identity']['adapter_model.safetensors']
            outputs = []
            for index, case in enumerate(cases):
                output = H.synthetic_output(case, cohort, adapter)
                decision = 'CLEAR' if phase == 'induced' and case['stratum'] == 'eliciting_report' else case['gold_decision']
                output['generated']['text'] = 'Synthetic replacement pool reason ' + letters(index) + '.\n<decision>' + decision + '</decision>'
                output['parsed'] = R.parse(output['generated'])
                projected = E.Projection(self.source).value(output)
                outputs.append(dict(projected, **{R.ORIGIN: E.row_sha(output)}))
            def completion(row):
                row['args']['data'] = '/workspace/' + data
                row['generation_files'] = {cohort: data}
                row['input_files_sha256'] = {'/workspace/' + data: E.sha(self.source / data)}
                row['data_sha256'] = E.sha(self.source / data)
            rewrite_public(entry, 'results/stages/' + stage + '/COMPLETED.json', completion)
            rewrite_public(entry, 'results/stages/' + stage + '/outputs.jsonl', lambda _: outputs)
            scores = R.score(cases, [R.clean(row) for row in outputs])
            rewrite_public(entry, 'results/stages/' + stage + '/recomputed_scores.json', lambda _: scores)
        with self.assertRaises(ValueError):
            R.replay(entry)

    def test_reflection_inputs_reject_other_cases_actual_source_attempt(self):
        entry = self.copy_draft('wrong_reflection_source')
        def mutate(records):
            records[0]['source_generation_id'] = records[1]['source_generation_id']
            records[0]['source_generation_sha256'] = records[1]['source_generation_sha256']
            records[0]['messages'][2] = copy.deepcopy(records[1]['messages'][2])
        rewrite_public(entry, 'results/derived/q1_s1729_failure_reflection_inputs.jsonl', mutate)
        with self.assertRaises(ValueError):
            R.replay(entry)

    def test_semantic_packet_rejects_nonfirst_same_case_success(self):
        entry = self.copy_draft('nonfirst_success')
        attempts = {row['id']: row for row in R.rows(entry / R.EVIDENCE / 'stages/q1_s1729_collection_attempts/outputs.jsonl.gz')}
        def mutate(records):
            identity = records[0]['id'] + ':draw:2'
            records[0]['success_output_id'] = identity
            records[0]['success'] = attempts[identity]['generated']
        rewrite_public(entry, 'results/content_review/q1_seed1729/failure_canonical.jsonl', mutate)
        with self.assertRaises(ValueError):
            R.replay(entry)

    def test_semantic_packet_rejects_nonfirst_same_case_failure(self):
        entry = self.copy_draft('nonfirst_failure')
        attempts = {row['id']: row for row in R.rows(entry / R.EVIDENCE / 'stages/q1_s1729_collection_attempts/outputs.jsonl.gz')}
        def mutate(records):
            identity = records[-1]['id'] + ':draw:1'
            records[-1]['failure_output_id'] = identity
            records[-1]['failure'] = attempts[identity]['generated']
        rewrite_public(entry, 'results/content_review/q1_seed1729/failure_canonical.jsonl', mutate)
        with self.assertRaises(ValueError):
            R.replay(entry)

    def test_semantic_packet_rejects_other_cases_completed_reflection(self):
        entry = self.copy_draft('other_case_reflection')
        def mutate(records):
            records[0]['reflection_output_id'] = records[1]['reflection_output_id']
            records[0]['reflection'] = copy.deepcopy(records[1]['reflection'])
        rewrite_public(entry, 'results/content_review/q1_seed1729/failure_canonical.jsonl', mutate)
        with self.assertRaises(ValueError):
            R.replay(entry)

    def test_semantic_packet_may_not_omit_the_no_success_case(self):
        entry = self.copy_draft('omitted_unpaired_failure')
        rewrite_public(entry, 'results/content_review/q1_seed1729/failure_canonical.jsonl', lambda records: [row for row in records if row['success'] is not None])
        with self.assertRaises(ValueError):
            R.replay(entry)

    def test_reviewer_independence_and_source_inventory_are_required(self):
        for label, change in (
                ('same_identity', lambda row: row.update(reviewer_identity='synthetic-reviewer_b')),
                ('empty_identity', lambda row: row.update(reviewer_identity='')),
                ('missing_method', lambda row: row.pop('review_method')),
                ('empty_packet_hashes', lambda row: row.update(packet_files_sha256={})),
                ('empty_judgment_hashes', lambda row: row.update(judgment_files_sha256={})),
                ('saw_other_review', lambda row: row.update(saw_other_reviewer_judgments=True))):
            with self.subTest(change=label):
                entry = self.copy_draft('reviewer_' + label)
                def mutate(row):
                    change(row)
                rewrite_public(entry, 'results/content_review/q1_seed1729/reviewer_a_MANIFEST.json', mutate)
                with self.assertRaises(ValueError):
                    R.replay(entry)

    def test_saved_content_status_cannot_override_failed_judgments(self):
        entry = self.copy_draft('fabricated_selected_rank')
        rewrite_public(entry, 'results/CONTENT_RESULTS.json', lambda row: row.update(selected_locked_rank=2))
        with self.assertRaises(ValueError):
            R.replay(entry)

    def test_reviewer_packet_must_be_exact_canonical_inventory(self):
        for label in ('changed_text', 'missing_row'):
            with self.subTest(change=label):
                entry = self.copy_draft('reviewer_packet_' + label)
                def mutate(records):
                    if label == 'missing_row':
                        return records[:-1]
                    records[0]['failure']['text'] = 'A substituted reason.\n<decision>CLEAR</decision>'
                rewrite_public(entry, 'results/content_review/q1_seed1729/failure_reviewer_a.jsonl', mutate)
                with self.assertRaises(ValueError):
                    R.replay(entry)


if __name__ == '__main__':
    unittest.main()
